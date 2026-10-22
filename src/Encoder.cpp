#include "Encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
#include <esp_heap_caps.h>
#endif

#include "CactArchive.h"
#include "FastMath.h"
#include "ModelWeights.h"
#include "RowParallel.h"

// Time-tile (frames) for the Hadamard MLP so the three nn-wide temporaries fit
// L1 (tile 4 = 3 x 8 KB). Rows are independent, so this is bit-identical.
#ifndef WHISTLE_MLP_TILE
#define WHISTLE_MLP_TILE 4
#endif

// 1 = split the Hadamard-MLP over the two P4 cores along the frame axis. Frames
// are independent, so this is bit-identical to the single-core run.
#ifndef WHISTLE_MLP_CORES
#define WHISTLE_MLP_CORES 1
#endif

// 1 = run the stem's output projection through the int8 CQ GEMV instead of a
// float matmul with dequantized weights (fewer PSRAM bytes). Changes numerics
// (activations quantized per 128-group); the two pointwise layers stay float.
#ifndef WHISTLE_STEM_OUT_INT8
#define WHISTLE_STEM_OUT_INT8 1
#endif

namespace whistle {

namespace {

constexpr float kEps = 1e-6f;

float sigmoid(float x) {
  x = (x < -30.0f) ? -30.0f : ((x > 30.0f) ? 30.0f : x);
  return 1.0f / (1.0f + fastmath::expf(-x));
}

float silu(float x) { return x * sigmoid(x); }

// Per-row fake int8 quantization, matching fake_quant_act (axis = last,
// qmax = 127, clip [-128, 127], round-half-to-even).
void fakeQuantRows(float* x, size_t rows, size_t cols) {
  const float qmax = 127.0f;
  for (size_t r = 0; r < rows; ++r) {
    float* row = x + r * cols;
    float amax = 0.0f;
    for (size_t c = 0; c < cols; ++c) {
      const float a = std::fabs(row[c]);
      if (a > amax) amax = a;
    }
    const float scale = (amax > 0.0f) ? amax / qmax : 1.0f;
    for (size_t c = 0; c < cols; ++c) {
      float v = std::nearbyintf(row[c] / scale);
      if (v < -128.0f) v = -128.0f;
      else if (v > 127.0f) v = 127.0f;
      row[c] = v * scale;
    }
  }
}

void fakeQuantQ8(const float* x, size_t rows, size_t cols, int8_t* q,
                 float* scale) {
  const float qmax = 127.0f;
  for (size_t r = 0; r < rows; ++r) {
    const float* xr = x + r * cols;
    float amax = 0.0f;
    for (size_t c = 0; c < cols; ++c) {
      const float a = std::fabs(xr[c]);
      if (a > amax) amax = a;
    }
    const float s = (amax > 0.0f) ? amax / qmax : 1.0f;
    scale[r] = s;
    int8_t* qr = q + r * cols;
    for (size_t c = 0; c < cols; ++c) {
      float v = std::nearbyintf(xr[c] / s);
      if (v < -128.0f) v = -128.0f;
      else if (v > 127.0f) v = 127.0f;
      qr[c] = static_cast<int8_t>(v);
    }
  }
}

void rmsNormRows(const float* x, size_t rows, size_t cols, const float* scale,
                 float* out) {
  for (size_t r = 0; r < rows; ++r) {
    const float* xr = x + r * cols;
    float sum = 0.0f;
    for (size_t c = 0; c < cols; ++c) sum += xr[c] * xr[c];
    const float rms = std::sqrt(sum / static_cast<float>(cols) + kEps);
    float* orow = out + r * cols;
    for (size_t c = 0; c < cols; ++c) {
      const float s = (scale != nullptr) ? 1.0f + scale[c] : 1.0f;
      orow[c] = s * xr[c] / rms;
    }
  }
}

// C[m,n] = sum_k A[m,k] * B[n,k] (B stored [out,in]). Four accumulators break
// the FPU dependency chain; per (m,n) the sum over k is re-associated into four
// partials (~1e-7 relative difference).
void matmulNT(const float* A, const float* B, float* C, size_t M, size_t N,
              size_t K) {
  // Stream B once and keep A resident. The stem's output projection is M=38,
  // N=512, K=1280: A is 195 KB and B is 2.6 MB, so the old (m outer, n inner)
  // order re-read the whole of B once per row - ~99 MB, measured at 1049 ms.
  // This order reads B once and A once per n (A stays in L2). Each output
  // element still accumulates the same products in the same order.
  for (size_t n = 0; n < N; ++n) {
    const float* br = B + n * K;
    for (size_t m = 0; m < M; ++m) {
      const float* ar = A + m * K;
      float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
      size_t k = 0;
      for (; k + 4 <= K; k += 4) {
        a0 += ar[k] * br[k];
        a1 += ar[k + 1] * br[k + 1];
        a2 += ar[k + 2] * br[k + 2];
        a3 += ar[k + 3] * br[k + 3];
      }
      float acc = (a0 + a1) + (a2 + a3);
      for (; k < K; ++k) acc += ar[k] * br[k];
      C[m * N + n] = acc;
    }
  }
}

// Position-blocked pointwise. NOTE: a version that put 8 positions in one
// weight-row pass (short runtime-bound q loop, ab[] in memory) was 6x SLOWER on
// this core - keep the simple per-position loop.
void pointwiseRows(const float* w, float* buf, uint32_t npos, uint32_t st) {
  // Block positions so the weight matrix (st*st floats) is streamed once per
  // block instead of once per position: at st=128 it is 64 KB re-read 1520 times
  // (~97 MB, measured at 379 ms for pointwise 1). Each output element still
  // accumulates k ascending, so the results are identical.
  const uint32_t kb = 32;
  std::vector<float> tmp(static_cast<size_t>(kb) * st);
  for (uint32_t p0 = 0; p0 < npos; p0 += kb) {
    const uint32_t n = (npos - p0 < kb) ? (npos - p0) : kb;
    for (uint32_t o = 0; o < st; ++o) {
      const float* wr = w + static_cast<size_t>(o) * st;
      for (uint32_t p = 0; p < n; ++p) {
        const float* v = buf + static_cast<size_t>(p0 + p) * st;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        uint32_t i = 0;
        for (; i + 4 <= st; i += 4) {
          a0 += v[i] * wr[i];
          a1 += v[i + 1] * wr[i + 1];
          a2 += v[i + 2] * wr[i + 2];
          a3 += v[i + 3] * wr[i + 3];
        }
        float acc = (a0 + a1) + (a2 + a3);
        for (; i < st; ++i) acc += v[i] * wr[i];
        tmp[static_cast<size_t>(p) * st + o] = acc;
      }
    }
    std::memcpy(buf + static_cast<size_t>(p0) * st, tmp.data(),
                static_cast<size_t>(n) * st * sizeof(float));
  }
}

void kronApply(const float* z, size_t T, const float* a, uint32_t a0, uint32_t a1,
               const float* b, uint32_t b0, uint32_t b1, float* out,
               float* scratch) {
  for (size_t t = 0; t < T; ++t) {
    const float* zt = z + t * (a0 * b0);
    float* ot = out + t * (a1 * b1);
    float* tmp = scratch;
    for (uint32_t j = 0; j < b0; ++j)
      for (uint32_t k = 0; k < a1; ++k) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < a0; ++i) acc += zt[i * b0 + j] * a[i * a1 + k];
        tmp[j * a1 + k] = acc;
      }
    for (uint32_t k = 0; k < a1; ++k)
      for (uint32_t l = 0; l < b1; ++l) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < b0; ++j) acc += tmp[j * a1 + k] * b[j * b1 + l];
        ot[k * b1 + l] = acc;
      }
  }
}

void sinkhorn(float* logK, uint32_t sz) {
  for (int it = 0; it < 20; ++it) {
    for (uint32_t i = 0; i < sz; ++i) {
      float m = -INFINITY;
      for (uint32_t j = 0; j < sz; ++j)
        if (logK[i * sz + j] > m) m = logK[i * sz + j];
      float s = 0.0f;
      for (uint32_t j = 0; j < sz; ++j) s += fastmath::expf(logK[i * sz + j] - m);
      const float lse = m + std::logf(s);
      for (uint32_t j = 0; j < sz; ++j) logK[i * sz + j] -= lse;
    }
    for (uint32_t j = 0; j < sz; ++j) {
      float m = -INFINITY;
      for (uint32_t i = 0; i < sz; ++i)
        if (logK[i * sz + j] > m) m = logK[i * sz + j];
      float s = 0.0f;
      for (uint32_t i = 0; i < sz; ++i) s += fastmath::expf(logK[i * sz + j] - m);
      const float lse = m + std::logf(s);
      for (uint32_t i = 0; i < sz; ++i) logK[i * sz + j] -= lse;
    }
  }
  for (uint32_t i = 0; i < sz * sz; ++i) logK[i] = fastmath::expf(logK[i]);
}

uint32_t nextPow2(uint32_t n) {
  uint32_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

}  // namespace

bool Encoder::readNorm(uint32_t index, float* dst) const {
  return weights_ != nullptr &&
         weights_->readIndex(index, dst, layout_.d_model);
}

void Encoder::progressStage(uint32_t layer, const char* suffix) const {
  if (progress_ == nullptr) return;
  char tag[16];
  std::snprintf(tag, sizeof(tag), "e%u%s", (unsigned)layer, suffix);
  progress_(progress_udata_, tag);
}

bool Encoder::dense(uint32_t index, const float* x, size_t M, uint32_t in,
                    uint32_t out_count, float* y) const {
  const size_t n = M * in;
  if (qScratch_.size() < n) qScratch_.resize(n);
  if (scaleScratch_.size() < M) scaleScratch_.resize(M);
  fakeQuantQ8(x, M, in, qScratch_.data(), scaleScratch_.data());
  return weights_ != nullptr &&
         weights_->gemvCqInt8Index(index, 0, out_count, qScratch_.data(),
                                   scaleScratch_.data(), M, y);
}

bool Encoder::prepare(const cactus::CactArchive& archive,
                      const ModelWeights& weights) {
  ready_ = false;
  if (!findAudioLayout(archive, &layout_) || !layout_.present) return false;
  weights_ = &weights;

  const uint32_t d = layout_.d_model;
  const uint32_t mels = layout_.mels;
  const uint32_t st = layout_.stem;
  const uint32_t taps = layout_.conv_kernel;

  MelConfig mcfg;
  mcfg.n_mels = mels;
  mcfg.n_fft = layout_.fft;
  mcfg.win_length = layout_.win;
  mcfg.hop = layout_.hop;
  mcfg.sample_rate = layout_.rate;
  filterbank_.resize(static_cast<size_t>(layout_.fft / 2 + 1) * mels);
  if (!weights.readIndex(layout_.mel, filterbank_.data(), filterbank_.size()))
    return false;
  if (!mel_.init(mcfg, filterbank_.data())) return false;
  // The mel keeps a packed, support-trimmed copy; the dense bank is not needed.
  filterbank_.clear();
  filterbank_.shrink_to_fit();

  stem_w_.resize(static_cast<size_t>(taps) * st);
  stem_dw1_.resize(static_cast<size_t>(taps) * st);
  stem_dw2_.resize(static_cast<size_t>(taps) * st);
  stem_pw1_.resize(static_cast<size_t>(st) * st);
  stem_pw2_.resize(static_cast<size_t>(st) * st);
  const uint32_t stem_out_in = st * (mels / 8);
  stem_out_.resize(static_cast<size_t>(d) * stem_out_in);
  if (!weights.readIndex(layout_.stem_w, stem_w_.data(), stem_w_.size()) ||
      !weights.readIndex(layout_.stem_dw1, stem_dw1_.data(), stem_dw1_.size()) ||
      !weights.readIndex(layout_.stem_dw2, stem_dw2_.data(), stem_dw2_.size()) ||
      !weights.readIndex(layout_.stem_pw1, stem_pw1_.data(), stem_pw1_.size()) ||
      !weights.readIndex(layout_.stem_pw2, stem_pw2_.data(), stem_pw2_.size()) ||
      !weights.readIndex(layout_.stem_out, stem_out_.data(), stem_out_.size()))
    return false;

  const uint32_t L = layout_.enc_layers;
  const uint32_t n = layout_.enc_lanes;
  mhc_a_pre_.resize(L);
  mhc_a_post_.resize(L);
  mhc_a_res_.resize(L);
  mhc_b_pre_.resize(static_cast<size_t>(L) * n);
  mhc_b_post_.resize(static_cast<size_t>(L) * n);
  mhc_b_res_.resize(static_cast<size_t>(L) * n * n);
  final_norm_.resize(d);
  if (!weights.readIndex(layout_.enc_mhc_base + 0, mhc_a_pre_.data(), L) ||
      !weights.readIndex(layout_.enc_mhc_base + 1, mhc_a_post_.data(), L) ||
      !weights.readIndex(layout_.enc_mhc_base + 2, mhc_a_res_.data(), L) ||
      !weights.readIndex(layout_.enc_mhc_base + 3, mhc_b_pre_.data(),
                         static_cast<size_t>(L) * n) ||
      !weights.readIndex(layout_.enc_mhc_base + 4, mhc_b_post_.data(),
                         static_cast<size_t>(L) * n) ||
      !weights.readIndex(layout_.enc_mhc_base + 5, mhc_b_res_.data(),
                         static_cast<size_t>(L) * n * n) ||
      !weights.readIndex(layout_.enc_mhc_base + 9, final_norm_.data(), d))
    return false;

  const uint32_t nn = nextPow2(d);
  std::vector<float> p1(nn), p2(nn);
  if (!weights.readTensor("hada_p1", p1.data(), nn) ||
      !weights.readTensor("hada_p2", p2.data(), nn))
    return false;
  perm1_.resize(nn);
  perm2_.resize(nn);
  for (uint32_t i = 0; i < nn; ++i) {
    perm1_[i] = static_cast<int32_t>(p1[i]);
    perm2_[i] = static_cast<int32_t>(p2[i]);
  }

  // Pre-load the Hadamard-MLP tensors (~66 KB per block/half, constant): doing it
  // here keeps the first utterance from paying ~1.8 s of reads plus 1 MB of
  // allocations that also reshape the heap. hadamardMlp() keeps a lazy path as a
  // fallback if this fails.
  for (uint32_t l = 0; l < L; ++l)
    for (uint32_t w = 0; w < 2; ++w) (void)ensureMlpWeights(l, w);

  ready_ = true;
  return true;
}

bool Encoder::stem(const float* mel, size_t mel_frames, size_t valid_mel,
                   std::vector<float>* out, size_t* out_frames,
                   size_t* valid_out) {
  const uint32_t d = layout_.d_model;
  const uint32_t st = layout_.stem;
  const uint32_t f0 = layout_.mels;
  const uint32_t f1 = (f0 + 1) / 2;
  const uint32_t f2 = (f1 + 1) / 2;
  const uint32_t f3 = (f2 + 1) / 2;
  const uint32_t t0 = static_cast<uint32_t>(mel_frames);
  const uint32_t t1 = (t0 + 1) / 2;
  const uint32_t t2 = (t1 + 1) / 2;
  const uint32_t t3 = (t2 + 1) / 2;
  const uint32_t l1 = static_cast<uint32_t>((valid_mel + 1) / 2);
  const uint32_t l2 = (l1 + 1) / 2;
  const uint32_t l3 = (l2 + 1) / 2;
  const uint32_t ks = 3;  // stem spatial kernel; conv_kernel=9 is the tap count

  // Stage tags: the stem is 23% of the run and ~16x its FPU cost, so the split
  // matters before touching it.
  auto stg = [&](const char* tag) {
    if (progress_ != nullptr) progress_(progress_udata_, tag);
  };

  // Stage 0: 1 -> stem channels, regular 3x3 stride 2.
  std::vector<float> s0(static_cast<size_t>(t1) * f1 * st, 0.0f);
  for (uint32_t t = 0; t < t1; ++t) {
    if (t >= l1) continue;
    const int64_t it0 = static_cast<int64_t>(t) * 2 - 1;
    for (uint32_t f = 0; f < f1; ++f) {
      const int64_t jf0 = static_cast<int64_t>(f) * 2 - 1;
      float* dst = s0.data() + (static_cast<size_t>(t) * f1 + f) * st;
      for (uint32_t oc = 0; oc < st; ++oc) dst[oc] = 0.0f;
      for (uint32_t kh = 0; kh < ks; ++kh) {
        const int64_t it = it0 + kh;
        if (it < 0 || it >= static_cast<int64_t>(t0)) continue;
        for (uint32_t kw = 0; kw < ks; ++kw) {
          const int64_t jf = jf0 + kw;
          if (jf < 0 || jf >= static_cast<int64_t>(f0)) continue;
          const float x = mel[static_cast<size_t>(it) * f0 + jf];
          const float* w = stem_w_.data() + (kh * ks + kw) * st;
          for (uint32_t oc = 0; oc < st; ++oc) dst[oc] += x * w[oc];
        }
      }
    }
  }

  stg("st_s0");
  // silu once per element: the 3x3 taps below read each source element up to
  // nine times and would otherwise recompute silu (one expf) every time. s0 is
  // not used after this stage.
  for (size_t i = 0; i < s0.size(); ++i) s0[i] = silu(s0[i]);
  stg("st_q0");

  // Stage 1: silu -> depthwise 3x3 stride 2 -> pointwise.
  std::vector<float> s1(static_cast<size_t>(t2) * f2 * st, 0.0f);
  for (uint32_t t = 0; t < t2; ++t) {
    if (t >= l2) continue;
    const int64_t it0 = static_cast<int64_t>(t) * 2 - 1;
    for (uint32_t f = 0; f < f2; ++f) {
      const int64_t jf0 = static_cast<int64_t>(f) * 2 - 1;
      float* dst = s1.data() + (static_cast<size_t>(t) * f2 + f) * st;
      float acc[128];
      for (uint32_t c = 0; c < st; ++c) acc[c] = 0.0f;
      for (uint32_t kh = 0; kh < ks; ++kh) {
        const int64_t it = it0 + kh;
        if (it < 0 || it >= static_cast<int64_t>(t1)) continue;
        for (uint32_t kw = 0; kw < ks; ++kw) {
          const int64_t jf = jf0 + kw;
          if (jf < 0 || jf >= static_cast<int64_t>(f1)) continue;
          const float* src = s0.data() + (static_cast<size_t>(it) * f1 + jf) * st;
          const float* w = stem_dw1_.data() + (kh * ks + kw) * st;
          for (uint32_t c = 0; c < st; ++c) acc[c] += src[c] * w[c];
        }
      }
      std::memcpy(dst, acc, st * sizeof(float));
    }
  }
  stg("st_s1");
  pointwiseRows(stem_pw1_.data(), s1.data(), t2 * f2, st);
  stg("st_p1");
  for (size_t i = 0; i < s1.size(); ++i) s1[i] = silu(s1[i]);
  stg("st_q1");

  // Stage 2: silu -> depthwise 3x3 stride 2 -> pointwise.
  std::vector<float> s2(static_cast<size_t>(t3) * f3 * st, 0.0f);
  for (uint32_t t = 0; t < t3; ++t) {
    if (t >= l3) continue;
    const int64_t it0 = static_cast<int64_t>(t) * 2 - 1;
    for (uint32_t f = 0; f < f3; ++f) {
      const int64_t jf0 = static_cast<int64_t>(f) * 2 - 1;
      float* dst = s2.data() + (static_cast<size_t>(t) * f3 + f) * st;
      float acc[128];
      for (uint32_t c = 0; c < st; ++c) acc[c] = 0.0f;
      for (uint32_t kh = 0; kh < ks; ++kh) {
        const int64_t it = it0 + kh;
        if (it < 0 || it >= static_cast<int64_t>(t2)) continue;
        for (uint32_t kw = 0; kw < ks; ++kw) {
          const int64_t jf = jf0 + kw;
          if (jf < 0 || jf >= static_cast<int64_t>(f2)) continue;
          const float* src = s1.data() + (static_cast<size_t>(it) * f2 + jf) * st;
          const float* w = stem_dw2_.data() + (kh * ks + kw) * st;
          for (uint32_t c = 0; c < st; ++c) acc[c] += src[c] * w[c];
        }
      }
      std::memcpy(dst, acc, st * sizeof(float));
    }
  }
  stg("st_s2");
  pointwiseRows(stem_pw2_.data(), s2.data(), t3 * f3, st);
  stg("st_p2");

  // silu, swap channel/frequency to [T][C][F], flatten C*F per frame.
  const uint32_t flat_in = st * f3;
  std::vector<float> flat(static_cast<size_t>(t3) * flat_in, 0.0f);
  for (uint32_t t = 0; t < t3; ++t)
    for (uint32_t c = 0; c < st; ++c)
      for (uint32_t f = 0; f < f3; ++f) {
        const float v = s2[(static_cast<size_t>(t) * f3 + f) * st + c];
        flat[static_cast<size_t>(t) * flat_in + c * f3 + f] = silu(v);
      }
  fakeQuantRows(flat.data(), t3, flat_in);
  stg("st_fl");

  out->assign(static_cast<size_t>(t3) * d, 0.0f);
#if WHISTLE_STEM_OUT_INT8
  if (!dense(layout_.stem_out, flat.data(), t3, flat_in, d, out->data()))
    return false;
#else
  matmulNT(flat.data(), stem_out_.data(), out->data(), t3, d, flat_in);
#endif
  stg("st_mm");
  if (debug_) debug_(debug_udata_, "stem", out->data(), out->size());
  *out_frames = t3;
  *valid_out = l3;
  return true;
}

bool Encoder::stemDebug(const float* mel, size_t mel_frames, size_t valid_mel,
                        std::vector<float>* out, size_t* out_frames,
                        size_t* valid_out) {
  if (!ready_) return false;
  return stem(mel, mel_frames, valid_mel, out, out_frames, valid_out);
}

bool Encoder::encode(const float* pcm, size_t num_samples,
                     std::vector<float>* mem, size_t* frames) {
  if (!ready_ || pcm == nullptr || mem == nullptr || frames == nullptr)
    return false;
  const size_t mel_frames = mel_.frames(num_samples);
  const size_t valid_mel = mel_.validFrames(num_samples);
  std::vector<float> melbuf(mel_frames * layout_.mels);
  mel_.setProgress(progress_, progress_udata_);
  mel_.compute(pcm, num_samples, melbuf.data());
  if (progress_) progress_(progress_udata_, "mel");

  std::vector<float> stem_out;
  size_t stem_frames = 0, valid_stem = 0;
  if (!stem(melbuf.data(), mel_frames, valid_mel, &stem_out, &stem_frames,
            &valid_stem))
    return false;
  if (progress_) progress_(progress_udata_, "stem");

  encoderForward(stem_out.data(), stem_frames, valid_stem, mem);
  *frames = valid_stem;
  return true;
}

void Encoder::encoderForward(const float* x, size_t frames, size_t valid,
                             std::vector<float>* out) {
  const uint32_t d = layout_.d_model;
  const uint32_t n = layout_.enc_lanes;
  const uint32_t L = layout_.enc_layers;
  const uint32_t nC = n * d;
  const size_t T = frames + (frames & 1);  // encode() pads time to even

  std::vector<float> stream(T * n * d, 0.0f);
  for (size_t t = 0; t < frames; ++t)
    for (uint32_t i = 0; i < n; ++i)
      for (uint32_t c = 0; c < d; ++c)
        stream[(t * n + i) * d + c] = x[t * d + c];

  std::vector<float> seen(T, 0.0f);
  for (size_t t = 0; t < valid && t < T; ++t) seen[t] = 1.0f;

  std::vector<float> nx(T * nC, 0.0f), hpre(T * n, 0.0f), hpost(T * n, 0.0f);
  std::vector<int8_t> nx8(T * nC, 0);
  std::vector<float> nx_scale(T, 0.0f);
  std::vector<float> hres(T * n * n, 0.0f), u(T * d, 0.0f), y(T * d, 0.0f);
  std::vector<float> blk(T * d, 0.0f), res(T * n * n, 0.0f);
  std::vector<float> kkmat(n * n, 0.0f), next(T * n * d, 0.0f);

  for (uint32_t layer = 0; layer < L; ++layer) {
    rmsNormRows(stream.data(), T, nC, nullptr, nx.data());
    fakeQuantRows(nx.data(), T, nC);
    fakeQuantQ8(nx.data(), T, nC, nx8.data(), nx_scale.data());

    if (!weights_->gemvCqInt8Index(layout_.enc_mhc_base + 6, layer * n, n,
                                   nx8.data(), nx_scale.data(), T,
                                   hpre.data()))
      return;
    for (size_t t = 0; t < T; ++t)
      for (uint32_t i = 0; i < n; ++i) {
        const float lane = ((layer % n) == i) ? 1.0f : 0.0f;
        const float pre_off = 8.0f * lane - 4.0f;
        hpre[t * n + i] =
            sigmoid(mhc_a_pre_[layer] * hpre[t * n + i] +
                    mhc_b_pre_[layer * n + i] + pre_off);
      }

    for (size_t t = 0; t < T; ++t)
      for (uint32_t c = 0; c < d; ++c) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < n; ++i)
          acc += hpre[t * n + i] * stream[(t * n + i) * d + c];
        u[t * d + c] = acc;
      }

    encoderBlock(u.data(), T, seen.data(), layer, blk.data());
    for (size_t i = 0; i < T * d; ++i) y[i] = blk[i] - u[i];

    if (!weights_->gemvCqInt8Index(layout_.enc_mhc_base + 7, layer * n, n,
                                   nx8.data(), nx_scale.data(), T,
                                   hpost.data()))
      return;
    for (size_t t = 0; t < T; ++t)
      for (uint32_t i = 0; i < n; ++i) {
        const float lane = ((layer % n) == i) ? 1.0f : 0.0f;
        const float post_off = -4.0f * (1.0f - lane);
        hpost[t * n + i] =
            2.0f * sigmoid(mhc_a_post_[layer] * hpost[t * n + i] +
                           mhc_b_post_[layer * n + i] + post_off);
      }

    if (!weights_->gemvCqInt8Index(layout_.enc_mhc_base + 8, layer * n * n,
                                   n * n, nx8.data(), nx_scale.data(), T,
                                   res.data()))
      return;
    for (size_t t = 0; t < T; ++t) {
      for (uint32_t i = 0; i < n * n; ++i)
        kkmat[i] = mhc_a_res_[layer] * res[t * n * n + i] +
                   mhc_b_res_[layer * n * n + i];
      sinkhorn(kkmat.data(), n);
      std::memcpy(hres.data() + t * n * n, kkmat.data(), n * n * sizeof(float));
    }

    for (size_t t = 0; t < T; ++t)
      for (uint32_t i = 0; i < n; ++i)
        for (uint32_t c = 0; c < d; ++c) {
          float acc = 0.0f;
          for (uint32_t j = 0; j < n; ++j)
            acc += hres[(t * n + i) * n + j] * stream[(t * n + j) * d + c];
          acc += hpost[t * n + i] * y[t * d + c];
          next[(t * n + i) * d + c] = acc;
        }
    std::memcpy(stream.data(), next.data(), T * n * d * sizeof(float));
    progressStage(layer, "mh");
    if (debug_) {
      char tag[8];
      std::snprintf(tag, sizeof(tag), "l%u", layer);
      debug_(debug_udata_, tag, stream.data(), stream.size());
    }
    if (progress_) {
      char tag[16];
      std::snprintf(tag, sizeof(tag), "enc_l%u", layer);
      progress_(progress_udata_, tag);
    }
  }

  std::vector<float> fx(T * d, 0.0f);
  for (size_t t = 0; t < T; ++t)
    for (uint32_t c = 0; c < d; ++c) {
      float acc = 0.0f;
      for (uint32_t i = 0; i < n; ++i) acc += stream[(t * n + i) * d + c];
      fx[t * d + c] = acc / static_cast<float>(n);
    }
  if (debug_) debug_(debug_udata_, "pre", fx.data(), valid * d);
  rmsNormRows(fx.data(), T, d, final_norm_.data(), fx.data());

  out->assign(valid * d, 0.0f);
  std::memcpy(out->data(), fx.data(), valid * d * sizeof(float));
  if (debug_) debug_(debug_udata_, "out", out->data(), out->size());
}


void Encoder::encoderBlock(const float* x, size_t T, const float* seen,
                           uint32_t layer, float* out) {
  const uint32_t d = layout_.d_model;
  const uint32_t k = layout_.conv_kernel;
  auto ensure = [](std::vector<float>& v, size_t n) {
    if (v.size() != n) v.assign(n, 0.0f);
  };
  ensure(scratch_.b_hn, T * d);
  ensure(scratch_.b_tmp, T * d);
  ensure(scratch_.b_at, T * d);
  ensure(scratch_.b_pw1, T * 2 * d);
  ensure(scratch_.b_h, T * d);
  ensure(scratch_.b_hconv, T * d);
  ensure(scratch_.b_y, T * d);
  ensure(scratch_.b_dw, static_cast<size_t>(k) * d);
  ensure(scratch_.b_nscale, d);
  auto& hn = scratch_.b_hn;
  auto& tmp = scratch_.b_tmp;
  auto& at = scratch_.b_at;
  auto& pw1 = scratch_.b_pw1;
  auto& h = scratch_.b_h;
  auto& hconv = scratch_.b_hconv;
  auto& y = scratch_.b_y;
  auto& dw = scratch_.b_dw;
  auto& nscale = scratch_.b_nscale;
  if (layer == 0) progressStage(layer, "blk_b");
  auto norm = [&](const float* src, uint32_t rel, float* dst) -> bool {
    if (!readNorm(layout_.encIndex(layer, rel), nscale.data())) return false;
    rmsNormRows(src, T, d, nscale.data(), dst);
    return true;
  };
  float gate_scalar = 0.0f;
  if (!weights_->readIndex(layout_.encIndex(layer, 13), &gate_scalar, 1))
    return;
  if (!weights_->readIndex(layout_.encIndex(layer, 16), dw.data(), dw.size()))
    return;

  // x += 0.5 * hadamard_mlp_0(norm_pre_hada_norm_0(x))
  if (!norm(x, 0, hn.data())) return;
  if (layer == 0) progressStage(layer, "blk_n");
  hadamardMlp(hn.data(), T, layer, 0, tmp.data());
  for (size_t i = 0; i < T * d; ++i) out[i] = x[i] + 0.5f * tmp[i];
  progressStage(layer, "m0");
  if (debug_) {
    char tag[16];
    std::snprintf(tag, sizeof(tag), "e%um0", layer);
    debug_(debug_udata_, tag, out, T * d);
  }

  // Gated self attention.
  if (!norm(out, 1, hn.data())) return;
  selfAttention(hn.data(), T, seen, layer, at.data());
  if (!norm(at.data(), 2, hn.data())) return;
  const float ag = sigmoid(gate_scalar);
  for (size_t i = 0; i < T * d; ++i) out[i] += ag * hn[i];
  progressStage(layer, "at");
  if (debug_) {
    char tag[16];
    std::snprintf(tag, sizeof(tag), "e%uat", layer);
    debug_(debug_udata_, tag, out, T * d);
  }

  // Gated depthwise convolution + GLU.
  if (!norm(out, 3, hn.data())) return;
  fakeQuantRows(hn.data(), T, d);
  if (!dense(layout_.encIndex(layer, 14), hn.data(), T, d, 2 * d,
             pw1.data()))
    return;
  for (size_t t = 0; t < T; ++t)
    for (uint32_t c = 0; c < d; ++c) {
      const float a = pw1[t * 2 * d + c];
      const float b = pw1[t * 2 * d + d + c];
      h[t * d + c] = a * sigmoid(b) * seen[t];
    }
  const int32_t pad = static_cast<int32_t>(k / 2);
  for (size_t t = 0; t < T; ++t)
    for (uint32_t c = 0; c < d; ++c) {
      float acc = 0.0f;
      for (uint32_t j = 0; j < k; ++j) {
        const int64_t s = static_cast<int64_t>(t) + j - pad;
        if (s < 0 || s >= static_cast<int64_t>(T)) continue;
        acc += h[static_cast<size_t>(s) * d + c] * dw[j * d + c];
      }
      hconv[t * d + c] = acc;
    }
  if (!norm(hconv.data(), 4, hn.data())) return;
  for (size_t i = 0; i < T * d; ++i) hn[i] = silu(hn[i]);
  fakeQuantRows(hn.data(), T, d);
  if (!dense(layout_.encIndex(layer, 15), hn.data(), T, d, d, y.data())) return;
  for (size_t i = 0; i < T * d; ++i) out[i] += y[i];
  progressStage(layer, "cv");
  if (debug_) {
    char tag[16];
    std::snprintf(tag, sizeof(tag), "e%ucv", layer);
    debug_(debug_udata_, tag, out, T * d);
  }

  // x += 0.5 * hadamard_mlp(norm_pre_hada_norm(x))
  if (!norm(out, 5, hn.data())) return;
  if (layer + 1 == layout_.enc_layers) progressStage(layer, "m1n");
  hadamardMlp(hn.data(), T, layer, 1, tmp.data());
  for (size_t i = 0; i < T * d; ++i) out[i] += 0.5f * tmp[i];
  progressStage(layer, "m1");
  if (debug_) {
    char tag[16];
    std::snprintf(tag, sizeof(tag), "e%um1", layer);
    debug_(debug_udata_, tag, out, T * d);
  }
}


void Encoder::selfAttention(const float* x, size_t T, const float* seen,
                            uint32_t layer, float* out) {
  const ModelGeometry& g = weights_->geometry();
  const uint32_t d = layout_.d_model;
  const uint32_t H = g.num_heads;
  const uint32_t KV = g.num_kv_heads;
  const uint32_t qk = g.qk_head_dim;
  const uint32_t vd = g.v_head_dim;
  const uint32_t rep = H / KV;
  const uint32_t half = qk / 2;

  auto& xq = scratch_.xq;
  xq.resize(T * d);
  std::memcpy(xq.data(), x, static_cast<size_t>(T) * d * sizeof(float));
  fakeQuantRows(xq.data(), T, d);
  auto& q = scratch_.q;
  q.resize(T * H * qk);
  auto& k = scratch_.k;
  k.resize(T * KV * qk);
  auto& v = scratch_.v;
  v.resize(T * KV * vd);
  auto& gate = scratch_.gate;
  gate.resize(T * d);
  auto& tmp = scratch_.satmp;
  tmp.resize(T * d);
  if (!dense(layout_.encIndex(layer, 6), xq.data(), T, d, H * qk, q.data()))
    return;
  if (!dense(layout_.encIndex(layer, 7), xq.data(), T, d, KV * qk, k.data()))
    return;
  if (!dense(layout_.encIndex(layer, 8), xq.data(), T, d, KV * vd, v.data()))
    return;
  progressStage(layer, "qkv");

  auto& qn = scratch_.qn;
  qn.resize(qk);
  auto& kn = scratch_.kn;
  kn.resize(qk);
  if (!weights_->readIndex(layout_.encIndex(layer, 11), qn.data(), qk) ||
      !weights_->readIndex(layout_.encIndex(layer, 12), kn.data(), qk))
    return;

  // RoPE tables: theta^(-2j/qk), j = 0..half-1. They depend only on T and theta,
  // so they are built once and shared by every block (and by later utterances
  // of the same length) instead of T*half sin/cos per layer.
  const float theta = g.rope_theta;
  auto& cos_t = scratch_.cos_t;
  auto& sin_t = scratch_.sin_t;
  if (scratch_.rope_frames != T || cos_t.size() != T * half ||
      sin_t.size() != T * half) {
    cos_t.resize(T * half);
    sin_t.resize(T * half);
    for (size_t t = 0; t < T; ++t)
      for (uint32_t j = 0; j < half; ++j) {
        const float freq =
            1.0f / std::pow(theta, (2.0f * j) / static_cast<float>(qk));
        const float a = static_cast<float>(t) * freq;
        cos_t[t * half + j] = std::cos(a);
        sin_t[t * half + j] = std::sin(a);
      }
    scratch_.rope_frames = T;
  }

  for (size_t t = 0; t < T; ++t) {
    for (uint32_t h = 0; h < H; ++h) {
      float* row = q.data() + (t * H + h) * qk;
      float sum = 0.0f;
      for (uint32_t c = 0; c < qk; ++c) sum += row[c] * row[c];
      const float rms = std::sqrt(sum / static_cast<float>(qk) + kEps);
      for (uint32_t c = 0; c < qk; ++c)
        row[c] = (1.0f + qn[c]) * row[c] / rms;
      for (uint32_t c = 0; c < half; ++c) {
        const float cv = cos_t[t * half + c];
        const float sv = sin_t[t * half + c];
        const float x1 = row[c], x2 = row[c + half];
        row[c] = x1 * cv - x2 * sv;
        row[c + half] = x2 * cv + x1 * sv;
      }
      fakeQuantRows(row, 1, qk);
    }
    for (uint32_t h = 0; h < KV; ++h) {
      float* row = k.data() + (t * KV + h) * qk;
      float sum = 0.0f;
      for (uint32_t c = 0; c < qk; ++c) sum += row[c] * row[c];
      const float rms = std::sqrt(sum / static_cast<float>(qk) + kEps);
      for (uint32_t c = 0; c < qk; ++c)
        row[c] = (1.0f + kn[c]) * row[c] / rms;
      for (uint32_t c = 0; c < half; ++c) {
        const float cv = cos_t[t * half + c];
        const float sv = sin_t[t * half + c];
        const float x1 = row[c], x2 = row[c + half];
        row[c] = x1 * cv - x2 * sv;
        row[c + half] = x2 * cv + x1 * sv;
      }
      fakeQuantRows(row, 1, qk);
    }
    for (uint32_t h = 0; h < KV; ++h)
      fakeQuantRows(v.data() + (t * KV + h) * vd, 1, vd);
  }
  progressStage(layer, "rope");

  const float invSqrtQk = 1.0f / std::sqrt(static_cast<float>(qk));
  auto& scores = scratch_.scores;
  scores.resize(T);
  auto& attn = scratch_.attn;
  attn.resize(T);
  for (uint32_t h = 0; h < H; ++h) {
    const uint32_t kvh = h / rep;
    for (size_t t = 0; t < T; ++t) {
      const float* qrow = q.data() + (t * H + h) * qk;
      float maxs = -INFINITY;
      for (size_t s = 0; s < T; ++s) {
        if (seen[s] == 0.0f) {
          scores[s] = -INFINITY;
          continue;
        }
        const float* krow = k.data() + (s * KV + kvh) * qk;
        float acc = 0.0f;
        for (uint32_t c = 0; c < qk; ++c) acc += qrow[c] * krow[c];
        const float val = acc * invSqrtQk;
        scores[s] = val;
        if (val > maxs) maxs = val;
      }
      float sum = 0.0f;
      for (size_t s = 0; s < T; ++s) {
        if (scores[s] == -INFINITY) {
          attn[s] = 0.0f;
        } else {
          attn[s] = fastmath::expf(scores[s] - maxs);
          sum += attn[s];
        }
      }
      if (sum <= 0.0f) {
        for (size_t s = 0; s < T; ++s) attn[s] = 0.0f;
      } else {
        for (size_t s = 0; s < T; ++s) attn[s] /= sum;
      }
      // Rank-1 accumulate (s outer): each V row is read once contiguously per
      // (head, query) instead of re-striding vd floats for every output tap.
      // The per-output accumulation order over s is unchanged.
      float* oh = tmp.data() + t * (H * vd) + h * vd;
      for (uint32_t c = 0; c < vd; ++c) oh[c] = 0.0f;
      for (size_t s = 0; s < T; ++s) {
        const float w = attn[s];
        const float* vrow = v.data() + (s * KV + kvh) * vd;
        for (uint32_t c = 0; c < vd; ++c) oh[c] += w * vrow[c];
      }
    }
  }

  progressStage(layer, "sc");
  if (!dense(layout_.encIndex(layer, 9), xq.data(), T, d, d, gate.data()))
    return;
  for (size_t i = 0; i < T * d; ++i) tmp[i] *= sigmoid(gate[i]);
  fakeQuantRows(tmp.data(), T, d);
  if (!dense(layout_.encIndex(layer, 10), tmp.data(), T, d, d, out)) return;
  progressStage(layer, "o");
}


bool Encoder::ensureMlpWeights(uint32_t layer, uint32_t which) {
  const uint32_t d = layout_.d_model;
  const uint32_t nn = nextPow2(d);
  const uint32_t base = layout_.encIndex(layer, 17 + which * 13);
  const size_t mi = static_cast<size_t>(layer) * 2 + which;
  if (mlpW_.size() <= mi)
    mlpW_.resize(static_cast<size_t>(layout_.enc_layers) * 2);
  MlpWeights& mw = mlpW_[mi];
  if (mw.loaded) return true;

  const cactus::TensorMeta* m1a = weights_->metaAt(base + 5);
  const cactus::TensorMeta* m1b = weights_->metaAt(base + 6);
  const cactus::TensorMeta* m2a = weights_->metaAt(base + 7);
  const cactus::TensorMeta* m2b = weights_->metaAt(base + 8);
  const cactus::TensorMeta* m3a = weights_->metaAt(base + 9);
  const cactus::TensorMeta* m3b = weights_->metaAt(base + 10);
  if (m1a == nullptr || m1b == nullptr || m2a == nullptr ||
      m2b == nullptr || m3a == nullptr || m3b == nullptr)
    return false;
  mw.w1a0 = m1a->shape[0];
  mw.w1a1 = m1a->shape[1];
  mw.w1b0 = m1b->shape[0];
  mw.w1b1 = m1b->shape[1];
  mw.w2a0 = m2a->shape[0];
  mw.w2a1 = m2a->shape[1];
  mw.w2b0 = m2b->shape[0];
  mw.w2b1 = m2b->shape[1];
  mw.w3a0 = m3a->shape[0];
  mw.w3a1 = m3a->shape[1];
  mw.w3b0 = m3b->shape[0];
  mw.w3b1 = m3b->shape[1];
  if (mw.w1a0 * mw.w1b0 != nn || mw.w2a0 * mw.w2b0 != nn ||
      mw.w3a0 * mw.w3b0 != nn)
    return false;

  mw.d1.resize(nn);
  mw.d2.resize(nn);
  mw.b2.resize(nn);
  mw.d3.resize(nn);
  mw.d4.resize(nn);
  mw.w1a.resize(static_cast<size_t>(mw.w1a0) * mw.w1a1);
  mw.w1b.resize(static_cast<size_t>(mw.w1b0) * mw.w1b1);
  mw.w2a.resize(static_cast<size_t>(mw.w2a0) * mw.w2a1);
  mw.w2b.resize(static_cast<size_t>(mw.w2b0) * mw.w2b1);
  mw.w3a.resize(static_cast<size_t>(mw.w3a0) * mw.w3a1);
  mw.w3b.resize(static_cast<size_t>(mw.w3b0) * mw.w3b1);
  mw.cond_v.resize(static_cast<size_t>(d) * 8);
  mw.cond_u.resize(static_cast<size_t>(8) * nn);
  if (!weights_->readIndex(base + 0, mw.d1.data(), nn) ||
      !weights_->readIndex(base + 1, mw.d2.data(), nn) ||
      !weights_->readIndex(base + 2, mw.b2.data(), nn) ||
      !weights_->readIndex(base + 3, mw.d3.data(), nn) ||
      !weights_->readIndex(base + 4, mw.d4.data(), nn) ||
      !weights_->readIndex(base + 5, mw.w1a.data(), mw.w1a.size()) ||
      !weights_->readIndex(base + 6, mw.w1b.data(), mw.w1b.size()) ||
      !weights_->readIndex(base + 7, mw.w2a.data(), mw.w2a.size()) ||
      !weights_->readIndex(base + 8, mw.w2b.data(), mw.w2b.size()) ||
      !weights_->readIndex(base + 9, mw.w3a.data(), mw.w3a.size()) ||
      !weights_->readIndex(base + 10, mw.w3b.data(), mw.w3b.size()) ||
      !weights_->readIndex(base + 11, mw.cond_v.data(), mw.cond_v.size()) ||
      !weights_->readIndex(base + 12, mw.cond_u.data(), mw.cond_u.size()))
    return false;
  mw.loaded = true;
  return true;
}


void Encoder::mlpTiles(const MlpWeights& mw, const float* x, size_t r0, size_t r1,
                       float* out, uint32_t d, uint32_t nn, uint32_t tile,
                       float* mlpTmp, float* kronTmp, float* kronW,
                       uint32_t layer, uint32_t which) {
  const float* d1 = mw.d1.data();
  const float* d2 = mw.d2.data();
  const float* b2 = mw.b2.data();
  const float* d3 = mw.d3.data();
  const float* d4 = mw.d4.data();
  const float* cond_v = mw.cond_v.data();
  const float* cond_u = mw.cond_u.data();

  // Stage the six factor matrices contiguously (see Encoder.h) into this core's
  // kronW so the two cores never share the factor lines under kronApply.
  const size_t na1 = static_cast<size_t>(mw.w1a0) * mw.w1a1;
  const size_t nb1 = static_cast<size_t>(mw.w1b0) * mw.w1b1;
  const size_t na2 = static_cast<size_t>(mw.w2a0) * mw.w2a1;
  const size_t nb2 = static_cast<size_t>(mw.w2b0) * mw.w2b1;
  const size_t na3 = static_cast<size_t>(mw.w3a0) * mw.w3a1;
  const size_t nb3 = static_cast<size_t>(mw.w3b0) * mw.w3b1;
  std::memcpy(kronW, mw.w1a.data(), na1 * sizeof(float));
  std::memcpy(kronW + na1, mw.w1b.data(), nb1 * sizeof(float));
  std::memcpy(kronW + na1 + nb1, mw.w2a.data(), na2 * sizeof(float));
  std::memcpy(kronW + na1 + nb1 + na2, mw.w2b.data(), nb2 * sizeof(float));
  std::memcpy(kronW + na1 + nb1 + na2 + nb2, mw.w3a.data(), na3 * sizeof(float));
  std::memcpy(kronW + na1 + nb1 + na2 + nb2 + na3, mw.w3b.data(),
              nb3 * sizeof(float));
  const float* const k1a = kronW;
  const float* const k1b = k1a + na1;
  const float* const k2a = k1b + nb1;
  const float* const k2b = k2a + na2;
  const float* const k3a = k2b + nb2;
  const float* const k3b = k3a + na3;

  float* const cond = mlpTmp;
  float* const z = cond + static_cast<size_t>(tile) * nn;
  float* const ztmp = z + static_cast<size_t>(tile) * nn;
  float* const kscratch = kronTmp;
  std::vector<float> xc(static_cast<size_t>(tile) * 8);

  const bool traceLast = progress_ != nullptr && which == 1 &&
                         layer + 1 == layout_.enc_layers;
  const bool trace = progress_ != nullptr && layer == 0 && which == 0;

  for (size_t t0 = r0; t0 < r1; t0 += tile) {
    const size_t n = (r1 - t0 < tile) ? (r1 - t0) : tile;
    const float* xt = x + t0 * d;
    float* ot = out + (t0 - r0) * d;
    const bool dbg = debug_ && layer == 0 && which == 0 && t0 == 0;

    for (size_t t = 0; t < n; ++t) {
      for (uint32_t k = 0; k < 8; ++k) {
        float acc = 0.0f;
        for (uint32_t din = 0; din < d; ++din)
          acc += xt[t * d + din] * cond_v[static_cast<size_t>(din) * 8 + k];
        xc[t * 8 + k] = acc;
      }
      float m = -INFINITY;
      for (uint32_t k = 0; k < 8; ++k)
        if (xc[t * 8 + k] > m) m = xc[t * 8 + k];
      float sum = 0.0f;
      for (uint32_t k = 0; k < 8; ++k) {
        xc[t * 8 + k] = fastmath::expf(xc[t * 8 + k] - m);
        sum += xc[t * 8 + k];
      }
      for (uint32_t k = 0; k < 8; ++k) xc[t * 8 + k] /= sum;
      for (uint32_t c = 0; c < nn; ++c) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < 8; ++k)
          acc += xc[t * 8 + k] * cond_u[static_cast<size_t>(k) * nn + c];
        cond[t * nn + c] = 1.0f + acc;
      }
    }
    if (trace && t0 == 0) progressStage(layer, "mlpc");

    for (size_t t = 0; t < n; ++t) {
      std::memcpy(z + t * nn, xt + t * d, d * sizeof(float));
      std::memset(z + t * nn + d, 0, (nn - d) * sizeof(float));
    }
    // Row-wise diagonal apply: a per-element i % nn is an integer division on a
    // core without one (libgcc call), and the row order/values are identical.
    for (size_t t = 0; t < n; ++t) {
      float* zr = z + t * nn;
      for (uint32_t c = 0; c < nn; ++c) zr[c] *= d1[c];
    }
    if (traceLast && t0 == 0) progressStage(layer, "t1a");
    kronApply(z, n, k1a, mw.w1a0, mw.w1a1, k1b, mw.w1b0, mw.w1b1, ztmp,
              kscratch);
    if (traceLast && t0 == 0) progressStage(layer, "t1b");
    for (size_t t = 0; t < n; ++t)
      for (uint32_t c = 0; c < nn; ++c)
        z[t * nn + c] = ztmp[t * nn + perm1_[c]];
    if (dbg) debug_(debug_udata_, "h1", z, n * nn);
    for (size_t t = 0; t < n; ++t)
      for (uint32_t c = 0; c < nn; ++c)
        z[t * nn + c] = silu(d2[c] * cond[t * nn + c] * z[t * nn + c] + b2[c]);
    if (dbg) debug_(debug_udata_, "h2", z, n * nn);
    if (traceLast && t0 == 0) progressStage(layer, "t1c");
    kronApply(z, n, k2a, mw.w2a0, mw.w2a1, k2b, mw.w2b0, mw.w2b1, ztmp,
              kscratch);
    for (size_t t = 0; t < n; ++t)
      for (uint32_t c = 0; c < nn; ++c)
        z[t * nn + c] = ztmp[t * nn + perm2_[c]];
    if (dbg) debug_(debug_udata_, "h3", z, n * nn);
    for (size_t t = 0; t < n; ++t) {
      float* zr = z + t * nn;
      for (uint32_t c = 0; c < nn; ++c) zr[c] *= d3[c];
    }
    if (traceLast && t0 == 0) progressStage(layer, "t1d");
    kronApply(z, n, k3a, mw.w3a0, mw.w3a1, k3b, mw.w3b0, mw.w3b1, ztmp,
              kscratch);
    if (dbg) debug_(debug_udata_, "h4", ztmp, n * nn);
    if (trace && t0 == 0) progressStage(layer, "mlpk");
    for (size_t t = 0; t < n; ++t)
      for (uint32_t c = 0; c < d; ++c) ot[t * d + c] = d4[c] * ztmp[t * nn + c];
    if (dbg) debug_(debug_udata_, "h5", ot, n * d);
    if (traceLast && t0 == 0) progressStage(layer, "t1e");
    if (trace && t0 == 0) progressStage(layer, "mlpo");
  }
}

void Encoder::mlpJobImpl(void* ctx, uint32_t r0, uint32_t r1, float* out) {
  const MlpJob* j = static_cast<const MlpJob*>(ctx);
  float* const* buf = (r0 == 0) ? j->buf[0] : j->buf[1];
  j->self->mlpTiles(*j->mw, j->x, r0, r1, out, j->d, j->nn, j->tile, buf[0],
                    buf[1], buf[2], j->layer, j->which);
}

void Encoder::hadamardMlp(const float* x, size_t T, uint32_t layer,
                          uint32_t which, float* out) {
  const uint32_t d = layout_.d_model;
  const uint32_t nn = nextPow2(d);
  if (nn != d) return;  // the archive stores d-sized diagonals

  const uint32_t base = layout_.encIndex(layer, 17 + which * 13);

  const cactus::TensorMeta* m1a = weights_->metaAt(base + 5);
  const cactus::TensorMeta* m1b = weights_->metaAt(base + 6);
  const cactus::TensorMeta* m2a = weights_->metaAt(base + 7);
  const cactus::TensorMeta* m2b = weights_->metaAt(base + 8);
  const cactus::TensorMeta* m3a = weights_->metaAt(base + 9);
  const cactus::TensorMeta* m3b = weights_->metaAt(base + 10);
  if (m1a == nullptr || m1b == nullptr || m2a == nullptr ||
      m2b == nullptr || m3a == nullptr || m3b == nullptr)
    return;
  const uint32_t w1a0 = m1a->shape[0], w1a1 = m1a->shape[1];
  const uint32_t w1b0 = m1b->shape[0], w1b1 = m1b->shape[1];
  const uint32_t w2a0 = m2a->shape[0], w2a1 = m2a->shape[1];
  const uint32_t w2b0 = m2b->shape[0], w2b1 = m2b->shape[1];
  const uint32_t w3a0 = m3a->shape[0], w3a1 = m3a->shape[1];
  const uint32_t w3b0 = m3b->shape[0], w3b1 = m3b->shape[1];
  if (w1a0 * w1b0 != nn || w2a0 * w2b0 != nn || w3a0 * w3b0 != nn) return;

  // These tensors are constant, so they are read once and kept: re-reading them
  // (and re-allocating the buffers) cost 115 ms per call, 16 calls per
  // utterance.
  const size_t mi = static_cast<size_t>(layer) * 2 + which;
  if (mlpW_.size() <= mi) mlpW_.resize(static_cast<size_t>(layout_.enc_layers) * 2);
  MlpWeights& mw = mlpW_[mi];
  if (!mw.loaded) {
    mw.d1.resize(nn);
    mw.d2.resize(nn);
    mw.b2.resize(nn);
    mw.d3.resize(nn);
    mw.d4.resize(nn);
    mw.w1a.resize(w1a0 * w1a1);
    mw.w1b.resize(w1b0 * w1b1);
    mw.w2a.resize(w2a0 * w2a1);
    mw.w2b.resize(w2b0 * w2b1);
    mw.w3a.resize(w3a0 * w3a1);
    mw.w3b.resize(w3b0 * w3b1);
    mw.cond_v.resize(static_cast<size_t>(d) * 8);
    mw.cond_u.resize(static_cast<size_t>(8) * nn);
    if (!weights_->readIndex(base + 0, mw.d1.data(), nn) ||
        !weights_->readIndex(base + 1, mw.d2.data(), nn) ||
        !weights_->readIndex(base + 2, mw.b2.data(), nn) ||
        !weights_->readIndex(base + 3, mw.d3.data(), nn) ||
        !weights_->readIndex(base + 4, mw.d4.data(), nn) ||
        !weights_->readIndex(base + 5, mw.w1a.data(), mw.w1a.size()) ||
        !weights_->readIndex(base + 6, mw.w1b.data(), mw.w1b.size()) ||
        !weights_->readIndex(base + 7, mw.w2a.data(), mw.w2a.size()) ||
        !weights_->readIndex(base + 8, mw.w2b.data(), mw.w2b.size()) ||
        !weights_->readIndex(base + 9, mw.w3a.data(), mw.w3a.size()) ||
        !weights_->readIndex(base + 10, mw.w3b.data(), mw.w3b.size()) ||
        !weights_->readIndex(base + 11, mw.cond_v.data(), mw.cond_v.size()) ||
        !weights_->readIndex(base + 12, mw.cond_u.data(), mw.cond_u.size()))
      return;
    mw.loaded = true;
  }
  // Process time in tiles so the MLP working set (xc/cond/z/ztmp) stays in
  // cache: at T=37 the three nn-wide buffers are 76 KB each, and their placement
  // alone moved this stage between 320 and 825 ms per call. Rows are
  // independent, so tiling changes neither the arithmetic nor the order within a
  // row, i.e. the result is bit-identical.
#if WHISTLE_MLP_TILE > 0
  const size_t tile = T < static_cast<size_t>(WHISTLE_MLP_TILE)
                          ? T
                          : static_cast<size_t>(WHISTLE_MLP_TILE);
#else
  const size_t tile = T;
#endif

  // Size the per-call temporaries (and the second-core mirrors used by the
  // dual-core path).
  {
    const size_t need = 3 * tile * nn;
    if (scratch_.mlpTmp.size() < need) scratch_.mlpTmp.assign(need, 0.0f);
    if (scratch_.mlpTmp2.size() < need) scratch_.mlpTmp2.assign(need, 0.0f);
  }
  const size_t kscratch_need =
      std::max(static_cast<size_t>(w1b0) * w1a1,
               std::max(static_cast<size_t>(w2b0) * w2a1,
                        static_cast<size_t>(w3b0) * w3a1));
  if (scratch_.kronTmp.size() < kscratch_need)
    scratch_.kronTmp.assign(kscratch_need, 0.0f);
  if (scratch_.kronTmp2.size() < kscratch_need)
    scratch_.kronTmp2.assign(kscratch_need, 0.0f);

  // The six factor matrices are staged into per-core kronW buffers by mlpTiles;
  // size both copies here.
  const size_t na1 = static_cast<size_t>(w1a0) * w1a1;
  const size_t nb1 = static_cast<size_t>(w1b0) * w1b1;
  const size_t na2 = static_cast<size_t>(w2a0) * w2a1;
  const size_t nb2 = static_cast<size_t>(w2b0) * w2b1;
  const size_t na3 = static_cast<size_t>(w3a0) * w3a1;
  const size_t nb3 = static_cast<size_t>(w3b0) * w3b1;
  const size_t kron_need = na1 + nb1 + na2 + nb2 + na3 + nb3 + 4096;
  if (scratch_.kronW.size() < kron_need) scratch_.kronW.assign(kron_need, 0.0f);
  if (scratch_.kronW2.size() < kron_need)
    scratch_.kronW2.assign(kron_need, 0.0f);

  const bool traceLast = progress_ != nullptr && which == 1 &&
                         layer + 1 == layout_.enc_layers;
  if (traceLast) {
    progressStage(layer, "m1a");
    char st[16];
    std::snprintf(st, sizeof st, "sh%ux%u", w1a0, w1a1);
    progressStage(layer, st);
  }
  const bool trace = progress_ != nullptr && layer == 0 && which == 0;
  if (trace) progressStage(layer, "mlpr");

#if WHISTLE_MLP_CORES
  // Split the frame range across the two cores. Frames are independent, so the
  // two halves are bit-identical to the single-core mlpTiles run below.
  if (T >= 2 * tile) {
    MlpJob job;
    job.self = this;
    job.mw = &mw;
    job.x = x;
    job.d = d;
    job.nn = nn;
    job.tile = static_cast<uint32_t>(tile);
    job.layer = layer;
    job.which = which;
    job.buf[0][0] = scratch_.mlpTmp.data();
    job.buf[0][1] = scratch_.kronTmp.data();
    job.buf[0][2] = scratch_.kronW.data();
    job.buf[1][0] = scratch_.mlpTmp2.data();
    job.buf[1][1] = scratch_.kronTmp2.data();
    job.buf[1][2] = scratch_.kronW2.data();
    if (runRowParallel(&mlpJobImpl, &job, static_cast<uint32_t>(T), d, out, x,
                       static_cast<size_t>(T) * d * sizeof(float))) {
      if (traceLast) progressStage(layer, "m1k");
      return;
    }
  }
#endif

  mlpTiles(mw, x, 0, T, out, d, nn, static_cast<uint32_t>(tile),
           scratch_.mlpTmp.data(), scratch_.kronTmp.data(),
           scratch_.kronW.data(), layer, which);
  if (traceLast) progressStage(layer, "m1k");
}

}  // namespace whistle

