#include "Decoder.h"
#include "FastMath.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace whistle {

namespace {

constexpr float kEps = 1e-6f;

float sigmoid(float x) {
  x = (x < -30.0f) ? -30.0f : ((x > 30.0f) ? 30.0f : x);
  return 1.0f / (1.0f + fastmath::expf(-x));
}

float silu(float x) { return x * sigmoid(x); }

// Nested bisection order of the decoder blocks (architecture.py
// `_ladder_layer_order`). Every prefix keeps both endpoint blocks, so it is the
// only order in which a truncated stack is a trained depth rung: for 8 layers
// the order is 0,7,3,5,1,2,4,6 (depth 4 -> blocks 0,3,5,7).
void ladderOrder(uint32_t layers, std::vector<uint32_t>* order) {
  order->clear();
  if (layers == 0) return;
  if (layers == 1) {
    order->push_back(0);
    return;
  }
  std::vector<uint32_t> selected = {0, layers - 1};
  *order = selected;
  while (order->size() < layers) {
    std::sort(selected.begin(), selected.end());
    uint32_t bestGap = 0, bestLeft = 0, bestRight = 0;
    for (size_t i = 1; i < selected.size(); ++i) {
      const uint32_t left = selected[i - 1], right = selected[i];
      const uint32_t gap = right - left;
      if (gap <= 1) continue;
      if (gap > bestGap || (gap == bestGap && left > bestLeft)) {
        bestGap = gap;
        bestLeft = left;
        bestRight = right;
      }
    }
    const uint32_t candidate = (bestLeft + bestRight) / 2;
    selected.push_back(candidate);
    order->push_back(candidate);
  }
}

constexpr uint32_t kNoRung = 0xFFFFFFFFu;

// Resolved ladder plan: `depth` executed blocks, `rungOf[block]` the in-rung
// index (kNoRung for a skipped block).
struct LadderPlan {
  uint32_t depth = 0;
  std::vector<uint32_t> rungOf;
};

LadderPlan makeLadderPlan(uint32_t requested, uint32_t layers) {
  LadderPlan plan;
  plan.depth = requested;
  if (plan.depth == 0 || plan.depth >= layers) plan.depth = layers;
  else if (plan.depth < 2) plan.depth = 2;
  plan.rungOf.assign(layers, kNoRung);
  if (plan.depth == layers) {
    for (uint32_t l = 0; l < layers; ++l) plan.rungOf[l] = l;
  } else {
    std::vector<uint32_t> order;
    ladderOrder(layers, &order);
    order.resize(plan.depth);
    std::sort(order.begin(), order.end());
    for (uint32_t r = 0; r < plan.depth; ++r) plan.rungOf[order[r]] = r;
  }
  return plan;
}

// In-place per-row fake int8 quantization (last axis), qmax 127, clip
// [-128, 127], round-half-to-even.
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

// Quantize x (rows x cols) to int8 + per-row scale, the same grid as
// fakeQuantRows but returning the integer code and scale instead of the
// re-scaled float (used by the int8 GEMV).
void fakeQuantQ8(const float* x, size_t rows, size_t cols, int8_t* q, float* scale) {
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
    int8_t* qrow = q + r * cols;
    for (size_t c = 0; c < cols; ++c) {
      float v = std::nearbyintf(xr[c] / s);
      if (v < -128.0f) v = -128.0f;
      else if (v > 127.0f) v = 127.0f;
      qrow[c] = static_cast<int8_t>(v);
    }
  }
}

// out[r,c] = (1 + scale[c]) * x[r,c] / rms(x[r,:])  (scale optional, may be null).
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

// C[m,n] = sum_k A[m,k] * B[n,k]  (B stored [out,in] pre-transposed, x @ B^T).
void matmulNT(const float* A, const float* B, float* C, size_t M, size_t N,
              size_t K) {
  for (size_t m = 0; m < M; ++m) {
    const float* ar = A + m * K;
    float* cr = C + m * N;
    for (size_t n = 0; n < N; ++n) {
      const float* br = B + n * K;
      float acc = 0.0f;
      for (size_t k = 0; k < K; ++k) acc += ar[k] * br[k];
      cr[n] = acc;
    }
  }
}

// "Monarch"/Kronecker-style apply used by the Hadamard MLP. z is (T, a0*b0)
// laid out so z.reshape(-1, a0, b0)[i,j]; out is (T, a1*b1) with
// out[k,l] = sum_{i,j} z[i,j] * a[i,k] * b[j,l]. a0*a1 == b0*b1 == nn. Matrix
// dimensions come from the actual factor tensors (16/32 for whistle, 32/32 for
// needle), so this must remain generic.
//
// Implemented as (A (x) B) z == B . Z . A^T where Z = z reshaped to (a0, b0):
//   tmp(j,k) = sum_i Z(i,j) A(i,k)      (b0 x a1)
//   out(k,l) = sum_j tmp(j,k) B(j,l)    (a1 x b1)
// O(a0*b0*(a1+b1)) vs the naive O(a0*a1*b0*b1); identical result, but a
// different summation order (fp32 rounding differs slightly from the naive
// 4-loop). `scratch` must hold b0*a1 floats.
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

}  // namespace

bool Decoder::prepare(const cactus::CactArchive& archive) {
  return weights_.prepare(archive);
}

const float* Decoder::cachedTensor(const char* name, size_t* elements) const {
  const int idx = weights_.indexOf(name);
  if (idx < 0) return nullptr;
  auto it = cache_.find(idx);
  if (it != cache_.end()) {
    if (elements) *elements = it->second.size();
    return it->second.data();
  }
  const cactus::TensorMeta* m = weights_.archive().tensor(static_cast<size_t>(idx));
  if (m == nullptr) return nullptr;
  size_t n = 1;
  for (uint8_t dd = 0; dd < m->ndim; ++dd) n *= m->shape[dd];
  std::vector<float> buf(n);
  if (!weights_.readTensor(name, buf.data(), n)) return nullptr;
  auto res = cache_.emplace(idx, std::move(buf));
  if (elements) *elements = n;
  return res.first->second.data();
}

const float* Decoder::tensor(const char* name, size_t* elements) const {
  return cachedTensor(name, elements);
}

void Decoder::evictRange(int begin, int end) const {
  for (int i = begin; i < end; ++i) cache_.erase(i);
}

bool Decoder::forward(const int32_t* tokens, size_t T, float* logits) const {
  return forwardImpl(tokens, T, nullptr, 0, nullptr, logits, -1);
}

bool Decoder::forwardWithMemory(const int32_t* tokens, size_t T,
                                const float* mem, size_t S,
                                const AudioLayout& layout, float* logits,
                                int32_t head_row) const {
  return forwardImpl(tokens, T, mem, S, &layout, logits, head_row);
}

void Decoder::dropCrossKv() const {
  // Invalidate but keep the buffers: they are ~1 MB and are rebuilt for every
  // utterance, so freeing and re-allocating them only churned the heap.
  for (auto& e : crossKv_) e.valid = false;
}

uint32_t Decoder::prepareCrossAttention(const float* mem, size_t S,
                                        const AudioLayout& layout) const {
  for (auto& e : crossKv_) e.valid = false;
  if (mem == nullptr || S == 0) return 0;

  const ModelGeometry& g = weights_.geometry();
  const uint32_t d = g.d_model;
  const uint32_t H = g.num_heads;
  const uint32_t qk = g.qk_head_dim;
  const uint32_t vd = g.v_head_dim;
  const uint32_t Lfull = g.num_layers;
  const LadderPlan plan = makeLadderPlan(depth_, Lfull);
  crossKv_.resize(Lfull);

  std::vector<int8_t> mem8(S * d);
  std::vector<float> mscale(S);
  fakeQuantQ8(mem, S, d, mem8.data(), mscale.data());

  std::vector<float> k(S * H * qk), v(S * H * vd);
  std::vector<float> kn(qk);
  bool ok = true;
  for (uint32_t layer = 0; layer < Lfull && ok; ++layer) {
    if (plan.rungOf[layer] == kNoRung) continue;
    if (!weights_.gemvCqInt8Index(layout.crossIndex(layer, 2), 0, H * qk,
                                  mem8.data(), mscale.data(),
                                  static_cast<uint32_t>(S), k.data()) ||
        !weights_.gemvCqInt8Index(layout.crossIndex(layer, 3), 0, H * vd,
                                  mem8.data(), mscale.data(),
                                  static_cast<uint32_t>(S), v.data()) ||
        !weights_.readIndex(layout.crossIndex(layer, 7), kn.data(), qk)) {
      ok = false;
      break;
    }
    CrossKv& e = crossKv_[layer];
    e.k.assign(S * H * qk, 0);
    e.v.assign(S * H * vd, 0);
    e.kscale.assign(S * H, 0.0f);
    e.vscale.assign(S * H, 0.0f);
    for (size_t s = 0; s < S; ++s)
      for (uint32_t h = 0; h < H; ++h) {
        float* krow = k.data() + (s * H + h) * qk;
        float sum = 0.0f;
        for (uint32_t c = 0; c < qk; ++c) sum += krow[c] * krow[c];
        const float rms = std::sqrt(sum / static_cast<float>(qk) + kEps);
        for (uint32_t c = 0; c < qk; ++c)
          krow[c] = (1.0f + kn[c]) * krow[c] / rms;
        fakeQuantQ8(krow, 1, qk, e.k.data() + (s * H + h) * qk,
                    e.kscale.data() + s * H + h);
        fakeQuantQ8(v.data() + (s * H + h) * vd, 1, vd,
                    e.v.data() + (s * H + h) * vd,
                    e.vscale.data() + s * H + h);
      }
    e.memS = S;
    e.valid = true;
  }
  if (!ok)
    for (auto& e : crossKv_) e.valid = false;
  return ok ? plan.depth : 0;
}

void Decoder::endSequence() const {
  // Only the logical state is reset: the per-block caches stay allocated and are
  // reused by the next utterance. Re-allocating (and zeroing) ~0.6-1.2 MB per
  // utterance cost a few ms and, worse, churned the heap between utterances,
  // which is what made the first run of a boot behave differently from later
  // ones. Every cached slot is written before it is read, so no clearing is
  // needed here.
  step_.active = false;
  step_.pos = 0;
}

bool Decoder::beginSequence(const AudioLayout& layout) const {
  (void)layout;
  const ModelGeometry& g = weights_.geometry();
  const uint32_t Lfull = g.num_layers;
  const LadderPlan plan = makeLadderPlan(depth_, Lfull);
  const uint32_t cap = 160;
  step_.cap = cap;
  step_.pos = 0;
  if (step_.layers.size() != Lfull) step_.layers.resize(Lfull);
  const uint32_t hist = g.qkv_conv_taps > 1 ? g.qkv_conv_taps - 1 : 0;
  const auto ensure = [](std::vector<float>& v, size_t n) {
    if (v.size() != n) v.assign(n, 0.0f);
  };
  for (uint32_t l = 0; l < Lfull; ++l) {
    if (plan.rungOf[l] == kNoRung) continue;
    LayerStep& ls = step_.layers[l];
    ensure(ls.kcache,
           static_cast<size_t>(cap) * g.num_kv_heads * g.qk_head_dim);
    ensure(ls.vcache,
           static_cast<size_t>(cap) * g.num_kv_heads * g.v_head_dim);
    ensure(ls.qhist, static_cast<size_t>(hist) * g.num_heads * g.qk_head_dim);
    ensure(ls.khist,
           static_cast<size_t>(hist) * g.num_kv_heads * g.qk_head_dim);
    ensure(ls.vhist, static_cast<size_t>(hist) * g.num_kv_heads * g.v_head_dim);
  }
  const uint32_t S = g.num_engram_sites;
  const uint32_t dil = g.engram_conv_dilation ? g.engram_conv_dilation : 1;
  step_.vvRows =
      g.engram_conv_taps > 1 ? (g.engram_conv_taps - 1) * dil + 1 : 1;
  if (step_.vv.size() != S) step_.vv.resize(S);
  for (uint32_t s = 0; s < S; ++s)
    ensure(step_.vv[s], static_cast<size_t>(step_.vvRows) * g.d_model);
  step_.active = true;
  return true;
}

bool Decoder::forwardStep(const int32_t* tokens, size_t n, const float* mem,
                          size_t memS, const AudioLayout& layout,
                          float* logits) const {
  if (!step_.active || tokens == nullptr || n == 0 || logits == nullptr)
    return false;
  if (n > step_.cap) return false;
  step_.pos = n - 1;
  return forwardImpl(tokens, 1, mem, memS, &layout, logits, 0, &step_);
}

bool Decoder::forwardImpl(const int32_t* tokens, size_t T, const float* mem,
                          size_t memS, const AudioLayout* audio,
                          float* logits, int32_t headRow, StepState* step) const {
  const ModelGeometry& g = weights_.geometry();
  const bool inc = step != nullptr;
  const size_t tbase = inc ? step->pos : 0;
  const uint32_t d = g.d_model;
  const uint32_t n = g.mhc_lanes;
  const uint32_t Lfull = g.num_layers;
  // Depth rungs select blocks in ladder order, not 0..depth-1. Learned tensors
  // stay keyed by the original block; the rung index only drives the mHC lane
  // offsets (architecture_mlx `stack(subnetwork_only=True)`).
  const LadderPlan plan = makeLadderPlan(depth_, Lfull);
  const uint32_t L = plan.depth;
  const std::vector<uint32_t>& rungOf = plan.rungOf;
  const uint32_t nC = n * d;
  const uint32_t H = g.num_heads;
  const uint32_t KV = g.num_kv_heads;
  const uint32_t qk = g.qk_head_dim;
  const uint32_t vd = g.v_head_dim;
  const uint32_t taps = g.qkv_conv_taps;
  const uint32_t out_vocab = g.out_vocab ? g.out_vocab : g.vocab;
  const uint32_t rep = H / KV;
  const uint32_t half_qk = qk / 2;
  const float invSqrtD = 1.0f / std::sqrt(static_cast<float>(d));

  // CQ-aware weight matmul: bucket_ on -> dequant-free FWHT-bucket GEMV, else
  // dequant + plain matmul (identical numerics up to float associativity).
  // `out_begin` slices the multi-layer mhc.phi_* tensors.
  auto wmatmul = [&](const char* nm, const float* xin, size_t M, size_t Nn,
                     size_t K, float* out, uint32_t out_begin = 0) {
    if (bucket_) {
      weights_.gemvCq(nm, out_begin, static_cast<uint32_t>(Nn), xin, M, out);
    } else {
      const float* W = cachedTensor(nm, nullptr) + static_cast<size_t>(out_begin) * K;
      matmulNT(xin, W, out, M, Nn, K);
    }
  };

  // int8-activation CQ matmul.
  auto qmatmul = [&](const char* nm, const int8_t* qq, const float* sc, size_t M,
                     size_t Nn, float* out, uint32_t out_begin = 0) {
    weights_.gemvCqInt8(nm, out_begin, static_cast<uint32_t>(Nn), qq, sc, M, out);
  };

  // ---- embedding + initial x (T, d) = emb[tok] * sqrt(d) ----
  scratch_.x.assign(T * d, 0.0f);
  auto& x = scratch_.x;
  const float sqrtD = std::sqrt(static_cast<float>(d));
  const float* emb = nullptr;
  if (bounded_ || bucket_ || int8_) {
    scratch_.erow.assign(d, 0.0f);
    auto& erow = scratch_.erow;
    for (size_t t = 0; t < T; ++t) {
      if (!weights_.readCqRow("embedding",
                              static_cast<uint32_t>(tokens[tbase + t]),
                              erow.data()))
        return false;
      for (uint32_t c = 0; c < d; ++c) x[t * d + c] = erow[c] * sqrtD;
    }
  } else {
    emb = cachedTensor("embedding", nullptr);
    if (emb == nullptr) return false;
    for (size_t t = 0; t < T; ++t) {
      const float* erow = emb + static_cast<size_t>(tokens[tbase + t]) * d;
      for (uint32_t c = 0; c < d; ++c) x[t * d + c] = erow[c] * sqrtD;
    }
  }

  // ---- engram indices / ngram mask / engram k,v ----
  const uint32_t slots = g.engram_slots;
  const uint32_t sub_dim = g.engram_sub_dim;
  const uint32_t num_tables = g.num_engram_tables;
  const uint32_t num_orders = g.num_engram_orders;
  const uint32_t S = g.num_engram_sites;
  const uint32_t heads = num_tables / num_orders;
  const uint32_t dil = g.engram_conv_dilation;
  uint32_t max_order = 0;
  for (uint32_t o = 0; o < num_orders; ++o)
    max_order = std::max(max_order, static_cast<uint32_t>(g.engram_orders[o]));

  scratch_.idx.assign(T * num_tables, 0);
  auto& idx = scratch_.idx;
  for (uint32_t oi = 0; oi < num_orders; ++oi) {
    const uint32_t order = g.engram_orders[oi];
    for (uint32_t h = 0; h < heads; ++h) {
      const uint32_t seed = (0x9E3779B9u * (oi * heads + h + 1)) & 0xFFFFFFFFu;
      const uint32_t table = oi * heads + h;
      for (size_t t = 0; t < T; ++t) {
        uint64_t acc = seed;
        for (uint32_t j = 0; j < order; ++j) {
          const uint64_t uj = (tbase + t >= j)
                                  ? static_cast<uint64_t>(static_cast<uint32_t>(
                                        tokens[tbase + t - j]))
                                  : 0ull;
          acc = ((acc ^ uj) * 0x01000193ull) & 0xFFFFFFFFull;
        }
        acc ^= (acc >> 15);
        idx[t * num_tables + table] = static_cast<int32_t>(acc % slots);
      }
    }
  }

  scratch_.ngramOk.assign(T * num_tables, 0.0f);
  auto& ngramOk = scratch_.ngramOk;
  for (uint32_t k = 0; k < num_orders; ++k) {
    const uint32_t o = g.engram_orders[k];
    for (uint32_t h = 0; h < heads; ++h) {
      const uint32_t table = k * heads + h;
      for (size_t t = 0; t < T; ++t)
        if (tbase + t + 1 >= o) ngramOk[t * num_tables + table] = 1.0f;
    }
  }

  scratch_.ek.assign(S * T * d, 0.0f);
  scratch_.ev.assign(S * T * d, 0.0f);
  auto& ek = scratch_.ek;
  auto& ev = scratch_.ev;
  scratch_.featch.assign(T * d, 0.0f);
  auto& featch = scratch_.featch;
  scratch_.featch8.assign(T * d, 0);
  auto& featch8 = scratch_.featch8;
  scratch_.featch_scale.assign(T, 0.0f);
  auto& featch_scale = scratch_.featch_scale;
  scratch_.kk.assign(T * d, 0.0f);
  scratch_.vv.assign(T * d, 0.0f);
  scratch_.evconv.assign(T * d, 0.0f);
  auto& kk = scratch_.kk;
  auto& vv = scratch_.vv;
  auto& evconv = scratch_.evconv;
  for (uint32_t s = 0; s < S; ++s) {
    char nm[64];
    std::snprintf(nm, sizeof nm, "engram%u.tables", s);
    for (size_t t = 0; t < T; ++t) {
      for (uint32_t ti = 0; ti < num_tables; ++ti) {
        const uint32_t row = ti * slots + static_cast<uint32_t>(idx[t * num_tables + ti]);
        float* dst = featch.data() + t * d + ti * sub_dim;
        if (!weights_.readCqRow(nm, row, dst)) return false;
        if (ngramOk[t * num_tables + ti] == 0.0f)
          for (uint32_t sd = 0; sd < sub_dim; ++sd) dst[sd] = 0.0f;
      }
    }
    if (int8_) {
      fakeQuantQ8(featch.data(), T, d, featch8.data(), featch_scale.data());
      std::snprintf(nm, sizeof nm, "engram%u.key_proj", s);
      qmatmul(nm, featch8.data(), featch_scale.data(), T, d, kk.data());
      std::snprintf(nm, sizeof nm, "engram%u.value_proj", s);
      qmatmul(nm, featch8.data(), featch_scale.data(), T, d, vv.data());
    } else {
      fakeQuantRows(featch.data(), T, d);
      std::snprintf(nm, sizeof nm, "engram%u.key_proj", s);
      wmatmul(nm, featch.data(), T, d, d, kk.data());
      std::snprintf(nm, sizeof nm, "engram%u.value_proj", s);
      wmatmul(nm, featch.data(), T, d, d, vv.data());
    }
    std::snprintf(nm, sizeof nm, "engram%u.taps", s);
    const float* tap = cachedTensor(nm, nullptr);
    if (inc) {
      // Push this position's value row into the ring, then tap the ring with the
      // same (j, off) masks as the batched conv.
      float* ring = step->vv[s].data();
      const uint32_t R = step->vvRows;
      const size_t p = step->pos;
      std::memcpy(ring + (p % R) * d, vv.data(), d * sizeof(float));
      for (uint32_t c = 0; c < d; ++c) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < 4; ++j) {
          const uint32_t off = j * dil;
          const uint32_t mo = j * max_order;
          if (p < off || p < mo) continue;
          acc += tap[j * d + c] * ring[((p - off) % R) * d + c];
        }
        evconv[c] = acc;
      }
    } else {
      for (size_t t = 0; t < T; ++t) {
        for (uint32_t c = 0; c < d; ++c) {
          float acc = 0.0f;
          for (uint32_t j = 0; j < 4; ++j) {
            const uint32_t off = j * dil;
            const uint32_t mo = j * max_order;
            if (t >= off && t >= mo)
              acc += tap[j * d + c] * vv[(t - off) * d + c];
          }
          evconv[t * d + c] = acc;
        }
      }
    }
    std::memcpy(ek.data() + s * T * d, kk.data(), T * d * sizeof(float));
    std::memcpy(ev.data() + s * T * d, evconv.data(), T * d * sizeof(float));
  }

  if (bounded_) {
    // engram 0..S key/value projections are no longer needed (ek/ev are held).
    const int engram_base = 1 + static_cast<int>(Lfull) * 27 + 11;
    evictRange(engram_base, engram_base + static_cast<int>(S) * 4);
  }

  if (captureDebug && !inc) {
    dbgEk = ek;
    dbgEv = ev;
    dbgLayers.assign(L * T * d, 0.0f);
  }

  // ---- RoPE tables (T x half_qk) ----
  const float theta = g.rope_theta;
  scratch_.cos_t.assign(T * half_qk, 0.0f);
  scratch_.sin_t.assign(T * half_qk, 0.0f);
  auto& cos_t = scratch_.cos_t;
  auto& sin_t = scratch_.sin_t;
  for (size_t t = 0; t < T; ++t) {
    for (uint32_t j = 0; j < half_qk; ++j) {
      const float freq = 1.0f / std::pow(theta, (2.0f * j) / static_cast<float>(qk));
      const float a = static_cast<float>(tbase + t) * freq;
      cos_t[t * half_qk + j] = std::cos(a);
      sin_t[t * half_qk + j] = std::sin(a);
    }
  }

  // ---- Hadamard MLP permutation tables (floats on disk -> int32) ----
  const float* p1f = cachedTensor("hada_p1", nullptr);
  const float* p2f = cachedTensor("hada_p2", nullptr);
  scratch_.p1.assign(g.hada_n, 0);
  scratch_.p2.assign(g.hada_n, 0);
  auto& p1 = scratch_.p1;
  auto& p2 = scratch_.p2;
  for (uint32_t c = 0; c < g.hada_n; ++c) {
    p1[c] = static_cast<int32_t>(p1f[c]);
    p2[c] = static_cast<int32_t>(p2f[c]);
  }

  // scratch buffers reused across helpers
  scratch_.xq.assign(T * d, 0.0f);
  auto& xq = scratch_.xq;
  scratch_.qflat.assign(T * H * qk, 0.0f);
  scratch_.qconv.assign(T * H * qk, 0.0f);
  auto& qflat = scratch_.qflat;
  auto& qconv = scratch_.qconv;
  scratch_.kflat.assign(T * KV * qk, 0.0f);
  scratch_.kconv.assign(T * KV * qk, 0.0f);
  auto& kflat = scratch_.kflat;
  auto& kconv = scratch_.kconv;
  scratch_.vflat.assign(T * KV * vd, 0.0f);
  scratch_.vconv.assign(T * KV * vd, 0.0f);
  auto& vflat = scratch_.vflat;
  auto& vconv = scratch_.vconv;
  scratch_.qhead.assign(H * T * qk, 0.0f);
  scratch_.khead.assign(KV * T * qk, 0.0f);
  scratch_.vhead.assign(KV * T * vd, 0.0f);
  auto& qhead = scratch_.qhead;
  auto& khead = scratch_.khead;
  auto& vhead = scratch_.vhead;
  scratch_.scores.assign(H * T * T, 0.0f);
  scratch_.attn.assign(H * T * T, 0.0f);
  auto& scores = scratch_.scores;
  auto& attn = scratch_.attn;
  scratch_.oh.assign(H * T * vd, 0.0f);
  auto& oh = scratch_.oh;
  scratch_.atmp.assign(T * d, 0.0f);
  scratch_.gate.assign(T * d, 0.0f);
  auto& atmp = scratch_.atmp;
  auto& gate = scratch_.gate;
  scratch_.ztmp.assign(T * g.hada_n, 0.0f);
  auto& ztmp = scratch_.ztmp;
  scratch_.xq8.assign(T * d, 0);
  scratch_.atmp8.assign(T * d, 0);
  auto& xq8 = scratch_.xq8;
  auto& atmp8 = scratch_.atmp8;
  scratch_.xq_scale.assign(T, 0.0f);
  scratch_.atmp_scale.assign(T, 0.0f);
  auto& xq_scale = scratch_.xq_scale;
  auto& atmp_scale = scratch_.atmp_scale;
  scratch_.kronScratch.assign(1024, 0.0f);
  auto& kronScratch = scratch_.kronScratch;

  auto attention = [&](const float* xin, uint32_t layer, float* aout) {
    char nm[64];
    if (int8_) {
      fakeQuantQ8(xin, T, d, xq8.data(), xq_scale.data());
      std::snprintf(nm, sizeof nm, "l%u.q_proj", layer);
      qmatmul(nm, xq8.data(), xq_scale.data(), T, H * qk, qflat.data());
      std::snprintf(nm, sizeof nm, "l%u.k_proj", layer);
      qmatmul(nm, xq8.data(), xq_scale.data(), T, KV * qk, kflat.data());
      std::snprintf(nm, sizeof nm, "l%u.v_proj", layer);
      qmatmul(nm, xq8.data(), xq_scale.data(), T, KV * vd, vflat.data());
    } else {
      std::memcpy(xq.data(), xin, T * d * sizeof(float));
      fakeQuantRows(xq.data(), T, d);
      std::snprintf(nm, sizeof nm, "l%u.q_proj", layer);
      wmatmul(nm, xq.data(), T, H * qk, d, qflat.data());
      std::snprintf(nm, sizeof nm, "l%u.k_proj", layer);
      wmatmul(nm, xq.data(), T, KV * qk, d, kflat.data());
      std::snprintf(nm, sizeof nm, "l%u.v_proj", layer);
      wmatmul(nm, xq.data(), T, KV * vd, d, vflat.data());
    }

    std::snprintf(nm, sizeof nm, "l%u.q_taps", layer);
    const float* qt = cachedTensor(nm, nullptr);
    std::snprintf(nm, sizeof nm, "l%u.k_taps", layer);
    const float* kt = cachedTensor(nm, nullptr);
    std::snprintf(nm, sizeof nm, "l%u.v_taps", layer);
    const float* vt = cachedTensor(nm, nullptr);

    if (inc) {
      // One row: tap the stored pre-conv history, then shift it. Same tap order
      // and same operand values as the batched conv.
      auto& ls = step->layers[layer];
      const size_t p = step->pos;
      for (uint32_t f = 0; f < H * qk; ++f) {
        float a = qt[f] * qflat[f];
        for (uint32_t j = 1; j < taps; ++j)
          if (p >= j)
            a += qt[j * (H * qk) + f] * ls.qhist[(j - 1) * (H * qk) + f];
        qconv[f] = a;
      }
      for (uint32_t f = 0; f < KV * qk; ++f) {
        float a = kt[f] * kflat[f];
        for (uint32_t j = 1; j < taps; ++j)
          if (p >= j)
            a += kt[j * (KV * qk) + f] * ls.khist[(j - 1) * (KV * qk) + f];
        kconv[f] = a;
      }
      for (uint32_t f = 0; f < KV * vd; ++f) {
        float a = vt[f] * vflat[f];
        for (uint32_t j = 1; j < taps; ++j)
          if (p >= j)
            a += vt[j * (KV * vd) + f] * ls.vhist[(j - 1) * (KV * vd) + f];
        vconv[f] = a;
      }
      if (taps > 2) {
        for (uint32_t j = taps - 1; j >= 2; --j) {
          std::memcpy(ls.qhist.data() + (j - 1) * (H * qk),
                      ls.qhist.data() + (j - 2) * (H * qk),
                      (H * qk) * sizeof(float));
          std::memcpy(ls.khist.data() + (j - 1) * (KV * qk),
                      ls.khist.data() + (j - 2) * (KV * qk),
                      (KV * qk) * sizeof(float));
          std::memcpy(ls.vhist.data() + (j - 1) * (KV * vd),
                      ls.vhist.data() + (j - 2) * (KV * vd),
                      (KV * vd) * sizeof(float));
        }
      }
      if (taps > 1) {
        std::memcpy(ls.qhist.data(), qflat.data(), (H * qk) * sizeof(float));
        std::memcpy(ls.khist.data(), kflat.data(), (KV * qk) * sizeof(float));
        std::memcpy(ls.vhist.data(), vflat.data(), (KV * vd) * sizeof(float));
      }
    } else {
      for (size_t t = 0; t < T; ++t) {
        for (uint32_t f = 0; f < H * qk; ++f) {
          float a = 0.0f;
          for (uint32_t j = 0; j < taps; ++j)
            if (t >= j) a += qt[j * (H * qk) + f] * qflat[(t - j) * (H * qk) + f];
          qconv[t * (H * qk) + f] = a;
        }
        for (uint32_t f = 0; f < KV * qk; ++f) {
          float a = 0.0f;
          for (uint32_t j = 0; j < taps; ++j)
            if (t >= j) a += kt[j * (KV * qk) + f] * kflat[(t - j) * (KV * qk) + f];
          kconv[t * (KV * qk) + f] = a;
        }
        for (uint32_t f = 0; f < KV * vd; ++f) {
          float a = 0.0f;
          for (uint32_t j = 0; j < taps; ++j)
            if (t >= j) a += vt[j * (KV * vd) + f] * vflat[(t - j) * (KV * vd) + f];
          vconv[t * (KV * vd) + f] = a;
        }
      }
    }

    for (uint32_t h = 0; h < H; ++h)
      for (size_t t = 0; t < T; ++t)
        for (uint32_t c = 0; c < qk; ++c)
          qhead[(h * T + t) * qk + c] = qconv[t * (H * qk) + h * qk + c];
    for (uint32_t h = 0; h < KV; ++h)
      for (size_t t = 0; t < T; ++t)
        for (uint32_t c = 0; c < qk; ++c)
          khead[(h * T + t) * qk + c] = kconv[t * (KV * qk) + h * qk + c];
    for (uint32_t h = 0; h < KV; ++h)
      for (size_t t = 0; t < T; ++t)
        for (uint32_t c = 0; c < vd; ++c)
          vhead[(h * T + t) * vd + c] = vconv[t * (KV * vd) + h * vd + c];

    std::snprintf(nm, sizeof nm, "l%u.q_norm", layer);
    const float* qn = cachedTensor(nm, nullptr);
    std::snprintf(nm, sizeof nm, "l%u.k_norm", layer);
    const float* kn = cachedTensor(nm, nullptr);

    for (uint32_t h = 0; h < H; ++h) {
      for (size_t t = 0; t < T; ++t) {
        float* row = qhead.data() + (h * T + t) * qk;
        float sum = 0.0f;
        for (uint32_t c = 0; c < qk; ++c) sum += row[c] * row[c];
        const float rms = std::sqrt(sum / static_cast<float>(qk) + kEps);
        for (uint32_t c = 0; c < qk; ++c) row[c] = (1.0f + qn[c]) * row[c] / rms;
        float tmp[64];
        for (uint32_t c = 0; c < half_qk; ++c) {
          const float cv = cos_t[t * half_qk + c];
          const float sv = sin_t[t * half_qk + c];
          const float x1 = row[c], x2 = row[c + half_qk];
          tmp[c] = x1 * cv - x2 * sv;
          tmp[c + half_qk] = x2 * cv + x1 * sv;
        }
        std::memcpy(row, tmp, qk * sizeof(float));
        fakeQuantRows(row, 1, qk);
      }
    }
    for (uint32_t h = 0; h < KV; ++h) {
      for (size_t t = 0; t < T; ++t) {
        float* row = khead.data() + (h * T + t) * qk;
        float sum = 0.0f;
        for (uint32_t c = 0; c < qk; ++c) sum += row[c] * row[c];
        const float rms = std::sqrt(sum / static_cast<float>(qk) + kEps);
        for (uint32_t c = 0; c < qk; ++c) row[c] = (1.0f + kn[c]) * row[c] / rms;
        float tmp[64];
        for (uint32_t c = 0; c < half_qk; ++c) {
          const float cv = cos_t[t * half_qk + c];
          const float sv = sin_t[t * half_qk + c];
          const float x1 = row[c], x2 = row[c + half_qk];
          tmp[c] = x1 * cv - x2 * sv;
          tmp[c + half_qk] = x2 * cv + x1 * sv;
        }
        std::memcpy(row, tmp, qk * sizeof(float));
        fakeQuantRows(row, 1, qk);
      }
    }
    for (uint32_t h = 0; h < KV; ++h)
      for (size_t t = 0; t < T; ++t)
        fakeQuantRows(vhead.data() + (h * T + t) * vd, 1, vd);

    if (inc) {
      // The post-norm/rope/quant K/V of this position go into the cache.
      auto& ls = step->layers[layer];
      std::memcpy(ls.kcache.data() + step->pos * (KV * qk), khead.data(),
                  static_cast<size_t>(KV) * qk * sizeof(float));
      std::memcpy(ls.vcache.data() + step->pos * (KV * vd), vhead.data(),
                  static_cast<size_t>(KV) * vd * sizeof(float));
    }

    const float invSqrtQk = 1.0f / std::sqrt(static_cast<float>(qk));
    if (inc) {
      // One query against the cached keys/values of positions 0..pos.
      auto& ls = step->layers[layer];
      const size_t len = step->pos + 1;
      step->sc.resize(len);
      step->pr.resize(len);
      float* sc = step->sc.data();
      float* pr = step->pr.data();
      for (uint32_t h = 0; h < H; ++h) {
        const uint32_t kvh = h / rep;
        const float* qrow = qhead.data() + h * qk;  // T == 1
        float maxs = -INFINITY;
        for (size_t s = 0; s < len; ++s) {
          const float* krow = ls.kcache.data() + (s * KV + kvh) * qk;
          float acc = 0.0f;
          for (uint32_t c = 0; c < qk; ++c) acc += qrow[c] * krow[c];
          sc[s] = acc * invSqrtQk;
          if (sc[s] > maxs) maxs = sc[s];
        }
        float sum = 0.0f;
        for (size_t s = 0; s < len; ++s) {
          pr[s] = fastmath::expf(sc[s] - maxs);
          sum += pr[s];
        }
        for (size_t s = 0; s < len; ++s) pr[s] /= sum;
        // Same rank-1 accumulate as the batched path.
        float* ohrow = oh.data() + h * vd;
        for (uint32_t c = 0; c < vd; ++c) ohrow[c] = 0.0f;
        for (size_t s = 0; s < len; ++s) {
          const float w = pr[s];
          const float* vrow = ls.vcache.data() + (s * KV + kvh) * vd;
          for (uint32_t c = 0; c < vd; ++c) ohrow[c] += w * vrow[c];
        }
      }
    } else {
      for (uint32_t h = 0; h < H; ++h) {
        const uint32_t kvh = h / rep;
        for (size_t t = 0; t < T; ++t) {
          const float* qrow = qhead.data() + (h * T + t) * qk;
          float* srow = scores.data() + (h * T + t) * T;
          float maxs = -INFINITY;
          for (size_t s = 0; s < T; ++s) {
            if (s > t) { srow[s] = -INFINITY; continue; }
            const float* krow = khead.data() + (kvh * T + s) * qk;
            float acc = 0.0f;
            for (uint32_t c = 0; c < qk; ++c) acc += qrow[c] * krow[c];
            const float v = acc * invSqrtQk;
            srow[s] = v;
            if (v > maxs) maxs = v;
          }
          float sum = 0.0f;
          for (size_t s = 0; s < T; ++s) {
            if (s > t) { attn[(h * T + t) * T + s] = 0.0f; continue; }
            const float e = fastmath::expf(srow[s] - maxs);
            attn[(h * T + t) * T + s] = e;
            sum += e;
          }
          for (size_t s = 0; s < T; ++s) attn[(h * T + t) * T + s] /= sum;
          float* ohrow = oh.data() + (h * T + t) * vd;
          const float* arow = attn.data() + (h * T + t) * T;
          for (uint32_t c = 0; c < vd; ++c) ohrow[c] = 0.0f;
          for (size_t s = 0; s < T; ++s) {
            const float w = arow[s];
            const float* vrow = vhead.data() + (kvh * T + s) * vd;
            for (uint32_t c = 0; c < vd; ++c) ohrow[c] += w * vrow[c];
          }
        }
      }
    }

    for (size_t t = 0; t < T; ++t)
      for (uint32_t h = 0; h < H; ++h)
        for (uint32_t c = 0; c < vd; ++c)
          atmp[t * (H * vd) + h * vd + c] = oh[(h * T + t) * vd + c];

    if (int8_) {
      std::snprintf(nm, sizeof nm, "l%u.gate_proj", layer);
      qmatmul(nm, xq8.data(), xq_scale.data(), T, d, gate.data());
    } else {
      std::snprintf(nm, sizeof nm, "l%u.gate_proj", layer);
      wmatmul(nm, xq.data(), T, d, d, gate.data());
    }
    for (size_t i = 0; i < T * d; ++i) atmp[i] *= sigmoid(gate[i]);
    if (int8_) {
      fakeQuantQ8(atmp.data(), T, d, atmp8.data(), atmp_scale.data());
      std::snprintf(nm, sizeof nm, "l%u.out_proj", layer);
      qmatmul(nm, atmp8.data(), atmp_scale.data(), T, d, aout);
    } else {
      fakeQuantRows(atmp.data(), T, d);
      std::snprintf(nm, sizeof nm, "l%u.out_proj", layer);
      wmatmul(nm, atmp.data(), T, d, d, aout);
    }
  };

  auto hadamard_mlp = [&](const float* xin, uint32_t layer, float* hout) {
    const uint32_t nn = g.hada_n;
    char nm[64];
    std::snprintf(nm, sizeof nm, "l%u.d1", layer); const float* d1 = cachedTensor(nm, nullptr);
    std::snprintf(nm, sizeof nm, "l%u.d2", layer); const float* d2 = cachedTensor(nm, nullptr);
    std::snprintf(nm, sizeof nm, "l%u.b2", layer); const float* b2 = cachedTensor(nm, nullptr);
    std::snprintf(nm, sizeof nm, "l%u.d3", layer); const float* d3 = cachedTensor(nm, nullptr);
    std::snprintf(nm, sizeof nm, "l%u.d4", layer); const float* d4 = cachedTensor(nm, nullptr);
    std::snprintf(nm, sizeof nm, "l%u.cond_v", layer); const float* cond_v = cachedTensor(nm, nullptr);
    std::snprintf(nm, sizeof nm, "l%u.cond_u", layer); const float* cond_u = cachedTensor(nm, nullptr);
    std::snprintf(nm, sizeof nm, "l%u.w1a", layer); const float* w1a = cachedTensor(nm, nullptr);
    std::snprintf(nm, sizeof nm, "l%u.w1b", layer); const float* w1b = cachedTensor(nm, nullptr);
    std::snprintf(nm, sizeof nm, "l%u.w2a", layer); const float* w2a = cachedTensor(nm, nullptr);
    std::snprintf(nm, sizeof nm, "l%u.w2b", layer); const float* w2b = cachedTensor(nm, nullptr);
    std::snprintf(nm, sizeof nm, "l%u.w3a", layer); const float* w3a = cachedTensor(nm, nullptr);
    std::snprintf(nm, sizeof nm, "l%u.w3b", layer); const float* w3b = cachedTensor(nm, nullptr);

    // Factor matrix dims (16x16 / 32x32 for whistle; 32x32 / 32x32 for needle).
    std::snprintf(nm, sizeof nm, "l%u.w1a", layer); const cactus::TensorMeta* m1a = weights_.meta(nm);
    std::snprintf(nm, sizeof nm, "l%u.w1b", layer); const cactus::TensorMeta* m1b = weights_.meta(nm);
    std::snprintf(nm, sizeof nm, "l%u.w2a", layer); const cactus::TensorMeta* m2a = weights_.meta(nm);
    std::snprintf(nm, sizeof nm, "l%u.w2b", layer); const cactus::TensorMeta* m2b = weights_.meta(nm);
    std::snprintf(nm, sizeof nm, "l%u.w3a", layer); const cactus::TensorMeta* m3a = weights_.meta(nm);
    std::snprintf(nm, sizeof nm, "l%u.w3b", layer); const cactus::TensorMeta* m3b = weights_.meta(nm);
    if (!m1a || !m1b || !m2a || !m2b || !m3a || !m3b) return;
    const uint32_t w1a0 = m1a->shape[0], w1a1 = m1a->shape[1];
    const uint32_t w1b0 = m1b->shape[0], w1b1 = m1b->shape[1];
    const uint32_t w2a0 = m2a->shape[0], w2a1 = m2a->shape[1];
    const uint32_t w2b0 = m2b->shape[0], w2b1 = m2b->shape[1];
    const uint32_t w3a0 = m3a->shape[0], w3a1 = m3a->shape[1];
    const uint32_t w3b0 = m3b->shape[0], w3b1 = m3b->shape[1];
    const size_t kscratch_need = std::max(w1b0 * w1a1,
        std::max(w2b0 * w2a1, w3b0 * w3a1));
    if (kronScratch.size() < kscratch_need) kronScratch.resize(kscratch_need);
    float* kscratch = kronScratch.data();

    scratch_.xc.assign(T * 8, 0.0f);
    auto& xc = scratch_.xc;
    for (size_t t = 0; t < T; ++t)
      for (uint32_t k = 0; k < 8; ++k) {
        float acc = 0.0f;
        for (uint32_t din = 0; din < d; ++din) acc += xin[t * d + din] * cond_v[din * 8 + k];
        xc[t * 8 + k] = acc;
      }
    for (size_t t = 0; t < T; ++t) {
      float m = -INFINITY;
      for (uint32_t i = 0; i < 8; ++i) if (xc[t * 8 + i] > m) m = xc[t * 8 + i];
      float ssum = 0.0f;
      for (uint32_t i = 0; i < 8; ++i) { xc[t * 8 + i] = fastmath::expf(xc[t * 8 + i] - m); ssum += xc[t * 8 + i]; }
      for (uint32_t i = 0; i < 8; ++i) xc[t * 8 + i] /= ssum;
    }
    scratch_.cond.assign(T * nn, 0.0f);
    auto& cond = scratch_.cond;
    for (size_t t = 0; t < T; ++t)
      for (uint32_t k2 = 0; k2 < nn; ++k2) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < 8; ++i) acc += xc[t * 8 + i] * cond_u[i * nn + k2];
        cond[t * nn + k2] = 1.0f + acc;
      }

    scratch_.z.assign(T * nn, 0.0f);
    auto& z = scratch_.z;
    for (size_t t = 0; t < T; ++t) {
      std::memcpy(z.data() + t * nn, xin + t * d, d * sizeof(float));
      std::memset(z.data() + t * nn + d, 0, (nn - d) * sizeof(float));
    }
    for (size_t i = 0; i < T * nn; ++i) z[i] *= d1[i % nn];
    kronApply(z.data(), T, w1a, w1a0, w1a1, w1b, w1b0, w1b1, ztmp.data(), kscratch);
    for (size_t t = 0; t < T; ++t)
      for (uint32_t c = 0; c < nn; ++c) z[t * nn + c] = ztmp[t * nn + p1[c]];
    for (size_t t = 0; t < T; ++t)
      for (uint32_t c = 0; c < nn; ++c)
        z[t * nn + c] = silu(d2[c] * cond[t * nn + c] * z[t * nn + c] + b2[c]);
    kronApply(z.data(), T, w2a, w2a0, w2a1, w2b, w2b0, w2b1, ztmp.data(), kscratch);
    for (size_t t = 0; t < T; ++t)
      for (uint32_t c = 0; c < nn; ++c) z[t * nn + c] = ztmp[t * nn + p2[c]];
    for (size_t i = 0; i < T * nn; ++i) z[i] *= d3[i % nn];
    kronApply(z.data(), T, w3a, w3a0, w3a1, w3b, w3b0, w3b1, ztmp.data(), kscratch);
    for (size_t t = 0; t < T; ++t)
      for (uint32_t c = 0; c < d; ++c) hout[t * d + c] = d4[c] * ztmp[t * nn + c];
  };

  auto cross_attn = [&](const float* xin, uint32_t layer, float* out) {
    if (audio == nullptr || mem == nullptr || memS == 0) return;
    const uint32_t H = g.num_heads;
    const uint32_t qk = g.qk_head_dim;
    const uint32_t vd = g.v_head_dim;
    const bool cached = layer < crossKv_.size() && crossKv_[layer].valid &&
                        crossKv_[layer].memS == memS;
    auto& xq = scratch_.c_xq;
    xq.resize(T * d);
    auto& q = scratch_.c_q;
    q.resize(T * H * qk);
    auto& k = scratch_.c_k;
    k.resize(memS * H * qk);
    auto& v = scratch_.c_v;
    v.resize(memS * H * vd);
    auto& qn = scratch_.c_qn;
    qn.resize(qk);
    auto& kn = scratch_.c_kn;
    kn.resize(qk);
    auto& gate = scratch_.c_gate;
    gate.resize(T * d);
    auto& oh = scratch_.c_oh;
    oh.resize(T * H * vd);
    auto& sc = scratch_.c_sc;
    sc.resize(memS);
    auto& pr = scratch_.c_pr;
    pr.resize(memS);
    auto& q8 = scratch_.c_q8;
    q8.resize(T * d);
    auto& mem8 = scratch_.c_mem8;
    mem8.resize(memS * d);
    auto& oh8 = scratch_.c_oh8;
    oh8.resize(T * d);
    auto& qscale = scratch_.c_qscale;
    qscale.resize(T);
    auto& mscale = scratch_.c_mscale;
    mscale.resize(memS);
    auto& ohscale = scratch_.c_ohscale;
    ohscale.resize(T);
    std::memcpy(xq.data(), xin, T * d * sizeof(float));
    fakeQuantRows(xq.data(), T, d);
    fakeQuantQ8(xq.data(), T, d, q8.data(), qscale.data());
    if (!weights_.gemvCqInt8Index(audio->crossIndex(layer, 1), 0, H * qk,
                                  q8.data(), qscale.data(), T, q.data()) ||
        !weights_.readIndex(audio->crossIndex(layer, 6), qn.data(), qk)) {
      std::memset(out, 0, T * d * sizeof(float));
      return;
    }

    if (cached) {
      // K/V were projected and normed once per utterance; only dequantize.
      const CrossKv& e = crossKv_[layer];
      for (size_t s = 0; s < memS; ++s)
        for (uint32_t h = 0; h < H; ++h) {
          const float ks = e.kscale[s * H + h];
          const int8_t* kc = e.k.data() + (s * H + h) * qk;
          float* krow = k.data() + (s * H + h) * qk;
          for (uint32_t c = 0; c < qk; ++c) krow[c] = kc[c] * ks;
          const float vs = e.vscale[s * H + h];
          const int8_t* vc = e.v.data() + (s * H + h) * vd;
          float* vrow = v.data() + (s * H + h) * vd;
          for (uint32_t c = 0; c < vd; ++c) vrow[c] = vc[c] * vs;
        }
    } else {
      fakeQuantQ8(mem, memS, d, mem8.data(), mscale.data());
      if (!weights_.gemvCqInt8Index(audio->crossIndex(layer, 2), 0, H * qk,
                                    mem8.data(), mscale.data(),
                                    static_cast<uint32_t>(memS), k.data()) ||
          !weights_.gemvCqInt8Index(audio->crossIndex(layer, 3), 0, H * vd,
                                    mem8.data(), mscale.data(),
                                    static_cast<uint32_t>(memS), v.data()) ||
          !weights_.readIndex(audio->crossIndex(layer, 7), kn.data(), qk)) {
        std::memset(out, 0, T * d * sizeof(float));
        return;
      }
      for (size_t s = 0; s < memS; ++s)
        for (uint32_t h = 0; h < H; ++h) {
          float* row = k.data() + (s * H + h) * qk;
          float sum = 0.0f;
          for (uint32_t c = 0; c < qk; ++c) sum += row[c] * row[c];
          const float rms = std::sqrt(sum / static_cast<float>(qk) + kEps);
          for (uint32_t c = 0; c < qk; ++c) row[c] = (1.0f + kn[c]) * row[c] / rms;
          fakeQuantRows(row, 1, qk);
          fakeQuantRows(v.data() + (s * H + h) * vd, 1, vd);
        }
    }

    for (size_t t = 0; t < T; ++t)
      for (uint32_t h = 0; h < H; ++h) {
        float* row = q.data() + (t * H + h) * qk;
        float sum = 0.0f;
        for (uint32_t c = 0; c < qk; ++c) sum += row[c] * row[c];
        const float rms = std::sqrt(sum / static_cast<float>(qk) + kEps);
        for (uint32_t c = 0; c < qk; ++c) row[c] = (1.0f + qn[c]) * row[c] / rms;
        fakeQuantRows(row, 1, qk);
      }

    const float invSqrtQk = 1.0f / std::sqrt(static_cast<float>(qk));
    for (uint32_t h = 0; h < H; ++h)
      for (size_t t = 0; t < T; ++t) {
        const float* qrow = q.data() + (t * H + h) * qk;
        float maxs = -INFINITY;
        for (size_t s = 0; s < memS; ++s) {
          const float* krow = k.data() + (s * H + h) * qk;
          float acc = 0.0f;
          for (uint32_t c = 0; c < qk; ++c) acc += qrow[c] * krow[c];
          sc[s] = acc * invSqrtQk;
          if (sc[s] > maxs) maxs = sc[s];
        }
        float sum = 0.0f;
        for (size_t s = 0; s < memS; ++s) {
          pr[s] = fastmath::expf(sc[s] - maxs);
          sum += pr[s];
        }
        for (size_t s = 0; s < memS; ++s) pr[s] /= sum;
        // Rank-1 accumulate (s outer) so the memory K/V rows stream instead of
        // being re-strided per output tap; s order per output is unchanged.
        float* ohrow = oh.data() + t * (H * vd) + h * vd;
        for (uint32_t c = 0; c < vd; ++c) ohrow[c] = 0.0f;
        for (size_t s = 0; s < memS; ++s) {
          const float w = pr[s];
          const float* vrow = v.data() + (s * H + h) * vd;
          for (uint32_t c = 0; c < vd; ++c) ohrow[c] += w * vrow[c];
        }
      }

    if (!weights_.gemvCqInt8Index(audio->crossIndex(layer, 4), 0, d,
                                  q8.data(), qscale.data(), T, gate.data())) {
      std::memset(out, 0, T * d * sizeof(float));
      return;
    }
    for (size_t i = 0; i < T * d; ++i) oh[i] *= sigmoid(gate[i]);
    fakeQuantRows(oh.data(), T, H * vd);
    fakeQuantQ8(oh.data(), T, H * vd, oh8.data(), ohscale.data());
    if (!weights_.gemvCqInt8Index(audio->crossIndex(layer, 5), 0, d,
                                  oh8.data(), ohscale.data(), T, out)) {
      std::memset(out, 0, T * d * sizeof(float));
    }
  };

  auto block = [&](const float* u, uint32_t layer, const float* sflags, float* bout) {
    scratch_.bx.assign(T * d, 0.0f);
    auto& bx = scratch_.bx;
    std::memcpy(bx.data(), u, T * d * sizeof(float));
    for (uint32_t s = 0; s < S; ++s) {
      if (sflags[s] == 0.0f) continue;
      for (size_t t = 0; t < T; ++t) {
        const float* xrow = u + t * d;
        const float* erow = ek.data() + s * T * d + t * d;
        float sxx = 0.0f, see = 0.0f;
        for (uint32_t c = 0; c < d; ++c) { sxx += xrow[c] * xrow[c]; see += erow[c] * erow[c]; }
        const float rx = std::sqrt(sxx / static_cast<float>(d) + kEps);
        const float re = std::sqrt(see / static_cast<float>(d) + kEps);
        float dot = 0.0f;
        for (uint32_t c = 0; c < d; ++c) dot += (xrow[c] / rx) * (erow[c] / re);
        const float alpha = sigmoid(dot * invSqrtD);
        for (uint32_t c = 0; c < d; ++c) bx[t * d + c] += alpha * ev[s * T * d + t * d + c];
      }
    }
    if (progress) progress(progressUdata, "eng");

    scratch_.resid.assign(T * d, 0.0f);
    auto& resid = scratch_.resid;
    std::memcpy(resid.data(), bx.data(), T * d * sizeof(float));
    char nm[64];
    std::snprintf(nm, sizeof nm, "l%u.norm_in", layer);
    const float* nin = cachedTensor(nm, nullptr);
    scratch_.hn.assign(T * d, 0.0f);
    auto& hn = scratch_.hn;
    rmsNormRows(bx.data(), T, d, nin, hn.data());

    scratch_.at.assign(T * d, 0.0f);
    auto& at = scratch_.at;
    attention(hn.data(), layer, at.data());
    if (progress) progress(progressUdata, "attn");

    std::snprintf(nm, sizeof nm, "l%u.post_norm", layer);
    const float* pn = cachedTensor(nm, nullptr);
    rmsNormRows(at.data(), T, d, pn, at.data());
    std::snprintf(nm, sizeof nm, "l%u.attn_gate", layer);
    const float* ag = cachedTensor(nm, nullptr);
    const float gsc = sigmoid(ag[0]);
    for (size_t i = 0; i < T * d; ++i) at[i] = resid[i] + gsc * at[i];

    if (audio != nullptr && mem != nullptr && memS > 0) {
      std::vector<float> cscale(d), cn(T * d), cout(T * d);
      if (!weights_.readIndex(audio->crossIndex(layer, 0), cscale.data(), d))
        return;
      rmsNormRows(at.data(), T, d, cscale.data(), cn.data());
      cross_attn(cn.data(), layer, cout.data());
      if (!weights_.readIndex(audio->crossIndex(layer, 8), cscale.data(), d))
        return;
      rmsNormRows(cout.data(), T, d, cscale.data(), cn.data());
      float cg = 0.0f;
      if (!weights_.readIndex(audio->crossIndex(layer, 9), &cg, 1)) return;
      cg = sigmoid(cg);
      for (size_t i = 0; i < T * d; ++i) at[i] += cg * cn[i];
    }

    std::memcpy(resid.data(), at.data(), T * d * sizeof(float));
    std::snprintf(nm, sizeof nm, "l%u.pre_hada", layer);
    const float* ph = cachedTensor(nm, nullptr);
    scratch_.hn2.assign(T * d, 0.0f);
    auto& hn2 = scratch_.hn2;
    rmsNormRows(at.data(), T, d, ph, hn2.data());
    scratch_.hmo.assign(T * d, 0.0f);
    auto& hmo = scratch_.hmo;
    hadamard_mlp(hn2.data(), layer, hmo.data());
    if (progress) progress(progressUdata, "hada");
    for (size_t i = 0; i < T * d; ++i) bout[i] = resid[i] + hmo[i];
  };

  // ---- MHC hyper-connections around the 20 blocks ----
  const float* a_pre = cachedTensor("mhc.a_pre", nullptr);
  const float* a_post = cachedTensor("mhc.a_post", nullptr);
  const float* a_res = cachedTensor("mhc.a_res", nullptr);
  const float* b_pre = cachedTensor("mhc.b_pre", nullptr);
  const float* b_post = cachedTensor("mhc.b_post", nullptr);
  const float* b_res = cachedTensor("mhc.b_res", nullptr);

  scratch_.pre_off.assign(Lfull * n, 0.0f);
  scratch_.post_off.assign(Lfull * n, 0.0f);
  auto& pre_off = scratch_.pre_off;
  auto& post_off = scratch_.post_off;
  for (uint32_t l = 0; l < Lfull; ++l) {
    const uint32_t rung = rungOf[l];
    if (rung == kNoRung) continue;
    const uint32_t lane = rung % n;
    for (uint32_t i = 0; i < n; ++i) {
      const float laneV = (i == lane) ? 1.0f : 0.0f;
      pre_off[l * n + i] = 8.0f * laneV - 4.0f;
      post_off[l * n + i] = -4.0f * (1.0f - laneV);
    }
  }
  scratch_.site_flags.assign(Lfull * S, 0.0f);
  auto& site_flags = scratch_.site_flags;
  for (uint32_t s = 0; s < S; ++s)
    site_flags[static_cast<size_t>(g.engram_sites[s]) * S + s] = 1.0f;

  // stream (T, n, d): each lane starts as the embedding x.
  scratch_.stream.assign(T * n * d, 0.0f);
  auto& stream = scratch_.stream;
  for (size_t t = 0; t < T; ++t)
    for (uint32_t i = 0; i < n; ++i)
      std::memcpy(stream.data() + (t * n + i) * d, x.data() + t * d, d * sizeof(float));

  scratch_.xf.assign(T * n * d, 0.0f);
  scratch_.nx.assign(T * nC, 0.0f);
  scratch_.nx8.assign(T * nC, 0);
  scratch_.nx_scale.assign(T, 0.0f);
  scratch_.hpre.assign(T * n, 0.0f);
  scratch_.hpost.assign(T * n, 0.0f);
  scratch_.hres.assign(T * n * n, 0.0f);
  scratch_.u.assign(T * d, 0.0f);
  scratch_.y.assign(T * d, 0.0f);
  scratch_.blk.assign(T * d, 0.0f);
  scratch_.res.assign(T * n * n, 0.0f);
  scratch_.kkmat.assign(n * n, 0.0f);
  scratch_.newstream.assign(T * n * d, 0.0f);
  auto& xf = scratch_.xf;
  auto& nx = scratch_.nx;
  auto& nx8 = scratch_.nx8;
  auto& nx_scale = scratch_.nx_scale;
  auto& hpre = scratch_.hpre;
  auto& hpost = scratch_.hpost;
  auto& hres = scratch_.hres;
  auto& u = scratch_.u;
  auto& y = scratch_.y;
  auto& blk = scratch_.blk;
  auto& res = scratch_.res;
  auto& kkmat = scratch_.kkmat;
  auto& newstream = scratch_.newstream;

  auto sinkhorn = [](float* k, uint32_t sz) {
    for (int it = 0; it < 20; ++it) {
      for (uint32_t i = 0; i < sz; ++i) {
        float m = -INFINITY;
        for (uint32_t j = 0; j < sz; ++j) if (k[i * sz + j] > m) m = k[i * sz + j];
        float ssum = 0.0f;
        for (uint32_t j = 0; j < sz; ++j) ssum += fastmath::expf(k[i * sz + j] - m);
        const float lse = m + std::logf(ssum);
        for (uint32_t j = 0; j < sz; ++j) k[i * sz + j] -= lse;
      }
      for (uint32_t j = 0; j < sz; ++j) {
        float m = -INFINITY;
        for (uint32_t i = 0; i < sz; ++i) if (k[i * sz + j] > m) m = k[i * sz + j];
        float ssum = 0.0f;
        for (uint32_t i = 0; i < sz; ++i) ssum += fastmath::expf(k[i * sz + j] - m);
        const float lse = m + std::logf(ssum);
        for (uint32_t i = 0; i < sz; ++i) k[i * sz + j] -= lse;
      }
    }
    for (uint32_t i = 0; i < sz * sz; ++i) k[i] = fastmath::expf(k[i]);
  };

  if (progress) progress(progressUdata, "pre");
  for (uint32_t layer = 0; layer < Lfull; ++layer) {
    const uint32_t rung = rungOf[layer];
    if (rung == kNoRung) continue;
    std::memcpy(xf.data(), stream.data(), T * n * d * sizeof(float));

    // nx = fake_quant(rms_unit(stream)) over nC per token
    rmsNormRows(stream.data(), T, nC, nullptr, nx.data());
    if (int8_) {
      fakeQuantQ8(nx.data(), T, nC, nx8.data(), nx_scale.data());
    } else {
      fakeQuantRows(nx.data(), T, nC);
    }

    if (int8_) {
      qmatmul("mhc.phi_pre", nx8.data(), nx_scale.data(), T, n, hpre.data(), layer * n);
    } else {
      wmatmul("mhc.phi_pre", nx.data(), T, n, nC, hpre.data(), layer * n);
    }
    for (size_t tt = 0; tt < T; ++tt)
      for (uint32_t i = 0; i < n; ++i)
        hpre[tt * n + i] = sigmoid(a_pre[layer] * hpre[tt * n + i] +
                                   b_pre[layer * n + i] + pre_off[layer * n + i]);

    for (size_t tt = 0; tt < T; ++tt)
      for (uint32_t c = 0; c < d; ++c) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < n; ++i) acc += hpre[tt * n + i] * xf[(tt * n + i) * d + c];
        u[tt * d + c] = acc;
      }

    block(u.data(), layer, site_flags.data() + static_cast<size_t>(layer) * S, blk.data());
    if (bounded_) {
      // This layer's 27 tensors (q/k/v proj, taps, norms, gate/out, hadamard
      // factors, etc.) are consumed; drop them before the next layer.
      evictRange(1 + static_cast<int>(layer) * 27,
                 1 + static_cast<int>(layer) * 27 + 27);
    }
    for (size_t i = 0; i < T * d; ++i) y[i] = blk[i] - u[i];

    if (int8_) {
      qmatmul("mhc.phi_post", nx8.data(), nx_scale.data(), T, n, hpost.data(), layer * n);
    } else {
      wmatmul("mhc.phi_post", nx.data(), T, n, nC, hpost.data(), layer * n);
    }
    for (size_t tt = 0; tt < T; ++tt)
      for (uint32_t i = 0; i < n; ++i)
        hpost[tt * n + i] = 2.0f * sigmoid(a_post[layer] * hpost[tt * n + i] +
                                           b_post[layer * n + i] + post_off[layer * n + i]);

    if (int8_) {
      qmatmul("mhc.phi_res", nx8.data(), nx_scale.data(), T, n * n, res.data(), layer * n * n);
    } else {
      wmatmul("mhc.phi_res", nx.data(), T, n * n, nC, res.data(), layer * n * n);
    }
    for (size_t tt = 0; tt < T; ++tt) {
      for (uint32_t i = 0; i < n * n; ++i)
        kkmat[i] = a_res[layer] * res[tt * n * n + i] + b_res[layer * n * n + i];
      sinkhorn(kkmat.data(), n);
      std::memcpy(hres.data() + tt * n * n, kkmat.data(), n * n * sizeof(float));
    }

    for (size_t tt = 0; tt < T; ++tt)
      for (uint32_t i = 0; i < n; ++i)
        for (uint32_t c = 0; c < d; ++c) {
          float acc = 0.0f;
          for (uint32_t j = 0; j < n; ++j)
            acc += hres[tt * n * n + i * n + j] * xf[(tt * n + j) * d + c];
          acc += hpost[tt * n + i] * y[tt * d + c];
          newstream[(tt * n + i) * d + c] = acc;
        }
    std::memcpy(stream.data(), newstream.data(), T * n * d * sizeof(float));

    if (captureDebug) {
      float* dst = dbgLayers.data() + static_cast<size_t>(rung) * T * d;
      for (size_t tt = 0; tt < T; ++tt)
        for (uint32_t c = 0; c < d; ++c) {
          float acc = 0.0f;
          for (uint32_t i = 0; i < n; ++i) acc += stream[(tt * n + i) * d + c];
          dst[tt * d + c] = acc / static_cast<float>(n);
        }
    }
    if (progress) {
      char tag[8];
      snprintf(tag, sizeof(tag), "l%u", layer);
      progress(progressUdata, tag);
    }
  }

  if (headRow >= 0 && static_cast<size_t>(headRow) >= T) headRow = -1;
  const size_t headRows = (headRow >= 0) ? 1 : T;
  const size_t headFirst = (headRow >= 0) ? static_cast<size_t>(headRow) : 0;
  float* headOut = logits + headFirst * out_vocab;

  scratch_.fx.assign(headRows * d, 0.0f);
  scratch_.fx8.assign(headRows * d, 0);
  scratch_.fx_scale.assign(headRows, 0.0f);
  auto& fx = scratch_.fx;
  auto& fx8 = scratch_.fx8;
  auto& fx_scale = scratch_.fx_scale;
  for (size_t tt = 0; tt < headRows; ++tt) {
    const float* srow = stream.data() + (headFirst + tt) * n * d;
    for (uint32_t c = 0; c < d; ++c) {
      float acc = 0.0f;
      for (uint32_t i = 0; i < n; ++i) acc += srow[i * d + c];
      fx[tt * d + c] = acc / static_cast<float>(n);
    }
  }
  const float* fn = cachedTensor("final_norm", nullptr);
  rmsNormRows(fx.data(), headRows, d, fn, fx.data());
  if (int8_) {
    fakeQuantQ8(fx.data(), headRows, d, fx8.data(), fx_scale.data());
    weights_.gemvCqInt8("embedding", 0, out_vocab, fx8.data(), fx_scale.data(),
                        headRows, headOut);
  } else {
    fakeQuantRows(fx.data(), headRows, d);
    if (bucket_) {
      weights_.gemvCq("embedding", 0, out_vocab, fx.data(), headRows, headOut);
    } else if (bounded_) {
      // Stream the output head: dequantize one embedding row at a time so the
      // 16.8 MB fp32 embedding is never materialized.
      scratch_.erow.assign(d, 0.0f);
      auto& erow = scratch_.erow;
      for (uint32_t v = 0; v < out_vocab; ++v) {
        if (!weights_.readCqRow("embedding", v, erow.data())) return false;
        for (size_t t = 0; t < headRows; ++t) {
          const float* frow = fx.data() + t * d;
          float acc = 0.0f;
          for (uint32_t c = 0; c < d; ++c) acc += frow[c] * erow[c];
          headOut[t * out_vocab + v] = acc;
        }
      }
    } else {
      matmulNT(fx.data(), emb, headOut, headRows, out_vocab, d);
    }
  }
  if (progress) progress(progressUdata, "head");
  return true;
}

}  // namespace whistle