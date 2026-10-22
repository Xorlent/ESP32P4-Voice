#include "MelSpectrogram.h"

#include <cmath>
#include <cstring>

namespace whistle {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

bool MelSpectrogram::init(const MelConfig& cfg, const float* filterbank) {
  if (cfg.n_fft == 0 || cfg.n_mels == 0 || cfg.win_length == 0 ||
      cfg.hop == 0 || cfg.win_length > cfg.n_fft || filterbank == nullptr) {
    return false;
  }
  // The FFT below is radix-2.
  if ((cfg.n_fft & (cfg.n_fft - 1)) != 0) return false;

  cfg_ = cfg;
  pow_bins_ = cfg.n_fft / 2 + 1;

  // np.hanning(N) is symmetric: 0.5 - 0.5*cos(2*pi*n/(N-1)).
  window_.assign(cfg.win_length, 0.0f);
  const double denom = (cfg.win_length > 1) ? (cfg.win_length - 1) : 1.0;
  double wsum = 0.0;
  for (uint32_t n = 0; n < cfg.win_length; ++n) {
    const double v = 0.5 - 0.5 * std::cos(2.0 * kPi * n / denom);
    window_[n] = static_cast<float>(v);
    wsum += v;
  }
  window_sum_ = static_cast<float>(wsum);

  // ---- filterbank: transpose, trim to each bin's nonzero span, pack ----
  // The model stores it row-major [pow_bins][n_mels], so reading one mel bin
  // walks with stride n_mels floats and every tap is a fresh cache line (the
  // whole 160 KB bank got re-streamed from PSRAM once per frame). Packing the
  // nonzero span of each bin leaves a few KB that are read sequentially.
  // Skipping exact zeros does not change the ascending float sum.
  fb_off_.assign(cfg.n_mels, 0u);
  fb_len_.assign(cfg.n_mels, 0u);
  fb_lo_.assign(cfg.n_mels, 0u);
  size_t packed = 0;
  for (uint32_t m = 0; m < cfg.n_mels; ++m) {
    uint32_t lo = 0;
    uint32_t hi = static_cast<uint32_t>(pow_bins_);
    while (lo < hi &&
           filterbank[static_cast<size_t>(lo) * cfg.n_mels + m] == 0.0f)
      ++lo;
    while (hi > lo &&
           filterbank[static_cast<size_t>(hi - 1) * cfg.n_mels + m] == 0.0f)
      --hi;
    fb_lo_[m] = lo;
    fb_len_[m] = hi - lo;
    packed += hi - lo;
  }
  fb_pack_.assign(packed, 0.0f);
  size_t w = 0;
  for (uint32_t m = 0; m < cfg.n_mels; ++m) {
    fb_off_[m] = static_cast<uint32_t>(w);
    for (uint32_t f = 0; f < fb_len_[m]; ++f, ++w)
      fb_pack_[w] =
          filterbank[static_cast<size_t>(fb_lo_[m] + f) * cfg.n_mels + m];
  }

  re_.assign(cfg.n_fft, 0.0f);
  im_.assign(cfg.n_fft, 0.0f);

  // Bit reversal for the iterative FFT.
  uint32_t bits = 0;
  while ((1u << bits) < cfg.n_fft) ++bits;
  rev_.assign(cfg.n_fft, 0u);
  for (uint32_t i = 0; i < cfg.n_fft; ++i) {
    uint32_t r = 0;
    for (uint32_t b = 0; b < bits; ++b)
      if (i & (1u << b)) r |= 1u << (bits - 1 - b);
    rev_[i] = r;
  }

  tw_cos_.assign(cfg.n_fft / 2, 0.0f);
  tw_sin_.assign(cfg.n_fft / 2, 0.0f);
  for (uint32_t k = 0; k < cfg.n_fft / 2; ++k) {
    const double ang = -2.0 * kPi * static_cast<double>(k) / cfg.n_fft;
    tw_cos_[k] = static_cast<float>(std::cos(ang));
    tw_sin_[k] = static_cast<float>(std::sin(ang));
  }
  return true;
}

size_t MelSpectrogram::validFrames(size_t num_samples) const {
  if (cfg_.hop == 0) return 0;
  return num_samples / cfg_.hop;
}

size_t MelSpectrogram::frames(size_t num_samples) const {
  return validFrames(num_samples) + 1;
}

void MelSpectrogram::fftPower(const float* pcm, size_t num_samples, size_t frame,
                              float* power) {
  const uint32_t n = cfg_.n_fft;
  std::memset(re_.data(), 0, n * sizeof(float));
  std::memset(im_.data(), 0, n * sizeof(float));

  // Centered framing: frame t starts at t*hop - win_length/2. Samples outside
  // the clip are zeros (boundary="zeros").
  const int64_t start =
      static_cast<int64_t>(frame) * cfg_.hop - static_cast<int64_t>(cfg_.win_length / 2);
  for (uint32_t i = 0; i < cfg_.win_length; ++i) {
    const int64_t idx = start + i;
    if (idx >= 0 && static_cast<size_t>(idx) < num_samples)
      re_[i] = pcm[idx] * window_[i];
  }

  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t j = rev_[i];
    if (j > i) {
      const float tr = re_[i], ti = im_[i];
      re_[i] = re_[j];
      im_[i] = im_[j];
      re_[j] = tr;
      im_[j] = ti;
    }
  }

  for (uint32_t len = 2; len <= n; len <<= 1) {
    const uint32_t half = len >> 1;
    const uint32_t step = n / len;
    for (uint32_t i = 0; i < n; i += len) {
      for (uint32_t j = 0; j < half; ++j) {
        const uint32_t k = j * step;
        const float wr = tw_cos_[k], wi = tw_sin_[k];
        const uint32_t a = i + j, b = a + half;
        const float tr = re_[b] * wr - im_[b] * wi;
        const float ti = re_[b] * wi + im_[b] * wr;
        re_[a] += tr;
        im_[a] += ti;
        re_[b] = re_[a] - 2.0f * tr;
        im_[b] = im_[a] - 2.0f * ti;
      }
    }
  }

  // The reference multiplies the windowed STFT by sum(window) before squaring.
  const float scale = window_sum_ * window_sum_;
  for (uint32_t k = 0; k < pow_bins_; ++k)
    power[k] = (re_[k] * re_[k] + im_[k] * im_[k]) * scale;
}

void MelSpectrogram::compute(const float* pcm, size_t num_samples, float* mel) {
  const size_t total = frames(num_samples);
  const size_t valid = validFrames(num_samples);
  const size_t mels = cfg_.n_mels;

  // Phase 1: power spectrum of every frame, written to a contiguous buffer so
  // the next phase reads it sequentially instead of re-gathering per bin.
  powers_.assign(total * pow_bins_, 0.0f);
  for (size_t t = 0; t < total; ++t)
    fftPower(pcm, num_samples, t, powers_.data() + t * pow_bins_);
  if (progress_ != nullptr) progress_(progress_udata_, "mel_fft");

  // Mel projection over the packed bank, then log. Single precision on purpose:
  // neither the P4 nor the S3 has a double FPU, so a double accumulator is
  // software-emulated per multiply-add.
  raw_.assign(total * mels, 0.0f);
  for (size_t t = 0; t < total; ++t) {
    const float* pw = powers_.data() + t * pow_bins_;
    float* out = raw_.data() + t * mels;
    for (size_t m = 0; m < mels; ++m) {
      const float* wgt = fb_pack_.data() + fb_off_[m];
      const float* p = pw + fb_lo_[m];
      const uint32_t n = fb_len_[m];
      float acc = 0.0f;
      for (uint32_t k = 0; k < n; ++k) acc += wgt[k] * p[k];
      out[m] = std::log(acc + cfg_.log_floor);
    }
  }
  if (progress_ != nullptr) progress_(progress_udata_, "mel_fb");

  if (valid == 0) {
    std::memset(mel, 0, total * mels * sizeof(float));
    if (progress_ != nullptr) progress_(progress_udata_, "mel_norm");
    return;
  }

  std::vector<double> mean(mels, 0.0), var(mels, 0.0);
  for (size_t t = 0; t < valid; ++t)
    for (size_t m = 0; m < mels; ++m) mean[m] += raw_[t * mels + m];
  for (size_t m = 0; m < mels; ++m)
    mean[m] /= static_cast<double>(valid);

  for (size_t t = 0; t < valid; ++t)
    for (size_t m = 0; m < mels; ++m) {
      const double d = raw_[t * mels + m] - mean[m];
      var[m] += d * d;
    }
  const double denom = (valid > 1) ? static_cast<double>(valid - 1) : 1.0;

  std::vector<double> inv(mels, 0.0);
  for (size_t m = 0; m < mels; ++m)
    inv[m] = 1.0 / (std::sqrt(var[m] / denom) + 1e-5);

  for (size_t t = 0; t < valid; ++t)
    for (size_t m = 0; m < mels; ++m)
      mel[t * mels + m] =
          static_cast<float>((raw_[t * mels + m] - mean[m]) * inv[m]);
  // The reference re-applies the frame mask after normalization.
  for (size_t t = valid; t < total; ++t)
    for (size_t m = 0; m < mels; ++m) mel[t * mels + m] = 0.0f;
  if (progress_ != nullptr) progress_(progress_udata_, "mel_norm");
}

}  // namespace whistle
