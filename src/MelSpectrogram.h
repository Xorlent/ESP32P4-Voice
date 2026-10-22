#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Whistle log-mel front end.
//
// Transform:
//   wav            masked by the valid-sample count
//   window         np.hanning(win_length) (symmetric)
//   stft           nperseg=win_length, noverlap=win_length-hop, nfft=n_fft,
//                  boundary="zeros", padded=False, detrend=False
//   power          |Z * sum(window)|^2
//   mel            power @ filterbank
//   log            natural log(mel + 2^-24)
//   normalize      per-clip (x - mean) / (std + 1e-5) over valid frames
//
// The STFT is centered: frame t starts at t*hop - win_length/2, so the signal
// is zero-padded by win_length/2 at both ends. That produces 1 + floor(N/hop)
// frames; the trailing frame is computed, excluded from the mean/std, and
// finally zeroed by the mask, exactly like the reference. Valid frames are
// floor(N/hop).
//
// The filterbank is the model's fp32 [n_fft/2 + 1][n_mels] tensor, not a
// reconstructed bank.
namespace whistle {

struct MelConfig {
  uint32_t n_mels = 80;
  uint32_t n_fft = 512;
  uint32_t win_length = 400;   // 25 ms @ 16 kHz, symmetric Hann
  uint32_t hop = 160;          // 10 ms
  uint32_t sample_rate = 16000;

  // Added inside the log: log(mel + log_floor), 2^-24 as in the reference.
  float log_floor = 5.9604644775390625e-8f;
};

class MelSpectrogram {
 public:
  MelSpectrogram() = default;

  // Copies cfg and stores filterbank (row-major [pow_bins][n_mels] float32,
  // pow_bins == n_fft/2 + 1). Returns false on bad geometry.
  bool init(const MelConfig& cfg, const float* filterbank);

  // Number of STFT frames produced for num_samples: 1 + floor(N / hop).
  // Callers allocate frames() * cfg.n_mels floats.
  size_t frames(size_t num_samples) const;

  // Number of frames that are actually valid (not masked): floor(N / hop).
  size_t validFrames(size_t num_samples) const;

  // Computes log-mel into mel (caller-allocated frames()*cfg_.n_mels float32,
  // row-major [frame][mel]). pcm is 16 kHz mono float in [-1,1].
  void compute(const float* pcm, size_t num_samples, float* mel);

  // Phase tags for on-device profiling (once per phase, not per frame).
  void setProgress(void (*cb)(void*, const char*), void* udata) {
    progress_ = cb;
    progress_udata_ = udata;
  }

 private:
  void fftPower(const float* pcm, size_t num_samples, size_t frame, float* power);

  MelConfig cfg_;
  size_t pow_bins_ = 0;
  float window_sum_ = 0.0f;
  std::vector<float> window_;       // symmetric Hann, win_length entries
  // Transposed and support-trimmed filterbank, packed: bin m lives at
  // fb_pack_[fb_off_[m] .. fb_off_[m] + fb_len_[m]) and covers power bins
  // [fb_lo_[m], fb_lo_[m] + fb_len_[m]). Only exact zeros are dropped, and the
  // remaining terms stay in ascending f, so the float sum is unchanged. Packing
  // turns the old stride-n_mels gather over the whole bank into a few KB read
  // sequentially per frame.
  std::vector<float> fb_pack_;
  std::vector<uint32_t> fb_off_;
  std::vector<uint32_t> fb_len_;
  std::vector<uint32_t> fb_lo_;
  std::vector<uint32_t> rev_;       // FFT bit reversal
  std::vector<float> tw_cos_, tw_sin_;  // FFT twiddles, n_fft/2 entries
  std::vector<float> re_, im_;      // FFT scratch, n_fft entries
  std::vector<float> powers_;       // frames * pow_bins, all frames
  std::vector<float> raw_;          // frames * n_mels, log-mel before norm

  // Optional phase tags ("mel_fft", "mel_fb", "mel_norm"); nullptr disables.
  void (*progress_)(void*, const char*) = nullptr;
  void* progress_udata_ = nullptr;
};

}  // namespace whistle
