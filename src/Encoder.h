#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "AudioLayout.h"
#include "MelSpectrogram.h"

namespace cactus {
class CactArchive;
}

namespace whistle {

class ModelWeights;

// Whistle audio encoder: log-mel -> convolutional stem -> 8 encoder blocks
// (4 mHC lanes) -> [frames][512] memory for the decoder's cross attention.
class Encoder {
 public:
  Encoder() = default;

  // `weights` must already be prepared for `archive` and outlive this object.
  bool prepare(const cactus::CactArchive& archive, const ModelWeights& weights);

  bool ready() const { return ready_; }
  const AudioLayout& layout() const { return layout_; }
  uint32_t width() const { return layout_.d_model; }

  // PCM 16 kHz mono in [-1,1]. On success `mem` holds `frames * width()` floats
  // (row-major [frame][channel]) and `frames` is the number of valid encoder
  // frames: ceil(floor(N/hop) / 8).
  bool encode(const float* pcm, size_t num_samples, std::vector<float>* mem,
              size_t* frames);

  // Optional per-stage progress hook (mel, stem, enc_lN). nullptr disables.
  using ProgressFn = void (*)(void* udata, const char* tag);
  void setProgress(ProgressFn fn, void* udata) {
    progress_ = fn;
    progress_udata_ = udata;
  }

  // Test hook: runs the stem on a precomputed mel buffer (frames x mels).
  bool stemDebug(const float* mel, size_t mel_frames, size_t valid_mel,
                 std::vector<float>* out, size_t* out_frames,
                 size_t* valid_out);

  // Optional per-stage debug hook: called with the stem output, each layer's
  // lane stream, the pre-norm mean, and the final encoder output. nullptr
  // disables. The sketch uses this to print checksums on device.
  using DebugFn = void (*)(void* udata, const char* tag, const float* x,
                           size_t n);
  void setDebug(DebugFn fn, void* udata) {
    debug_ = fn;
    debug_udata_ = udata;
  }

 private:
  bool stem(const float* mel, size_t mel_frames, size_t valid_mel,
            std::vector<float>* out, size_t* out_frames, size_t* valid_out);
  void encoderForward(const float* x, size_t frames, size_t valid,
                      std::vector<float>* out);
  void encoderBlock(const float* x, size_t T, const float* seen, uint32_t layer,
                    float* out);
  void selfAttention(const float* x, size_t T, const float* seen, uint32_t layer,
                     float* out);
  void hadamardMlp(const float* x, size_t T, uint32_t layer, uint32_t which,
                   float* out);

  bool readNorm(uint32_t index, float* dst) const;
  bool dense(uint32_t index, const float* x, size_t M, uint32_t in,
             uint32_t out_count, float* y) const;
  // Emits "e<layer><suffix>" (m0, at, cv, m1, mh) so a layer's stages can be
  // attributed without the per-group fwht/buck tags.
  void progressStage(uint32_t layer, const char* suffix) const;

  // Persistent attention workspace. The block and Hadamard-MLP buffers stay
  // per-call; hoisting them measurably slowed the MLP stages.
  struct Scratch {
    std::vector<float> xq, q, k, v, gate, satmp, qn, kn, cos_t, sin_t, scores,
        attn;
    // RoPE tables depend only on T and theta, so every block reuses them.
    size_t rope_frames = 0;
    // One pool for the Hadamard-MLP tile temporaries (cond/z/ztmp); a request
    // above the 16 KB internal-SRAM threshold lands in PSRAM deterministically.
    std::vector<float> mlpTmp;
    // kronApply scratch, persistent.
    std::vector<float> kronTmp;
    // Staging for the six Kronecker factor matrices (one per core under the
    // dual-core path, so the cores never share factor lines).
    std::vector<float> kronW;
    // Second-core mirrors of the MLP temporaries for the dual-core MLP path.
    std::vector<float> mlpTmp2;
    std::vector<float> kronTmp2;
    std::vector<float> kronW2;
    // Per-block buffers, allocated once; each is written before it is read.
    std::vector<float> b_hn, b_tmp, b_at, b_pw1, b_h, b_hconv, b_y, b_dw,
        b_nscale;
  };
  Scratch scratch_;

  AudioLayout layout_;
  const ModelWeights* weights_ = nullptr;
  MelSpectrogram mel_;
  std::vector<float> filterbank_;

  // Stem weights in fp32 (decoded once). The three depthwise kernels are fp16
  // in the archive; the pointwise and output kernels are CQ2.
  std::vector<float> stem_w_, stem_dw1_, stem_dw2_;
  std::vector<float> stem_pw1_, stem_pw2_, stem_out_;

  // Encoder mHC parameters.
  std::vector<float> mhc_a_pre_, mhc_a_post_, mhc_a_res_;
  std::vector<float> mhc_b_pre_, mhc_b_post_, mhc_b_res_;
  std::vector<float> final_norm_;
  std::vector<int32_t> perm1_, perm2_;

  // Hadamard-MLP tensors, constant per (block, half).
  struct MlpWeights {
    std::vector<float> d1, d2, b2, d3, d4;
    std::vector<float> w1a, w1b, w2a, w2b, w3a, w3b;
    std::vector<float> cond_v, cond_u;
    uint32_t w1a0 = 0, w1a1 = 0, w1b0 = 0, w1b1 = 0;
    uint32_t w2a0 = 0, w2a1 = 0, w2b0 = 0, w2b1 = 0;
    uint32_t w3a0 = 0, w3a1 = 0, w3b0 = 0, w3b1 = 0;
    bool loaded = false;
  };
  mutable std::vector<MlpWeights> mlpW_;
  // Reads and caches one (block, half)'s Hadamard-MLP tensors. Called eagerly
  // from prepare() so the first utterance does not pay for it.
  bool ensureMlpWeights(uint32_t layer, uint32_t which);
  // Tile loop of hadamardMlp over frames [r0, r1) (see Encoder.cpp). The per-core
  // temporaries are passed in so the dual-core path can hand each worker its own
  // buffers; a single-core call passes r0=0, r1=T and the scratch_ buffers.
  void mlpTiles(const MlpWeights& mw, const float* x, size_t r0, size_t r1,
                float* out, uint32_t d, uint32_t nn, uint32_t tile,
                float* mlpTmp, float* kronTmp, float* kronW, uint32_t layer,
                uint32_t which);
  // Context for the dual-core MLP dispatch: per-core temporaries plus the
  // parameters mlpTiles needs. buf[0] is the caller's set (scratch_), buf[1] the
  // worker's mirror (scratch_ *2); the trampoline picks one by r0.
  struct MlpJob {
    Encoder* self;
    const MlpWeights* mw;
    const float* x;
    uint32_t d, nn, tile;
    uint32_t layer, which;
    float* buf[2][3];
  };
  static void mlpJobImpl(void* ctx, uint32_t r0, uint32_t r1, float* out);

  mutable std::vector<int8_t> qScratch_;
  mutable std::vector<float> scaleScratch_;

  DebugFn debug_ = nullptr;
  void* debug_udata_ = nullptr;
  ProgressFn progress_ = nullptr;
  void* progress_udata_ = nullptr;

  bool ready_ = false;
};

}  // namespace whistle
