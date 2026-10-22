#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "AudioLayout.h"
#include "ModelWeights.h"

// Platform-independent C++ port of the needle/whistle decoder forward pass.
// Produces greedy logits for a token sequence; no Arduino/ESP-IDF deps.
//
// All math is float32. Weight tensors are dequantized on demand and cached,
// except the engram tables, which are fetched row-by-row to avoid materializing
// them.

namespace whistle {

class Decoder {
 public:
  bool prepare(const cactus::CactArchive& archive);

  const ModelWeights& weights() const { return weights_; }

  // Runs the forward pass over `T` token ids; writes `T * vocab` float32
  // logits (row-major [t][vocab]). Returns false if not prepared.
  bool forward(const int32_t* tokens, size_t T, float* logits) const;

  // Speech-conditioned forward: adds the encoder memory `mem` [S][d] to every
  // decoder layer through cross attention. `mem` must already include the
  // positional sinusoids and int8 activation quantization. `head_row` >= 0
  // computes only that row of logits (greedy decode only reads the last row);
  // -1 computes every row.
  bool forwardWithMemory(const int32_t* tokens, size_t T, const float* mem,
                         size_t S, const AudioLayout& layout, float* logits,
                         int32_t head_row = -1) const;

  // Dequantizes the named tensor (cached). Returns nullptr on failure.
  const float* tensor(const char* name, size_t* elements) const;

  // When on, forward() avoids materializing the fp32 embedding (rows fetched on
  // demand, output head streamed) and evicts per-layer tensors from the dequant
  // cache once used. Keeps peak dequant memory within roughly one layer + the
  // shared hyper-connection tensors, so a full forward fits the ~15 MB of PSRAM
  // left after staging whistle.cact. Produces bit-identical logits.
  void setBounded(bool b) { bounded_ = b; }
  bool bounded() const { return bounded_; }

  // When on, CQ weight matmuls use the dequant-free FWHT-bucket GEMV
  // (ModelWeights::gemvCq) instead of materializing fp32 weights.
  void setBucket(bool b) { bucket_ = b; }
  bool bucket() const { return bucket_; }

  // int8-activation CQ matmuls (fake-quant to int8 + int16 FWHT + int32 buckets
  // + float reconstruction). Implies the bucket path in fixed point.
  void setInt8(bool b) { int8_ = b; }
  bool int8() const { return int8_; }

  // Decoder ladder depth: 0 = full (the archive's num_layers), otherwise run
  // the depth rung of 2..num_layers blocks selected in ladder order (both
  // endpoint blocks plus bisected interiors), which is how the model was
  // trained. A shallower rung trades accuracy for latency.
  void setDepth(uint32_t depth) { depth_ = depth; }
  uint32_t depth() const { return depth_; }

  // Preprojects the utterance memory K/V once per active block so the greedy
  // loop does not repeat it at every step. Cached as int8 codes + per-row
  // scales (the exact grid fakeQuantRows produces, so the dequantized values
  // are bit-identical). Returns the number of cached blocks, or 0 with the
  // cache invalidated; forwardWithMemory then projects K/V per call.
  uint32_t prepareCrossAttention(const float* mem, size_t S,
                                 const AudioLayout& layout) const;
  // Releases the cached memory K/V (call after an utterance's last step).
  void dropCrossKv() const;

  // ---- Incremental (one token at a time) decode -------------------------
  // Keeps per-block self-attention K/V (post-norm/rope/quantized), the taps-1
  // pre-conv q/k/v rows and a per-site engram value ring, so a greedy step costs
  // one token instead of the whole prefix. `prepareCrossAttention` must have run
  // for the same utterance. beginSequence() returns false when it cannot
  // allocate; forwardStep() returns false when the token cap is exceeded (the
  // caller can fall back to the batched path, whose numerics are identical).
  bool beginSequence(const AudioLayout& layout) const;
  bool forwardStep(const int32_t* tokens, size_t n, const float* mem,
                   size_t memS, const AudioLayout& layout,
                   float* logits) const;
  void endSequence() const;
  bool incremental() const { return step_.active; }

  // When enabled, forward() records engram k/v (S*T*d each) and the mean over
  // lanes after every layer (L*T*d) into the dbg* buffers, for validation.
  mutable bool captureDebug = false;
  mutable std::vector<float> dbgEk, dbgEv, dbgLayers;

  // Optional per-stage profiling hook (for on-device timing). Fired with a
  // short tag at key points ("pre" before layers, "l<N>" after each layer,
  // "head" after the embedding head). udata is passed through; nullptr disables.
  mutable void (*progress)(void* udata, const char* tag) = nullptr;
  mutable void* progressUdata = nullptr;

 private:
  struct StepState;

  bool forwardImpl(const int32_t* tokens, size_t T, const float* mem,
                   size_t memS, const AudioLayout* audio, float* logits,
                   int32_t headRow, StepState* step = nullptr) const;

  ModelWeights weights_;
  // Dequant cache keyed by tensor index.
  mutable std::unordered_map<int, std::vector<float>> cache_;
  bool bounded_ = false;
  bool bucket_ = false;
  bool int8_ = false;
  uint32_t depth_ = 0;

  // Post-norm, post-quantization cross-attention K/V for one block: one row per
  // memory frame and head, so the per-step gate only has to dequantize.
  struct CrossKv {
    bool valid = false;
    size_t memS = 0;
    std::vector<int8_t> k, v;
    std::vector<float> kscale, vscale;
  };
  mutable std::vector<CrossKv> crossKv_;

  // Per-block incremental state: self-attention K/V (one row per position) and
  // the taps-1 pre-conv q/k/v rows needed by the causal conv.
  struct LayerStep {
    std::vector<float> kcache;  // cap * KV * qk
    std::vector<float> vcache;  // cap * KV * vd
    std::vector<float> qhist;   // (taps-1) * H * qk
    std::vector<float> khist;   // (taps-1) * KV * qk
    std::vector<float> vhist;   // (taps-1) * KV * vd
  };
  struct StepState {
    bool active = false;
    uint32_t cap = 0;
    size_t pos = 0;
    std::vector<LayerStep> layers;        // indexed by original block
    std::vector<std::vector<float>> vv;   // per site: ring of value rows
    uint32_t vvRows = 0;                  // (taps-1)*dilation + 1
    std::vector<float> sc, pr;            // scores / probabilities scratch
  };
  mutable StepState step_;

  // Reusable forward workspace; persistent members retain capacity across the
  // per-step greedy calls instead of re-allocating each call.
  struct Scratch {
    std::vector<float> x, erow, ngramOk, ek, ev, featch, kk, vv, evconv;
    std::vector<float> cos_t, sin_t;
    std::vector<float> xq, qflat, qconv, kflat, kconv, vflat, vconv;
    std::vector<float> qhead, khead, vhead, scores, attn, oh, atmp, gate, ztmp;
    std::vector<float> xc, cond, z, bx, resid, hn, at, hn2, hmo;
    std::vector<float> pre_off, post_off, site_flags;
    std::vector<float> stream, xf, nx, hpre, hpost, hres, u, y, blk, res, kkmat,
        newstream;
    std::vector<float> fx, kronScratch;
    // cross_attn workspace: allocated once per block/step instead of per call,
    // which was the remaining run-to-run swing in the `hada` delta.
    std::vector<float> c_xq, c_q, c_k, c_v, c_qn, c_kn, c_gate, c_oh, c_sc,
        c_pr, c_qscale, c_mscale, c_ohscale;
    std::vector<int8_t> c_q8, c_mem8, c_oh8;
    std::vector<float> featch_scale, xq_scale, atmp_scale, nx_scale, fx_scale;
    std::vector<int8_t> featch8, xq8, atmp8, nx8, fx8;
    std::vector<int32_t> idx, p1, p2;
  };
  mutable Scratch scratch_;

  const float* cachedTensor(const char* name, size_t* elements) const;
  void evictRange(int begin, int end) const;
};

}  // namespace whistle