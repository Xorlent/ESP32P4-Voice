#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "CactArchive.h"
#include "fp16.h"

// Platform-independent weight access for the needle/whistle .cact model.
//
// Provides on-demand dequantization of CQ tensors and conversion of fp16/fp32
// tensors into float32 buffers, keyed by symbolic names (see symbolicNames()).
// Free of Arduino/ESP-IDF dependencies.

namespace whistle {

struct ModelGeometry {
  uint32_t vocab = 0;
  uint32_t out_vocab = 0;
  uint32_t d_model = 0;
  uint32_t num_heads = 0;
  uint32_t num_kv_heads = 0;
  uint32_t num_layers = 0;
  uint32_t qk_head_dim = 0;
  uint32_t v_head_dim = 0;
  uint32_t hada_n = 0;
  uint32_t mhc_lanes = 0;
  uint32_t qkv_conv_taps = 0;
  uint32_t engram_slots = 0;
  uint32_t engram_sub_dim = 0;
  uint32_t num_engram_tables = 0;
  uint32_t engram_conv_taps = 0;
  uint32_t engram_conv_dilation = 0;
  uint32_t num_engram_orders = 0;
  uint32_t engram_orders[4] = {0, 0, 0, 0};
  uint32_t num_engram_sites = 0;
  uint32_t engram_sites[16] = {0};
  float rope_theta = 10000.0f;
  size_t tensor_count = 0;
};

class ModelWeights {
 public:
  ModelWeights() = default;

  // `archive` must outlive this object. Derives geometry + the symbolic name
  // mapping from the parsed header.
  bool prepare(const cactus::CactArchive& archive);

  const ModelGeometry& geometry() const { return geom_; }
  const cactus::CactArchive& archive() const { return *archive_; }

  // Resolves a symbolic name to a directory index, or -1 if unknown.
  int indexOf(const char* name) const;

  // Returns the tensor meta for a symbolic name, or nullptr.
  const cactus::TensorMeta* meta(const char* name) const;

  // Directory-index access for the nameless audio region (see AudioLayout).
  const cactus::TensorMeta* metaAt(size_t index) const;

  // Dequantizes/reads the named tensor into `out` (exactly `elements` floats,
  // in row-major order). Returns false if unknown/tensor absent. `elements`
  // must equal the logical element count (product of the stored shape).
  bool readTensor(const char* name, float* out, size_t elements) const;

  // Same, addressed by directory index.
  bool readIndex(size_t index, float* out, size_t elements) const;

  // Convenience: returns whether `name` exists.
  bool has(const char* name) const { return indexOf(name) >= 0; }

  // Optional row-parallel hook for the CQ GEMV. fn(ctx, r0, r1) must process
  // output rows [r0, r1) and return only when finished; the platform layer may
  // run one half on a second core. `sync` describes the buffers the two cores
  // share, so a platform with non-coherent L1 caches can write back/invalidate
  // them around the handoff. Null (default) keeps it inline/single-core.
  struct RowsSync {
    const void* ro = nullptr;  // shared read-only input (may be null)
    size_t roBytes = 0;
    void* rw = nullptr;        // caller output: rwRows rows of rwStride floats
    size_t rwStride = 0;
    size_t rwRows = 0;
  };
  // `out` is the base for row r0 and `outStride` the stride between mm rows.
  using RowFn = void (*)(void* ctx, uint32_t r0, uint32_t r1, float* out,
                         size_t outStride);
  using RowsHook = void (*)(RowFn fn, void* ctx, uint32_t nrows,
                            const RowsSync& sync);
  static void setRowsHook(RowsHook hook) { rowsHook_ = hook; }
  // Largest worker staging (floats) the platform hook will accept, 0 = none.
  static void setRowsStaging(size_t floats) { rowsStaging_ = floats; }

  // Dequantize a single CQ output row (`row` of shape[0]) into `out`, which
  // must hold `shape[1]` floats (tightly packed, no padding). Used to fetch
  // engram table rows without materializing the whole (tables*slots, sub_dim)
  // tensor. Returns false if the tensor is not CQ or `row` is out of range.
  bool readCqRow(const char* name, uint32_t row, float* out) const;

  // Dequant-free CQ matrix-vector product: y[m, n] = sum_k x[m,k] * W[out_begin+n, k]
  // for the CQ tensor `name` [out=N, in], via the FWHT-symmetry bucket identity
  // (FWHT of the activation once per group, bucket by code, codebook dot). No
  // fp32 weight is ever materialized, replacing the O(groups*128*128) per-row
  // Hadamard reconstruction with O(groups*128) adds. x is (M x K) row-major,
  // y is (M x out_count) row-major. `out_begin` slices multi-layer tensors
  // (mhc.phi_*). Returns false on non-CQ / out-of-range.
  bool gemvCq(const char* name, uint32_t out_begin, uint32_t out_count,
              const float* x, size_t M, float* y) const;

  // Same, addressed by directory index.
  bool gemvCqIndex(size_t index, uint32_t out_begin, uint32_t out_count,
                   const float* x, size_t M, float* y) const;

  // int8-activation CQ GEMV. q is (M x K) int8 (per-row fake-quant),
  // scale is (M) floats (the per-row fake_quant scale). Pre-decodes the packed
  // indices once per call, computes int16 FWHT of q per group, int32 buckets,
  // then a float reconstruction:
  //   y[m,n] = scale[m] * inv_sqrt(128) * sum_g norm[g] * sum_c cb[c]*bucket[g][c].
  // Matches the float bucket path to the int8 activation rounding floor.
  bool gemvCqInt8(const char* name, uint32_t out_begin, uint32_t out_count,
                  const int8_t* q, const float* scale, size_t M, float* y) const;

  // Same, addressed by directory index.
  bool gemvCqInt8Index(size_t index, uint32_t out_begin, uint32_t out_count,
                       const int8_t* q, const float* scale, size_t M,
                       float* y) const;

  // Optional per-phase profiling hook for gemvCqInt8 (on-device timing). See the
  // `progress` member below; nullptr disables.
  void setProgress(void (*cb)(void*, const char*), void* udata) const {
    progress = cb;
    progressUdata = udata;
  }

 private:
  const cactus::CactArchive* archive_ = nullptr;
  ModelGeometry geom_;
  const float* codebooks_ = nullptr;  // 28 floats: cb2[4] cb3[8] cb4[16]
  // Fixed-point (Q15) codebooks: cbQ15[c] = round(cb[c] * 32768), shared and
  // analytic (Lloyd-Max / sqrt(group)); computed once in prepare(). Lets the
  // per-group weight w[k] = cb[idx[k]] be an int16 LUT so the hot dot is an
  // int16 x int16 -> int40 accumulation (VMULAS.S16.XACC) instead of float FMA.
  int16_t cbQ15_[28] = {0};
  // Decode LUT for 4-bit weights: lut4_[b] packs the two adjacent Q15 codebook
  // entries for one packed byte (low 16 bits = cbQ15[b&15], high = cbQ15[b>>4]),
  // so the per-group index unpack becomes one 256-entry lookup per byte instead
  // of word assembly + 8 variable shifts. Computed once in prepare().
  int32_t lut4_[256] = {};
  // Decode LUT for 2-bit weights: byte b holds four indices (LSB-first), so
  // lut2_[4*b .. 4*b+3] are their Q15 codebook values (a 128 group = 32 bytes).
  int16_t lut2_[1024] = {};

  // int8 mirror of the codebooks, used by the s8 GEMV path. Each bit class is
  // scaled so its largest level lands on 127; cb8Inv_[class] maps an int8 level
  // back to the codebook value. lut4_8_/lut2_8_ pack a whole source byte's
  // levels so a 128-wide group decodes with 32 (2-bit) or 64 (4-bit) stores.
  int8_t cb8_[28] = {0};
  float cb8Inv_[3] = {1.0f, 1.0f, 1.0f};  // 2-bit, 3-bit, 4-bit
  uint32_t lut2_8_[256] = {};
  uint16_t lut4_8_[256] = {};
  mutable std::vector<int8_t> q8Scratch_;  // M x group int8 activations
  mutable std::vector<int8_t> rowScratch_;  // one decoded weight row (M == 1 path)
  // Decoded-weight-group scratch, one per core (the row hook runs the second
  // half on core 1). Padded above the 16 KB internal-SRAM threshold on purpose:
  // it is written 128 bytes at a time and then read by the dot, for every output
  // row, so it wants to be PSRAM-backed and L1-resident.
  mutable std::vector<int8_t> wqRow_[2];
  // The fused M == 1 path is self-validating: the first time it runs, its group
  // sums are compared against the scalar reference dots. A wrong PIE kernel then
  // costs speed instead of silently corrupting logits (that happened once).
  mutable bool fusedChecked = false;
  mutable bool fusedOk = true;
  mutable std::vector<float> sxScratch_;   // M activation group scales
  static RowsHook rowsHook_;
  static size_t rowsStaging_;

  // Precomputed normalized Hadamard of the CQ group size (128): H_raw / sqrt(128).
  // Heap-allocated (128*128 floats = 64 KB) so a ModelWeights instance stays
  // small and never blows a task stack.
  std::vector<float> hadamard_;

  // Reused scratch for gemvCqInt8 (avoid per-call malloc/free churning the heap
  // across ~68 matmuls per forward). With the group-outer loop this is only
  // M x group int16 (~10 KB for the encoder), small enough for the normal
  // internal heap.
  mutable std::vector<int16_t> fwScratch_;
  mutable std::vector<int16_t> wbScratch_;
  mutable std::vector<float> accScratch_;

  // Optional per-phase profiling hook for gemvCqInt8 (on-device timing). Fires
  // "dec" (after pre-decode), "fwht" (after Hadamard), "buck" (after the bucket
  // + reconstruction loop). nullptr disables; udata is passed through.
  mutable void (*progress)(void* udata, const char* tag) = nullptr;
  mutable void* progressUdata = nullptr;

  // Internal implementations shared by the name and directory-index wrappers.
  bool readMeta(const cactus::TensorMeta* m, float* out, size_t elements) const;
  bool gemvCqMeta(const cactus::TensorMeta* m, uint32_t out_begin,
                  uint32_t out_count, const float* x, size_t M, float* y) const;
  bool gemvCqInt8Meta(const cactus::TensorMeta* m, uint32_t out_begin,
                      uint32_t out_count, const int8_t* q, const float* scale,
                      size_t M, float* y) const;

  // Dequantize a CQ blob into `out` (tightly packed, stride = in).
  bool dequantCq(const cactus::TensorMeta* m, const uint8_t* blob, float* out) const;
  // Dequantize a single CQ output row into `out` (in = shape[1] floats).
  void dequantCqRowImpl(const cactus::TensorMeta* m, const uint8_t* blob,
                        uint32_t r, float* out) const;
};

}  // namespace whistle