#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Minimal parser for the Cactus ".cact" container (Needle 3 format; used by
// both Needle and Whistle). The file is designed to be memory-mapped and read
// in place: a fixed 196-byte header, a codebook table, a nameless tensor
// directory, the raw weight blobs, then a SentencePiece tokenizer.
//
// Reference: https://cactuscompute.com/blog/cact-format

namespace cactus {

inline constexpr uint32_t kNeedle2Tag = 0x05E12A83u;
inline constexpr uint32_t kNeedle3Tag = 0x05E12A84u;

enum class DType : uint8_t {
  FP16 = 1,
  FP32 = 2,
  CQ = 3,
  RAW = 4,
};

// 49 x u32 fields (last field rope_theta is an f32 stored in its slot), in the
// exact order written by the Cactus container:
//   tag, num_tensors, codebook_len, kv_window, kv_bits, vocab, out_vocab,
//   d_model, num_heads, num_kv_heads, num_layers, qk_head_dim, v_head_dim,
//   max_seq_len, hada_n, mhc_lanes, sliding_window, global_mask_lo,
//   global_mask_hi, qkv_conv_taps, engram_slots, engram_sub_dim,
//   num_engram_tables, engram_conv_taps, engram_conv_dilation,
//   engram_seed_heads, num_engram_orders, engram_orders[4], num_engram_sites,
//   engram_sites[16], rope_theta(f32).
struct CactHeader {
  uint32_t tag = 0;
  uint32_t num_tensors = 0;
  uint32_t codebook_len = 0;  // in floats (follows the header)
  uint32_t f[49] = {0};

  // Geometry accessors (byte offsets in the header, converted to u32 index).
  uint32_t kvWindow() const { return f[3]; }
  uint32_t kvBits() const { return f[4]; }
  uint32_t vocab() const { return f[5]; }
  uint32_t outVocab() const { return f[6]; }
  uint32_t width() const { return f[7]; }
  uint32_t numHeads() const { return f[8]; }
  uint32_t numKvHeads() const { return f[9]; }
  uint32_t numLayers() const { return f[10]; }
  uint32_t qkHeadDim() const { return f[11]; }
  uint32_t vHeadDim() const { return f[12]; }
  uint32_t maxSeqLen() const { return f[13]; }
  uint32_t hadaN() const { return f[14]; }
  uint32_t mhcLanes() const { return f[15]; }
  uint32_t slidingWindow() const { return f[16]; }
  uint32_t globalMaskLo() const { return f[17]; }
  uint32_t globalMaskHi() const { return f[18]; }
  uint32_t qkvConvTaps() const { return f[19]; }
  uint32_t engramSlots() const { return f[20]; }
  uint32_t engramSubDim() const { return f[21]; }
  uint32_t numEngramTables() const { return f[22]; }
  uint32_t engramConvTaps() const { return f[23]; }
  uint32_t engramConvDilation() const { return f[24]; }
  uint32_t engramSeedHeads() const { return f[25]; }
  uint32_t numEngramOrders() const { return f[26]; }
  uint32_t engramOrder(int i) const { return f[27 + i]; }
  uint32_t numEngramSites() const { return f[31]; }
  uint32_t engramSite(int i) const { return f[32 + i]; }
  float ropeTheta() const {
    return *reinterpret_cast<const float*>(&f[48]);
  }
};

struct TensorMeta {
  DType dtype = DType::FP16;
  uint8_t ndim = 0;
  uint32_t shape[4] = {0, 0, 0, 0};
  uint64_t offset = 0;  // absolute offset into the mapped archive
  uint64_t nbytes = 0;
  uint32_t group = 0;
  uint32_t bits = 0;
};

class CactArchive {
 public:
  // Parses the header, codebooks and directory in place. `data` must remain
  // valid (mapped and alive) for the lifetime of this object. Returns false
  // if the tag is not a recognised Needle archive or the file is truncated.
  bool parse(const uint8_t* data, size_t size);

  bool valid() const { return parsed_; }

  const CactHeader& header() const { return header_; }
  size_t tensorCount() const { return num_tensors_; }
  const TensorMeta* tensor(size_t i) const;

  // Raw mapped base pointer and byte size (for dequantizing blobs in place).
  const uint8_t* base() const { return base_; }
  size_t size() const { return size_; }

  // 28 floats: 4/8/16-entry Lloyd-Max codebooks for 2/3/4-bit weights.
  const float* codebooks() const { return codebooks_; }

 private:
  const uint8_t* base_ = nullptr;
  size_t size_ = 0;
  bool parsed_ = false;
  CactHeader header_;
  const float* codebooks_ = nullptr;

  static constexpr size_t kMaxTensors = 1024;
  std::vector<TensorMeta> tensors_;
  size_t num_tensors_ = 0;
};

}  // namespace cactus
