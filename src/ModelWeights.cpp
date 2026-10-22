#include "ModelWeights.h"

#include "kernels.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace whistle {

// int8 CQ GEMV path (byte-LUT decode + VMULAS.S8). Set 0 to fall back to the
// int16 levels/VMULAS.S16 path for an A/B or if a transcript regresses.
#ifndef WHISTLE_CQ_INT8
#define WHISTLE_CQ_INT8 1
#endif

// 1 = fused M == 1 GEMV path (one dot per weight row instead of one per group).
// MEASURED SLOWER: 429 vs 224 ms per decode step, depth 4 9877 -> 11738 ms. The
// 4x reduction in dot calls does not pay for the per-row int8 buffer round-trip,
// so the decoder's cost is per-row work, not per-call overhead - which rules out
// the hypothesis this was written for. Kept default-off for reference.
#ifndef WHISTLE_CQ_FUSE_M1
#define WHISTLE_CQ_FUSE_M1 0
#endif

ModelWeights::RowsHook ModelWeights::rowsHook_ = nullptr;
size_t ModelWeights::rowsStaging_ = 0;

namespace {

constexpr uint32_t kGroup = 128;

const char* kLayerSuffixOrder[27] = {
    "norm_in", "q_proj", "k_proj", "v_proj",
    "q_taps", "k_taps", "v_taps",
    "q_norm", "k_norm", "gate_proj", "out_proj",
    "post_norm", "attn_gate", "pre_hada",
    "d1", "d2", "b2", "d3", "d4",
    "w1a", "w1b", "w2a", "w2b", "w3a", "w3b",
    "cond_v", "cond_u",
};

inline uint16_t rd_u16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

// In-place unnormalized Walsh-Hadamard transform (n a power of two). This is
// H_raw @ x in butterfly order, where H_raw is the same Sylvester matrix the
// normalized Hadamard in prepare() is built from. Matches the naive H_raw @ x
// sum to float associativity.
void fwhtRaw(float* x, size_t n) {
  for (size_t h = 1; h < n; h <<= 1) {
    for (size_t i = 0; i < n; i += (h << 1)) {
      for (size_t j = i; j < i + h; ++j) {
        const float a = x[j];
        const float b = x[j + h];
        x[j] = a + b;
        x[j + h] = a - b;
      }
    }
  }
}

// PIE VLD.128.IP requires 16-byte-aligned sources. Bump an int16 buffer pointer
// to the next 16-byte boundary; the caller sizes its backing vector with 8 extra
// int16 (16 bytes) of headroom so the aligned window still fits.
inline int16_t* align16(int16_t* p) {
  return reinterpret_cast<int16_t*>(
      (reinterpret_cast<uintptr_t>(p) + 15u) & ~uintptr_t(15u));
}

bool matchLayerSuffix(const char* name, int* layer, int* suffix_off) {
  if (name[0] != 'l') return false;
  int i = 1;
  if (name[i] < '0' || name[i] > '9') return false;
  int l = 0;
  while (name[i] >= '0' && name[i] <= '9') {
    l = l * 10 + (name[i] - '0');
    ++i;
  }
  if (name[i] != '.') return false;
  const char* suffix = name + i + 1;
  for (int s = 0; s < 27; ++s) {
    if (std::strcmp(suffix, kLayerSuffixOrder[s]) == 0) {
      *layer = l;
      *suffix_off = s;
      return true;
    }
  }
  return false;
}

// ---- CQ GEMV row worker: one core, or half the rows on each core ----
// Rows are independent, so the platform hook may split [0, out_count). For a
// given (mm, row) the group sum still accumulates in ascending g, so any split
// is bit-identical to the single-core order.
struct CqRowCtx {
  const uint8_t* blob;
  const int16_t* cbq;
  const int16_t* lut2;
  const int32_t* lut4;
  const int16_t* fwAll;
  float* y;
  size_t idx_bytes;
  size_t row_pack;
  size_t group_pack;
  size_t M;
  uint32_t out_begin;
  uint32_t out_count;
  uint32_t groups;
  uint32_t group;
  uint32_t g;
  uint32_t bits;
  uint32_t mask;
  float kQ15;
};

void cqRowsImpl(void* vp, uint32_t r0, uint32_t r1, float* out,
                size_t outStride) {
  const CqRowCtx& c = *static_cast<const CqRowCtx*>(vp);
  int16_t wbStack[256 + 8];
  int16_t* wb = align16(wbStack);
  for (uint32_t r = r0; r < r1; ++r) {
    const size_t row = static_cast<size_t>(c.out_begin + r);
    const uint8_t* rbase = c.blob + row * c.row_pack;
    const float norm = fp16ToF32(
        rd_u16(c.blob + c.idx_bytes + row * c.groups * 2 + c.g * 2));
    const uint8_t* gidx = rbase + static_cast<size_t>(c.g) * c.group_pack;
    if (c.bits == 2) {
      const uint32_t nbytes = c.group >> 2;
      for (uint32_t b = 0; b < nbytes; ++b) {
        const int16_t* e = c.lut2 + 4 * gidx[b];
        int16_t* wp = wb + 4 * b;
        wp[0] = e[0];
        wp[1] = e[1];
        wp[2] = e[2];
        wp[3] = e[3];
      }
    } else if (c.bits == 4) {
      for (uint32_t k = 0; k < c.group; k += 2) {
        const int32_t pair = c.lut4[gidx[k >> 1]];
        wb[k] = static_cast<int16_t>(pair);
        wb[k + 1] = static_cast<int16_t>(pair >> 16);
      }
    } else {
      for (uint32_t chunk = 0; chunk < (c.group >> 3); ++chunk) {
        const uint8_t* cbytes = gidx + static_cast<size_t>(chunk) * c.bits;
        uint32_t word = 0;
        for (uint32_t b = 0; b < c.bits; ++b)
          word |= static_cast<uint32_t>(cbytes[b]) << (8 * b);
        for (uint32_t which = 0; which < 8; ++which)
          wb[chunk * 8 + which] = c.cbq[(word >> (which * c.bits)) & c.mask];
      }
    }
    for (size_t mm = 0; mm < c.M; ++mm) {
      const int16_t* fg = c.fwAll + mm * c.group;
      const int64_t s = kernels::dot_s16(wb, fg, c.group);
      out[mm * outStride + (r - r0)] +=
          norm * static_cast<float>(s) * c.kQ15;
    }
  }
}

// ---- int8 row worker: 2/4-bit groups decode to int8 levels ----
// One 256-entry LUT turns a packed byte into four (2-bit) or two (4-bit) int8
// levels, so a 128-wide group decodes with 32/64 stores instead of 128 LUT
// reads, and the dot runs as VMULAS.S8 (16 MACs per instruction) rather than
// VMULAS.S16. Accumulation order is unchanged: one dot per group, ascending g.
struct Cq8RowCtx {
  const uint8_t* blob;
  const int8_t* xq;  // M x group int8 activations for this group
  const float* sx;   // M activation group scales
  const uint32_t* tab2;
  const uint16_t* tab4;
  float* y;
  int8_t* wb[2];  // decoded group, one buffer per core
  size_t idx_bytes;
  size_t row_pack;
  size_t group_pack;
  size_t M;
  uint32_t out_begin;
  uint32_t out_count;
  uint32_t groups;
  uint32_t group;
  uint32_t g;
  uint32_t bits;
  float inv_sw;
};

void cq8RowsImpl(void* vp, uint32_t r0, uint32_t r1, float* out,
                 size_t outStride) {
  const Cq8RowCtx& c = *static_cast<const Cq8RowCtx*>(vp);
  int8_t* wq = c.wb[r0 == 0 ? 0 : 1];
  for (uint32_t r = r0; r < r1; ++r) {
    const size_t row = static_cast<size_t>(c.out_begin + r);
    const uint8_t* rbase = c.blob + row * c.row_pack;
    const float norm =
        fp16ToF32(rd_u16(c.blob + c.idx_bytes + row * c.groups * 2 + c.g * 2));
    const uint8_t* gidx = rbase + static_cast<size_t>(c.g) * c.group_pack;
    if (c.bits == 2) {
      const uint32_t nbytes = c.group >> 2;
      for (uint32_t b = 0; b < nbytes; ++b)
        reinterpret_cast<uint32_t*>(wq + 4 * b)[0] = c.tab2[gidx[b]];
    } else {
      const uint32_t nbytes = c.group >> 1;
      for (uint32_t b = 0; b < nbytes; ++b)
        reinterpret_cast<uint16_t*>(wq + 2 * b)[0] = c.tab4[gidx[b]];
    }
    for (size_t mm = 0; mm < c.M; ++mm) {
      const int32_t d = kernels::dot_s8(wq, c.xq + mm * c.group, c.group);
      out[mm * outStride + (r - r0)] +=
          norm * c.inv_sw * c.sx[mm] * static_cast<float>(d);
    }
  }
}

// Round half away from zero without libm (trunc.s on the FPU).
inline int8_t q8i(float v) {
  return static_cast<int8_t>(static_cast<int32_t>(v + (v >= 0.0f ? 0.5f : -0.5f)));
}

}  // namespace

bool ModelWeights::prepare(const cactus::CactArchive& archive) {
  archive_ = &archive;
  const cactus::CactHeader& h = archive.header();

  geom_.vocab = h.vocab();
  geom_.out_vocab = h.outVocab();
  geom_.d_model = h.width();
  geom_.num_heads = h.numHeads();
  geom_.num_kv_heads = h.numKvHeads();
  geom_.num_layers = h.numLayers();
  geom_.qk_head_dim = h.qkHeadDim();
  geom_.v_head_dim = h.vHeadDim();
  geom_.hada_n = h.hadaN();
  geom_.mhc_lanes = h.mhcLanes();
  geom_.qkv_conv_taps = h.qkvConvTaps();
  geom_.engram_slots = h.engramSlots();
  geom_.engram_sub_dim = h.engramSubDim();
  geom_.num_engram_tables = h.numEngramTables();
  geom_.engram_conv_taps = h.engramConvTaps();
  geom_.engram_conv_dilation = h.engramConvDilation();
  geom_.num_engram_orders = h.numEngramOrders();
  for (int i = 0; i < 4; ++i) geom_.engram_orders[i] = h.engramOrder(i);
  geom_.num_engram_sites = h.numEngramSites();
  for (int i = 0; i < 16; ++i) geom_.engram_sites[i] = h.engramSite(i);
  geom_.rope_theta = h.ropeTheta();
  geom_.tensor_count = archive.tensorCount();

  codebooks_ = archive.codebooks();
  for (int i = 0; i < 28; ++i)
    cbQ15_[i] = static_cast<int16_t>(std::lround(codebooks_[i] * 32768.0f));
  for (int b = 0; b < 256; ++b) {
    const int32_t lo = static_cast<int32_t>(cbQ15_[12 + (b & 15)]);
    const int32_t hi = static_cast<int32_t>(cbQ15_[12 + (b >> 4)]);
    lut4_[b] = (hi << 16) | (lo & 0xFFFF);
    for (uint32_t j = 0; j < 4; ++j)
      lut2_[4 * b + j] = cbQ15_[(b >> (2 * j)) & 3];
  }

  // int8 mirrors of the same codebooks. The largest |level| of each bit class
  // maps to 127, so a level scaled back by cb8Inv_ is the codebook value.
  {
    const uint32_t base[3] = {0, 4, 12};
    const uint32_t cnt[3] = {4, 8, 16};
    for (int cl = 0; cl < 3; ++cl) {
      float mx = 0.0f;
      for (uint32_t i = 0; i < cnt[cl]; ++i) {
        const float a = std::fabs(codebooks_[base[cl] + i]);
        if (a > mx) mx = a;
      }
      const float sc = mx > 0.0f ? 127.0f / mx : 0.0f;
      for (uint32_t i = 0; i < cnt[cl]; ++i) {
        int32_t v = static_cast<int32_t>(
            std::lround(codebooks_[base[cl] + i] * sc));
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        cb8_[base[cl] + i] = static_cast<int8_t>(v);
      }
      cb8Inv_[cl] = mx > 0.0f ? mx / 127.0f : 1.0f;
    }
  }
  for (int b = 0; b < 256; ++b) {
    const uint32_t u = static_cast<uint32_t>(b);
    const uint32_t l0 = static_cast<uint8_t>(cb8_[(u >> 0) & 3]);
    const uint32_t l1 = static_cast<uint8_t>(cb8_[(u >> 2) & 3]);
    const uint32_t l2 = static_cast<uint8_t>(cb8_[(u >> 4) & 3]);
    const uint32_t l3 = static_cast<uint8_t>(cb8_[(u >> 6) & 3]);
    lut2_8_[b] = l0 | (l1 << 8) | (l2 << 16) | (l3 << 24);
    lut4_8_[b] = static_cast<uint16_t>(
        static_cast<uint8_t>(cb8_[12 + (u & 15)]) |
        (static_cast<uint32_t>(static_cast<uint8_t>(cb8_[12 + (u >> 4)])) << 8));
  }

  // Normalized Walsh-Hadamard H_raw / sqrt(group), row-major stride kGroup.
  const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(kGroup));
  std::vector<float> H(kGroup * kGroup, 0.0f);
  H[0] = 1.0f;
  uint32_t n = 1;
  while (n < kGroup) {
    for (uint32_t i = 0; i < n; ++i) {
      for (uint32_t j = 0; j < n; ++j) {
        const float v = H[i * kGroup + j];
        H[i * kGroup + j + n] = v;
        H[(i + n) * kGroup + j] = v;
        H[(i + n) * kGroup + j + n] = -v;
      }
    }
    n <<= 1;
  }
  hadamard_.resize(kGroup * kGroup);
  for (uint32_t i = 0; i < kGroup * kGroup; ++i) {
    hadamard_[i] = H[i] * inv_sqrt;
  }
  return true;
}

int ModelWeights::indexOf(const char* name) const {
  if (std::strcmp(name, "embedding") == 0) return 0;

  int layer, suffix;
  if (matchLayerSuffix(name, &layer, &suffix)) {
    if (layer < 0 || layer >= static_cast<int>(geom_.num_layers)) return -1;
    const int per = 27;
    return 1 + layer * per + suffix;
  }

  const int mhc_base = 1 + static_cast<int>(geom_.num_layers) * 27;
  const int hada_p1 = mhc_base + 9;
  const int engram_base = hada_p1 + 2;
  const int final_idx =
      engram_base + static_cast<int>(geom_.num_engram_sites) * 4;

  static const struct Mhc { const char* n; int rel; } mhc_tbl[] = {
      {"mhc.a_pre", 0}, {"mhc.a_post", 1}, {"mhc.a_res", 2},
      {"mhc.b_pre", 3}, {"mhc.b_post", 4}, {"mhc.b_res", 5},
      {"mhc.phi_pre", 6}, {"mhc.phi_post", 7}, {"mhc.phi_res", 8},
  };
  for (const auto& e : mhc_tbl) {
    if (std::strcmp(name, e.n) == 0) return mhc_base + e.rel;
  }
  if (std::strcmp(name, "hada_p1") == 0) return hada_p1;
  if (std::strcmp(name, "hada_p2") == 0) return hada_p1 + 1;
  if (std::strcmp(name, "final_norm") == 0) return final_idx;

  if (std::strncmp(name, "engram", 6) == 0 && name[6] >= '0' && name[6] <= '9') {
    const int s = name[6] - '0';
    const char* p = name + 7;
    if (*p != '.') return -1;
    ++p;
    int rel = -1;
    if (std::strcmp(p, "tables") == 0) rel = 0;
    else if (std::strcmp(p, "key_proj") == 0) rel = 1;
    else if (std::strcmp(p, "value_proj") == 0) rel = 2;
    else if (std::strcmp(p, "taps") == 0) rel = 3;
    if (rel < 0 || s >= static_cast<int>(geom_.num_engram_sites)) return -1;
    return engram_base + s * 4 + rel;
  }
  return -1;
}

const cactus::TensorMeta* ModelWeights::meta(const char* name) const {
  const int i = indexOf(name);
  if (i < 0) return nullptr;
  return archive_->tensor(static_cast<size_t>(i));
}

const cactus::TensorMeta* ModelWeights::metaAt(size_t index) const {
  return (archive_ != nullptr) ? archive_->tensor(index) : nullptr;
}

bool ModelWeights::readTensor(const char* name, float* out, size_t elements) const {
  return readMeta(meta(name), out, elements);
}

bool ModelWeights::readIndex(size_t index, float* out, size_t elements) const {
  return readMeta(metaAt(index), out, elements);
}

bool ModelWeights::readMeta(const cactus::TensorMeta* m, float* out,
                            size_t elements) const {
  if (m == nullptr || archive_ == nullptr || out == nullptr) return false;

  {
    size_t logical = 1;
    for (uint8_t d = 0; d < m->ndim; ++d) logical *= m->shape[d];
    if (logical != elements) return false;
  }

  const uint8_t* blob = archive_->base() + m->offset;

  switch (m->dtype) {
    case cactus::DType::FP32: {
      for (size_t i = 0; i < elements; ++i) {
        uint32_t b;
        std::memcpy(&b, blob + i * 4, 4);
        // file is little-endian; host is little-endian -> direct copy
        std::memcpy(&out[i], &b, 4);
      }
      return true;
    }
    case cactus::DType::FP16: {
      for (size_t i = 0; i < elements; ++i) {
        out[i] = fp16ToF32(rd_u16(blob + i * 2));
      }
      return true;
    }
    case cactus::DType::CQ:
      return dequantCq(m, blob, out);
    default:
      return false;  // RAW not supported here
  }
}

bool ModelWeights::dequantCq(const cactus::TensorMeta* m, const uint8_t* blob,
                             float* out) const {
  const uint32_t out_rows = m->shape[0];
  const uint32_t in = (m->ndim > 1) ? m->shape[1] : 1;
  for (uint32_t r = 0; r < out_rows; ++r) {
    dequantCqRowImpl(m, blob, r, out + static_cast<size_t>(r) * in);
  }
  return true;
}

void ModelWeights::dequantCqRowImpl(const cactus::TensorMeta* m,
                                    const uint8_t* blob, uint32_t r,
                                    float* out) const {
  const uint32_t in = (m->ndim > 1) ? m->shape[1] : 1;
  const uint32_t bits = m->bits;
  const uint32_t group = m->group ? m->group : kGroup;
  const uint32_t in_pad = ((in + group - 1) / group) * group;
  const uint32_t norm_count = in_pad / group;

  const float* cb;
  if (bits == 2) cb = codebooks_;              // cb2[4]
  else if (bits == 3) cb = codebooks_ + 4;     // cb3[8]
  else if (bits == 4) cb = codebooks_ + 12;    // cb4[16]
  else return;

  const uint8_t* row_idx = blob + static_cast<size_t>(r) * in_pad * bits / 8;

  // Norms for this row start after ALL index rows.
  const uint32_t out_rows = m->shape[0];
  const size_t idx_bytes = static_cast<size_t>(out_rows) * in_pad * bits / 8;
  const uint8_t* row_norm = blob + idx_bytes + static_cast<size_t>(r) * norm_count * 2;

  std::vector<float> rowbuf(in_pad);
  for (uint32_t g = 0; g < in_pad / group; ++g) {
    const float norm = fp16ToF32(rd_u16(row_norm + g * 2));
    float rot[kGroup];
    for (uint32_t col = 0; col < group; ++col) {
      const uint32_t chunk = col >> 3;
      const uint32_t which = col & 7;
      const uint8_t* cbytes = row_idx + g * group * bits / 8 + chunk * bits;
      uint32_t code = 0;
      for (uint32_t b = 0; b < bits; ++b) {
        code |= static_cast<uint32_t>(cbytes[b]) << (8 * b);
      }
      code = (code >> (which * bits)) & ((1u << bits) - 1);
      rot[col] = cb[code] * norm;
    }
    float* dst = rowbuf.data() + g * group;
    for (uint32_t row2 = 0; row2 < group; ++row2) {
      float acc = 0.0f;
      for (uint32_t col = 0; col < group; ++col) {
        acc += rot[col] * hadamard_[col * kGroup + row2];
      }
      dst[row2] = acc;
    }
  }
  std::memcpy(out, rowbuf.data(), in * sizeof(float));
}

bool ModelWeights::readCqRow(const char* name, uint32_t row, float* out) const {
  const cactus::TensorMeta* m = meta(name);
  if (m == nullptr || m->dtype != cactus::DType::CQ || out == nullptr) return false;
  if (row >= m->shape[0]) return false;
  const uint8_t* blob = archive_->base() + m->offset;
  dequantCqRowImpl(m, blob, row, out);
  return true;
}

bool ModelWeights::gemvCq(const char* name, uint32_t out_begin, uint32_t out_count,
                          const float* x, size_t M, float* y) const {
  return gemvCqMeta(meta(name), out_begin, out_count, x, M, y);
}

bool ModelWeights::gemvCqIndex(size_t index, uint32_t out_begin,
                               uint32_t out_count, const float* x, size_t M,
                               float* y) const {
  return gemvCqMeta(metaAt(index), out_begin, out_count, x, M, y);
}

bool ModelWeights::gemvCqMeta(const cactus::TensorMeta* m, uint32_t out_begin,
                              uint32_t out_count, const float* x, size_t M,
                              float* y) const {
  if (m == nullptr || m->dtype != cactus::DType::CQ) return false;
  const uint32_t N = m->shape[0];
  const uint32_t in = (m->ndim > 1) ? m->shape[1] : 1;
  const uint32_t bits = m->bits;
  const uint32_t group = m->group ? m->group : kGroup;
  const uint32_t in_pad = ((in + group - 1) / group) * group;
  const uint32_t groups = in_pad / group;
  if (out_begin + out_count > N || x == nullptr || y == nullptr || M == 0 ||
      out_count == 0)
    return false;

  const float* cb;
  uint32_t ncb;
  if (bits == 2) { cb = codebooks_; ncb = 4; }
  else if (bits == 3) { cb = codebooks_ + 4; ncb = 8; }
  else if (bits == 4) { cb = codebooks_ + 12; ncb = 16; }
  else return false;

  const size_t idx_bytes = static_cast<size_t>(N) * in_pad * bits / 8;
  const uint8_t* blob = archive_->base() + m->offset;
  const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(kGroup));

  std::vector<float> u(in_pad);
  std::vector<float> bucket(ncb);

  for (size_t mm = 0; mm < M; ++mm) {
    const float* xrow = x + mm * in;
    // activations -> unnormalized FWHT, once per group (shared across all rows)
    for (uint32_t g = 0; g < groups; ++g) {
      float* ug = u.data() + static_cast<size_t>(g) * group;
      const size_t src = static_cast<size_t>(g) * group;
      if (src + group <= in) {
        std::memcpy(ug, xrow + src, group * sizeof(float));
      } else {
        const size_t avail = (in > src) ? (in - src) : 0;
        std::memcpy(ug, xrow + src, avail * sizeof(float));
        std::memset(ug + avail, 0, (group - avail) * sizeof(float));
      }
      fwhtRaw(ug, group);
    }

    for (uint32_t o = 0; o < out_count; ++o) {
      const uint32_t row = out_begin + o;
      const uint8_t* ridx = blob + static_cast<size_t>(row) * in_pad * bits / 8;
      const uint8_t* rnorm = blob + idx_bytes + static_cast<size_t>(row) * groups * 2;
      float acc = 0.0f;
      for (uint32_t g = 0; g < groups; ++g) {
        const float norm = fp16ToF32(rd_u16(rnorm + g * 2));
        for (uint32_t c = 0; c < ncb; ++c) bucket[c] = 0.0f;
        const float* ug = u.data() + static_cast<size_t>(g) * group;
        const uint8_t* gidx = ridx + static_cast<size_t>(g) * group * bits / 8;
        for (uint32_t k = 0; k < group; ++k) {
          const uint32_t chunk = k >> 3;
          const uint32_t which = k & 7;
          const uint8_t* cbytes = gidx + static_cast<size_t>(chunk) * bits;
          uint32_t code = 0;
          for (uint32_t b = 0; b < bits; ++b)
            code |= static_cast<uint32_t>(cbytes[b]) << (8 * b);
          code = (code >> (which * bits)) & ((1u << bits) - 1);
          bucket[code] += ug[k];
        }
        float cb_dot = 0.0f;
        for (uint32_t c = 0; c < ncb; ++c) cb_dot += cb[c] * bucket[c];
        acc += norm * cb_dot;
      }
y[mm * out_count + o] = acc * inv_sqrt;
    }
  }
  return true;
}

bool ModelWeights::gemvCqInt8(const char* name, uint32_t out_begin,
                              uint32_t out_count, const int8_t* q,
                              const float* scale, size_t M, float* y) const {
  return gemvCqInt8Meta(meta(name), out_begin, out_count, q, scale, M, y);
}

bool ModelWeights::gemvCqInt8Index(size_t index, uint32_t out_begin,
                                   uint32_t out_count, const int8_t* q,
                                   const float* scale, size_t M, float* y) const {
  return gemvCqInt8Meta(metaAt(index), out_begin, out_count, q, scale, M, y);
}

bool ModelWeights::gemvCqInt8Meta(const cactus::TensorMeta* m,
                                  uint32_t out_begin, uint32_t out_count,
                                  const int8_t* q, const float* scale, size_t M,
                                  float* y) const {
  if (m == nullptr || m->dtype != cactus::DType::CQ) return false;
  const uint32_t N = m->shape[0];
  const uint32_t in = (m->ndim > 1) ? m->shape[1] : 1;
  const uint32_t bits = m->bits;
  const uint32_t group = m->group ? m->group : kGroup;
  const uint32_t in_pad = ((in + group - 1) / group) * group;
  const uint32_t groups = in_pad / group;
  if (out_begin + out_count > N || q == nullptr || scale == nullptr ||
      y == nullptr || M == 0 || out_count == 0)
    return false;

  const int16_t* cbq;
  if (bits == 2) cbq = cbQ15_;
  else if (bits == 3) cbq = cbQ15_ + 4;
  else if (bits == 4) cbq = cbQ15_ + 12;
  else return false;

  const size_t idx_bytes = static_cast<size_t>(N) * in_pad * bits / 8;
  const uint8_t* blob = archive_->base() + m->offset;
  const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(kGroup));
  const float kQ15 = 1.0f / 32768.0f;
  const auto fire = [&](const char* tag) {
    if (out_count >= 512 && progress) progress(progressUdata, tag);
  };
  const uint32_t mask = (1u << bits) - 1;
  const size_t row_pack = static_cast<size_t>(in_pad) * bits / 8;
  const size_t group_pack = static_cast<size_t>(group) * bits / 8;

  // Group OUTER. The FWHT working set is only M x group int16 (~10 KB for the
  // encoder), so it can live in internal SRAM; the previous M x in_pad buffer
  // (up to ~156 KB) fell back to PSRAM and made every butterfly a PSRAM
  // read-modify-write. Output rows are still unpacked once per group and the
  // per-group float accumulation order is unchanged.
  const size_t fw_need = M * group + 8;
  if (fwScratch_.size() < fw_need) fwScratch_.resize(fw_need);
  int16_t* fwAll = align16(fwScratch_.data());

  // int8 path: the 2/4-bit levels fit int8, so the weight group decodes with a
  // byte LUT and the dot runs at 16 MACs/instruction. Group 128 is assumed by
  // both the LUT decode and the 16-byte dot granularity.
  const bool i8path = WHISTLE_CQ_INT8 && (bits == 2 || bits == 4) && group == 128;
  int8_t* q8All = nullptr;
  float* sx8 = nullptr;
  if (i8path) {
    const size_t q_need_raw = M * group + 16;
    // Read once per (row, group, activation row) - i.e. hundreds of times per
    // output row. Requests <= 16 KB land in internal SRAM, which is uncached
    // here; staging the same pattern into a fresh PSRAM buffer was worth 13x on
    // kronApply (1608 -> 105 ms), so keep this above the threshold.
    const size_t q8_need = q_need_raw > 20 * 1024 ? q_need_raw : 20 * 1024;
    if (q8Scratch_.size() < q8_need) q8Scratch_.resize(q8_need);
    if (sxScratch_.size() < M) sxScratch_.resize(M);
    q8All = reinterpret_cast<int8_t*>(
        (reinterpret_cast<uintptr_t>(q8Scratch_.data()) + 15u) & ~uintptr_t(15u));
    sx8 = sxScratch_.data();
  }

  if (group > 256) return false;  // wb is a per-core stack buffer

  for (size_t i = 0; i < M * out_count; ++i) y[i] = 0.0f;

  // ---- M == 1 fast path ------------------------------------------------
  // The decoder GEMVs run at M == 1 and cost one dot call per (output row, group)
  // - 43,776 calls per decode step - each paying the asm prologue plus a
  // read-modify-write of y. Here the activation row is transformed once for all
  // groups and each weight row is decoded whole, so one dot_groups_s8 covers the
  // row. The per-output accumulation is still ascending g with the same product
  // order, so the result is bit-identical to the group-outer loop below.
  if (WHISTLE_CQ_FUSE_M1 && i8path && M == 1 && groups <= 32 && fusedOk) {
    const size_t q_need = static_cast<size_t>(in_pad) + 16;
    if (q8Scratch_.size() < q_need) q8Scratch_.resize(q_need);
    if (rowScratch_.size() < q_need) rowScratch_.resize(q_need);
    if (sxScratch_.size() < groups) sxScratch_.resize(groups);
    int8_t* xq = reinterpret_cast<int8_t*>(
        (reinterpret_cast<uintptr_t>(q8Scratch_.data()) + 15u) & ~uintptr_t(15u));
    int8_t* wq = reinterpret_cast<int8_t*>(
        (reinterpret_cast<uintptr_t>(rowScratch_.data()) + 15u) & ~uintptr_t(15u));
    float* sx = sxScratch_.data();

    for (uint32_t g = 0; g < groups; ++g) {
      const size_t src = static_cast<size_t>(g) * group;
      int16_t* fg = fwAll;
      if (src + group <= in) {
        for (uint32_t k = 0; k < group; ++k)
          fg[k] = static_cast<int16_t>(q[src + k]);
      } else {
        const size_t avail = (in > src) ? (in - src) : 0;
        for (size_t k = 0; k < avail; ++k)
          fg[k] = static_cast<int16_t>(q[src + k]);
        for (uint32_t k = static_cast<uint32_t>(avail); k < group; ++k) fg[k] = 0;
      }
      kernels::fwht_s16(fg, group);
      int32_t amax = 0;
      for (uint32_t k = 0; k < group; ++k) {
        const int32_t v = fg[k] < 0 ? -fg[k] : fg[k];
        if (v > amax) amax = v;
      }
      const float s = amax > 0 ? static_cast<float>(amax) / 127.0f : 1.0f;
      const float inv = 1.0f / s;
      int8_t* xd = xq + src;
      for (uint32_t k = 0; k < group; ++k)
        xd[k] = q8i(static_cast<float>(fg[k]) * inv);
      sx[g] = s;
    }
    fire("fwht");

    const float inv_sw = cb8Inv_[bits == 2 ? 0 : 2];
    const uint32_t tab_bytes = (bits == 2) ? (group >> 2) : (group >> 1);
    int64_t d64[32];
    for (uint32_t o = 0; o < out_count; ++o) {
      const size_t row = static_cast<size_t>(out_begin + o);
      const uint8_t* rbase = blob + row * row_pack;
      const uint8_t* rnorm = blob + idx_bytes + row * groups * 2;
      for (uint32_t g = 0; g < groups; ++g) {
        const uint8_t* gidx = rbase + static_cast<size_t>(g) * group_pack;
        int8_t* dst = wq + static_cast<size_t>(g) * group;
        if (bits == 2) {
          for (uint32_t b = 0; b < tab_bytes; ++b)
            reinterpret_cast<uint32_t*>(dst + 4 * b)[0] = lut2_8_[gidx[b]];
        } else {
          for (uint32_t b = 0; b < tab_bytes; ++b)
            reinterpret_cast<uint16_t*>(dst + 2 * b)[0] = lut4_8_[gidx[b]];
        }
      }
      kernels::dot_groups_s8(wq, xq, d64, groups);
      if (!fusedChecked) {
        // One-shot self-check against the scalar reference dots: if the PIE
        // kernel is wrong, fall back for good instead of corrupting logits.
        bool ok = true;
        for (uint32_t g = 0; g < groups && ok; ++g) {
          const int32_t ref = kernels::dot_s8(
              wq + static_cast<size_t>(g) * group,
              xq + static_cast<size_t>(g) * group, group);
          if (static_cast<int64_t>(ref) != d64[g]) ok = false;
        }
        if (!ok)
          for (uint32_t g = 0; g < groups; ++g)
            d64[g] = kernels::dot_s8(wq + static_cast<size_t>(g) * group,
                                     xq + static_cast<size_t>(g) * group,
                                     group);
        fusedChecked = true;
        fusedOk = ok;
      }
      float acc = 0.0f;
      for (uint32_t g = 0; g < groups; ++g)
        acc += fp16ToF32(rd_u16(rnorm + g * 2)) * inv_sw * sx[g] *
               static_cast<float>(d64[g]);
      y[o] += acc;
    }
    fire("buck");
    const float s0 = inv_sqrt * scale[0];
    for (uint32_t o = 0; o < out_count; ++o) y[o] *= s0;
    return true;
  }

  for (uint32_t g = 0; g < groups; ++g) {
    const size_t src = static_cast<size_t>(g) * group;
    for (size_t mm = 0; mm < M; ++mm) {
      const int8_t* qrow = q + mm * in;
      int16_t* fg = fwAll + mm * group;
      if (src + group <= in) {
        for (uint32_t k = 0; k < group; ++k)
          fg[k] = static_cast<int16_t>(qrow[src + k]);
      } else {
        const size_t avail = (in > src) ? (in - src) : 0;
        for (size_t k = 0; k < avail; ++k)
          fg[k] = static_cast<int16_t>(qrow[src + k]);
        for (uint32_t k = static_cast<uint32_t>(avail); k < group; ++k)
          fg[k] = 0;
      }
      kernels::fwht_s16(fg, group);
      if (i8path) {
        // One scale per (row, group) for the int8 re-quantization, mirroring the
        // float path's per-group FWHT normalization.
        int32_t amax = 0;
        for (uint32_t k = 0; k < group; ++k) {
          const int32_t v = fg[k] < 0 ? -fg[k] : fg[k];
          if (v > amax) amax = v;
        }
        const float s = amax > 0 ? static_cast<float>(amax) / 127.0f : 1.0f;
        const float inv = 1.0f / s;
        int8_t* xd = q8All + mm * group;
        for (uint32_t k = 0; k < group; ++k)
          xd[k] = q8i(static_cast<float>(fg[k]) * inv);
        sx8[mm] = s;
      }
    }
    fire("fwht");

    if (i8path) {
      Cq8RowCtx c8;
      c8.blob = blob;
      c8.xq = q8All;
      c8.sx = sx8;
      c8.tab2 = lut2_8_;
      c8.tab4 = lut4_8_;
      c8.y = y;
      c8.idx_bytes = idx_bytes;
      c8.row_pack = row_pack;
      c8.group_pack = group_pack;
      c8.M = M;
      c8.out_begin = out_begin;
      c8.out_count = out_count;
      c8.groups = groups;
      c8.group = group;
      c8.g = g;
      c8.bits = bits;
      c8.inv_sw = cb8Inv_[bits == 2 ? 0 : 2];
      for (int wbi = 0; wbi < 2; ++wbi) {
        if (wqRow_[wbi].size() < 20 * 1024)
          wqRow_[wbi].assign(20 * 1024, 0);
        c8.wb[wbi] = reinterpret_cast<int8_t*>(
            (reinterpret_cast<uintptr_t>(wqRow_[wbi].data()) + 15u) &
            ~uintptr_t(15u));
      }
      RowsSync sync;
      sync.ro = q8All;
      sync.roBytes = static_cast<size_t>(M) * group;
      sync.rw = y;
      sync.rwStride = out_count;
      sync.rwRows = M;
      const uint32_t wrows = out_count - out_count / 2;
      if (rowsHook_ != nullptr && out_count >= 2 &&
          static_cast<size_t>(M) * wrows <= rowsStaging_) {
        rowsHook_(&cq8RowsImpl, &c8, out_count, sync);
      } else {
        cq8RowsImpl(&c8, 0, out_count, y, out_count);
      }
      fire("buck");
      continue;
    }

    CqRowCtx ctx;
    ctx.blob = blob;
    ctx.cbq = cbq;
    ctx.lut2 = lut2_;
    ctx.lut4 = lut4_;
    ctx.fwAll = fwAll;
    ctx.y = y;
    ctx.idx_bytes = idx_bytes;
    ctx.row_pack = row_pack;
    ctx.group_pack = group_pack;
    ctx.M = M;
    ctx.out_begin = out_begin;
    ctx.out_count = out_count;
    ctx.groups = groups;
    ctx.group = group;
    ctx.g = g;
    ctx.bits = bits;
    ctx.mask = mask;
    ctx.kQ15 = kQ15;
    // Rows within this group are independent; the platform may run half on the
    // second core. The per-row group sum still accumulates in ascending g.
    RowsSync sync;
    sync.ro = fwAll;
    sync.roBytes = static_cast<size_t>(M) * group * sizeof(int16_t);
    sync.rw = y;
    sync.rwStride = out_count;
    sync.rwRows = M;
    const uint32_t wrows = out_count - out_count / 2;
    if (rowsHook_ != nullptr && out_count >= 2 &&
        static_cast<size_t>(M) * wrows <= rowsStaging_) {
      rowsHook_(&cqRowsImpl, &ctx, out_count, sync);
    } else {
      cqRowsImpl(&ctx, 0, out_count, y, out_count);
    }
    fire("buck");
  }

  for (size_t mm = 0; mm < M; ++mm) {
    const float s = inv_sqrt * scale[mm];
    for (uint32_t o = 0; o < out_count; ++o) y[mm * out_count + o] *= s;
  }
  return true;
}

}  // namespace whistle