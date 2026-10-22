#include "AudioLayout.h"

#include <cstring>
#include <initializer_list>

#include "CactArchive.h"

namespace whistle {

namespace {

bool shapeIs(const cactus::CactArchive& a, uint32_t idx,
             std::initializer_list<uint32_t> dims) {
  const cactus::TensorMeta* m = a.tensor(idx);
  if (m == nullptr || m->ndim != dims.size()) return false;
  uint32_t k = 0;
  for (uint32_t d : dims) {
    if (m->shape[k++] != d) return false;
  }
  return true;
}

bool typeIs(const cactus::CactArchive& a, uint32_t idx, cactus::DType t) {
  const cactus::TensorMeta* m = a.tensor(idx);
  return m != nullptr && m->dtype == t;
}

float readF32(const cactus::CactArchive& a, uint32_t idx, size_t i) {
  const cactus::TensorMeta* m = a.tensor(idx);
  if (m == nullptr) return 0.0f;
  float v = 0.0f;
  std::memcpy(&v, a.base() + m->offset + i * sizeof(float), sizeof(float));
  return v;
}

}  // namespace

bool findAudioLayout(const cactus::CactArchive& archive, AudioLayout* out) {
  if (out == nullptr || !archive.valid()) return false;
  *out = AudioLayout{};

  const uint32_t n = static_cast<uint32_t>(archive.tensorCount());
  if (n < 32) return false;

  // The audio region's first tensor is the 13-float manifest, magic 3246.
  uint32_t manifest = 0;
  bool found = false;
  for (uint32_t i = 0; i < n; ++i) {
    const cactus::TensorMeta* m = archive.tensor(i);
    if (m == nullptr || m->dtype != cactus::DType::FP32 || m->ndim != 1 ||
        m->shape[0] != 13) {
      continue;
    }
    if (readF32(archive, i, 0) == 3246.0f) {
      manifest = i;
      found = true;
      break;
    }
  }
  if (!found) return false;

  const uint32_t magic = static_cast<uint32_t>(readF32(archive, manifest, 0));
  const uint32_t version = static_cast<uint32_t>(readF32(archive, manifest, 1));
  const uint32_t enc_layers = static_cast<uint32_t>(readF32(archive, manifest, 3));
  const uint32_t enc_lanes = static_cast<uint32_t>(readF32(archive, manifest, 4));
  const uint32_t conv_kernel = static_cast<uint32_t>(readF32(archive, manifest, 5));
  const uint32_t stem = static_cast<uint32_t>(readF32(archive, manifest, 6));
  const uint32_t stem_stages = static_cast<uint32_t>(readF32(archive, manifest, 7));
  const uint32_t mels = static_cast<uint32_t>(readF32(archive, manifest, 8));
  const uint32_t rate = static_cast<uint32_t>(readF32(archive, manifest, 9));
  const uint32_t fft = static_cast<uint32_t>(readF32(archive, manifest, 10));
  const uint32_t win = static_cast<uint32_t>(readF32(archive, manifest, 11));
  const uint32_t hop = static_cast<uint32_t>(readF32(archive, manifest, 12));

  const cactus::CactHeader& h = archive.header();
  const uint32_t d = h.width();
  const uint32_t qk = h.qkHeadDim();
  const uint32_t vd = h.vHeadDim();
  const uint32_t kv = h.numKvHeads();
  const uint32_t dec_layers = h.numLayers();
  const uint32_t dec_lanes = h.mhcLanes();
  if (magic != 3246u || version != 1 || d == 0 || qk == 0 || vd == 0 ||
      kv == 0 || dec_layers == 0 || dec_lanes == 0 || enc_layers == 0 ||
      enc_lanes == 0 || conv_kernel == 0 || stem == 0 || stem_stages != 3 ||
      mels == 0 || (mels % 8) != 0) {
    return false;
  }
  const uint32_t heads = d / vd;
  if (heads == 0) return false;

  AudioLayout l;
  l.d_model = d;
  l.enc_layers = enc_layers;
  l.enc_lanes = enc_lanes;
  l.dec_layers = dec_layers;
  l.dec_lanes = dec_lanes;
  l.conv_kernel = conv_kernel;
  l.stem = stem;
  l.stem_stages = stem_stages;
  l.mels = mels;
  l.rate = rate;
  l.fft = fft;
  l.win = win;
  l.hop = hop;

  l.manifest = manifest;
  l.tokenizer = n - 1;
  l.mel = n - 2;
  l.stem_out = n - 3;
  l.stem_pw2 = n - 4;
  l.stem_dw2 = n - 5;
  l.stem_pw1 = n - 6;
  l.stem_dw1 = n - 7;
  l.stem_w = n - 8;
  l.enc_final_norm = n - 9;
  l.enc_mhc_base = n - 18;

  const uint64_t enc_span =
      static_cast<uint64_t>(AudioLayout::kEncPerLayer) * enc_layers;
  const uint64_t cross_span =
      static_cast<uint64_t>(AudioLayout::kCrossPerLayer) * dec_layers;
  if (static_cast<uint64_t>(l.enc_mhc_base) < enc_span ||
      static_cast<uint64_t>(l.enc_mhc_base) - enc_span <= manifest) {
    return false;
  }
  l.enc_base = static_cast<uint32_t>(static_cast<uint64_t>(l.enc_mhc_base) - enc_span);
  l.pe_gate = l.enc_base - 1;
  if (static_cast<uint64_t>(l.pe_gate) < cross_span) return false;
  l.cross_base = static_cast<uint32_t>(static_cast<uint64_t>(l.pe_gate) - cross_span);
  if (l.cross_base != manifest + 1) return false;
  if (l.enc_base != l.pe_gate + 1) return false;
  if (l.enc_mhc_base != l.enc_base + AudioLayout::kEncPerLayer * enc_layers)
    return false;

  // ---- decoder cross-attention ----
  for (uint32_t i = 0; i < dec_layers; ++i) {
    const uint32_t b = l.crossIndex(i, 0);
    if (!shapeIs(archive, b + 0, {d})) return false;
    if (!shapeIs(archive, b + 1, {heads * qk, d})) return false;
    if (!shapeIs(archive, b + 2, {heads * qk, d})) return false;
    if (!shapeIs(archive, b + 3, {heads * vd, d})) return false;
    if (!shapeIs(archive, b + 4, {d, d})) return false;
    if (!shapeIs(archive, b + 5, {d, d})) return false;
    if (!shapeIs(archive, b + 6, {qk})) return false;
    if (!shapeIs(archive, b + 7, {qk})) return false;
    if (!shapeIs(archive, b + 8, {d})) return false;
    if (!shapeIs(archive, b + 9, {1})) return false;
  }
  if (!typeIs(archive, l.pe_gate, cactus::DType::FP16) ||
      !shapeIs(archive, l.pe_gate, {1})) {
    return false;
  }

  // ---- encoder layers ----
  uint32_t nn = 1;
  while (nn < d) nn <<= 1;
  uint32_t ba = 1;
  while ((ba << 1) * (ba << 1) <= nn) ba <<= 1;
  const uint32_t bb = nn / ba;
  for (uint32_t i = 0; i < enc_layers; ++i) {
    const uint32_t b = l.encIndex(i, 0);
    for (uint32_t r = 0; r < 6; ++r)
      if (!shapeIs(archive, b + r, {d})) return false;
    if (!shapeIs(archive, b + 6, {heads * qk, d})) return false;
    if (!shapeIs(archive, b + 7, {kv * qk, d})) return false;
    if (!shapeIs(archive, b + 8, {kv * vd, d})) return false;
    if (!shapeIs(archive, b + 9, {d, d})) return false;
    if (!shapeIs(archive, b + 10, {d, d})) return false;
    if (!shapeIs(archive, b + 11, {qk})) return false;
    if (!shapeIs(archive, b + 12, {qk})) return false;
    if (!shapeIs(archive, b + 13, {1})) return false;
    if (!shapeIs(archive, b + 14, {2 * d, d})) return false;
    if (!shapeIs(archive, b + 15, {d, d})) return false;
    if (!shapeIs(archive, b + 16, {conv_kernel, d})) return false;
    for (uint32_t mlp = 0; mlp < 2; ++mlp) {
      const uint32_t m = b + 17 + mlp * 13;
      for (uint32_t r = 0; r < 5; ++r)
        if (!shapeIs(archive, m + r, {d})) return false;
      if (!shapeIs(archive, m + 5, {ba, ba})) return false;
      if (!shapeIs(archive, m + 6, {bb, bb})) return false;
      if (!shapeIs(archive, m + 7, {ba, ba})) return false;
      if (!shapeIs(archive, m + 8, {bb, bb})) return false;
      if (!shapeIs(archive, m + 9, {ba, ba})) return false;
      if (!shapeIs(archive, m + 10, {bb, bb})) return false;
      if (!shapeIs(archive, m + 11, {d, 8})) return false;
      if (!shapeIs(archive, m + 12, {8, nn})) return false;
    }
  }

  // ---- encoder mHC ----
  if (!shapeIs(archive, l.enc_mhc_base + 0, {enc_layers}) ||
      !shapeIs(archive, l.enc_mhc_base + 1, {enc_layers}) ||
      !shapeIs(archive, l.enc_mhc_base + 2, {enc_layers}) ||
      !shapeIs(archive, l.enc_mhc_base + 3, {enc_layers, enc_lanes}) ||
      !shapeIs(archive, l.enc_mhc_base + 4, {enc_layers, enc_lanes}) ||
      !shapeIs(archive, l.enc_mhc_base + 5, {enc_layers, enc_lanes, enc_lanes}) ||
      !shapeIs(archive, l.enc_mhc_base + 6,
               {enc_layers * enc_lanes, enc_lanes * d}) ||
      !shapeIs(archive, l.enc_mhc_base + 7,
               {enc_layers * enc_lanes, enc_lanes * d}) ||
      !shapeIs(archive, l.enc_mhc_base + 8,
               {enc_layers * enc_lanes * enc_lanes, enc_lanes * d}) ||
      !shapeIs(archive, l.enc_mhc_base + 9, {d})) {
    return false;
  }

  // ---- stem ----
  const uint32_t stem_out_in = stem * (mels / 8);
  if (!shapeIs(archive, l.stem_w, {conv_kernel, stem}) ||
      !shapeIs(archive, l.stem_dw1, {conv_kernel, stem}) ||
      !shapeIs(archive, l.stem_dw2, {conv_kernel, stem}) ||
      !shapeIs(archive, l.stem_pw1, {stem, stem}) ||
      !shapeIs(archive, l.stem_pw2, {stem, stem}) ||
      !shapeIs(archive, l.stem_out, {d, stem_out_in})) {
    return false;
  }
  if (!typeIs(archive, l.stem_w, cactus::DType::FP16) ||
      !typeIs(archive, l.stem_dw1, cactus::DType::FP16) ||
      !typeIs(archive, l.stem_dw2, cactus::DType::FP16) ||
      !typeIs(archive, l.stem_pw1, cactus::DType::CQ) ||
      !typeIs(archive, l.stem_pw2, cactus::DType::CQ) ||
      !typeIs(archive, l.stem_out, cactus::DType::CQ)) {
    return false;
  }

  // ---- mel + tokenizer ----
  if (!typeIs(archive, l.mel, cactus::DType::FP32) ||
      !shapeIs(archive, l.mel, {fft / 2 + 1, mels}) ||
      !typeIs(archive, l.tokenizer, cactus::DType::RAW)) {
    return false;
  }

  l.present = true;
  *out = l;
  return true;
}

}  // namespace whistle

