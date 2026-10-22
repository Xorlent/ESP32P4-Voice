#include "Tokenizer.h"

#include <algorithm>
#include <cstring>

#include "CactArchive.h"

namespace whistle {

namespace {

constexpr const char* kMetaSpace = "\xE2\x96\x81";  // U+2581
enum { kNormal = 0, kUnknown = 1, kControl = 2, kUserDefined = 3, kByte = 4 };

uint32_t rdU32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
uint16_t rdU16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

int hexVal(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return 0;
}

size_t utf8CharLen(uint8_t b) {
  if ((b & 0x80) == 0) return 1;
  if ((b & 0xE0) == 0xC0) return 2;
  if ((b & 0xF0) == 0xE0) return 3;
  if ((b & 0xF8) == 0xF0) return 4;
  return 1;
}

std::vector<std::string> utf8CodePoints(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    size_t len = utf8CharLen(static_cast<uint8_t>(s[i]));
    if (i + len > s.size()) len = s.size() - i;
    out.push_back(s.substr(i, len));
    i += len;
  }
  return out;
}

}  // namespace

bool Tokenizer::prepare(const cactus::CactArchive& archive) {
  const size_t n = archive.tensorCount();
  if (n == 0) return false;
  const cactus::TensorMeta* m = archive.tensor(n - 1);
  const uint8_t* blob = archive.base() + m->offset;
  const size_t size = m->nbytes;
  if (size < 24) return false;

  const size_t np = rdU32(blob);
  pad_id_ = static_cast<int32_t>(rdU32(blob + 4));
  eos_id_ = static_cast<int32_t>(rdU32(blob + 8));
  bos_id_ = static_cast<int32_t>(rdU32(blob + 12));
  unk_id_ = static_cast<int32_t>(rdU32(blob + 16));
  add_dummy_ = blob[20] != 0;
  byte_fallback_ = blob[21] != 0;

  pieces_.resize(np);
  scores_.resize(np);
  types_.resize(np);

  size_t off = 24;
  for (size_t i = 0; i < np; ++i) {
    if (off + 7 > size) return false;
    float score;
    std::memcpy(&score, blob + off, 4);  // little-endian f32
    const uint8_t t = blob[off + 4];
    const uint16_t len = rdU16(blob + off + 5);
    off += 7;
    if (off + len > size) return false;
    pieces_[i].assign(reinterpret_cast<const char*>(blob + off), len);
    scores_[i] = score;
    types_[i] = t;
    off += len;
  }

  for (int i = 0; i < 256; ++i) byte_id_[i] = -1;
  p2id_.clear();
  markers_.clear();
  for (size_t i = 0; i < np; ++i) {
    p2id_[pieces_[i]] = static_cast<int32_t>(i);
    if (types_[i] == kByte) {
      const std::string& p = pieces_[i];
      if (p.size() >= 5) {
        const int b = hexVal(p[3]) * 16 + hexVal(p[4]);
        byte_id_[b] = static_cast<int32_t>(i);
      }
    } else if (types_[i] == kUserDefined) {
      markers_.push_back(pieces_[i]);
    }
  }
  std::sort(markers_.begin(), markers_.end(),
            [](const std::string& a, const std::string& b) {
              return utf8CodePoints(a).size() > utf8CodePoints(b).size();
            });
  return true;
}

std::string Tokenizer::decode(const int32_t* ids, size_t n) const {
  std::string out;
  out.reserve(n * 4);
  for (size_t k = 0; k < n; ++k) {
    const int32_t i = ids[k];
    if (i < 0 || i >= static_cast<int32_t>(pieces_.size())) continue;
    const uint8_t t = types_[i];
    if (t == kByte) {
      const std::string& p = pieces_[i];
      if (p.size() >= 5)
        out.push_back(static_cast<char>(hexVal(p[3]) * 16 + hexVal(p[4])));
    } else if (t == kControl || t == kUnknown) {
      continue;
    } else {
      out += pieces_[i];
    }
  }
  // Replace the U+2581 meta-space byte sequence (E2 96 81) with a single space.
  std::string res;
  res.reserve(out.size());
  for (size_t i = 0; i < out.size();) {
    if (i + 2 < out.size() && static_cast<uint8_t>(out[i]) == 0xE2 &&
        static_cast<uint8_t>(out[i + 1]) == 0x96 &&
        static_cast<uint8_t>(out[i + 2]) == 0x81) {
      res.push_back(' ');
      i += 3;
    } else {
      res.push_back(out[i]);
      ++i;
    }
  }
  if (add_dummy_ && !res.empty() && res[0] == ' ') res.erase(0, 1);
  return res;
}

std::vector<int32_t> Tokenizer::bpe(const std::vector<std::string>& syms) const {
  std::vector<std::string> s = syms;
  while (s.size() > 1) {
    int best_j = -1;
    float best_score = -1e30f;
    for (size_t j = 0; j + 1 < s.size(); ++j) {
      const std::string pair = s[j] + s[j + 1];
      const auto it = p2id_.find(pair);
      if (it != p2id_.end()) {
        const float sc = scores_[it->second];
        if (sc > best_score) {
          best_score = sc;
          best_j = static_cast<int>(j);
        }
      }
    }
    if (best_j < 0) break;
    s[best_j] = s[best_j] + s[best_j + 1];
    s.erase(s.begin() + best_j + 1);
  }

  std::vector<int32_t> ids;
  for (const auto& sym : s) {
    const auto it = p2id_.find(sym);
    if (it != p2id_.end()) {
      ids.push_back(it->second);
    } else if (byte_fallback_) {
      for (const unsigned char b : sym) ids.push_back(byte_id_[b]);
    } else {
      ids.push_back(unk_id_);
    }
  }
  return ids;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
  std::vector<int32_t> ids;
  if (text.empty()) return ids;

  std::string esc;
  esc.reserve(text.size());
  for (const char c : text) {
    if (c == ' ') esc += kMetaSpace;
    else esc.push_back(c);
  }
  if (add_dummy_) esc = std::string(kMetaSpace) + esc;

  const std::vector<std::string> cps = utf8CodePoints(esc);
  std::vector<std::string> buf;
  const size_t n = cps.size();
  size_t i = 0;
  auto flush = [&]() {
    std::vector<int32_t> part = bpe(buf);
    ids.insert(ids.end(), part.begin(), part.end());
    buf.clear();
  };
  while (i < n) {
    bool matched = false;
    for (const auto& mk : markers_) {
      const std::vector<std::string> mkcps = utf8CodePoints(mk);
      if (i + mkcps.size() > n) continue;
      bool ok = true;
      for (size_t k = 0; k < mkcps.size(); ++k) {
        if (cps[i + k] != mkcps[k]) {
          ok = false;
          break;
        }
      }
      if (ok) {
        flush();
        ids.push_back(p2id_.at(mk));
        i += mkcps.size();
        matched = true;
        break;
      }
    }
    if (!matched) {
      buf.push_back(cps[i]);
      ++i;
    }
  }
  flush();
  return ids;
}

}  // namespace whistle