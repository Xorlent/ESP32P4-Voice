#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cactus {
class CactArchive;
}

namespace whistle {

// Minimal SentencePiece tokenizer: the needle3/whistle .cact stores the
// SentencePiece table as its final (trailing "raw") tensor. Platform-independent
// (no Arduino/ESP-IDF deps).
//
// decode() is the hot path used to turn Whistle decoder token ids into the
// transcript; encode() builds a text prompt.

class Tokenizer {
 public:
  // Reads the token table from the archive's LAST tensor. Returns false if the
  // archive has no tensors or the blob is truncated.
  bool prepare(const cactus::CactArchive& archive);

  int32_t padId() const { return pad_id_; }
  int32_t eosId() const { return eos_id_; }
  int32_t bosId() const { return bos_id_; }
  int32_t unkId() const { return unk_id_; }
  size_t pieceCount() const { return pieces_.size(); }

  // Decodes token ids to text (U+2581 "meta space" replaced with ' ').
  std::string decode(const int32_t* ids, size_t n) const;
  // Encodes raw text (spaces -> meta space, BPE, byte fallback) into ids.
  std::vector<int32_t> encode(const std::string& text) const;

 private:
  std::vector<std::string> pieces_;
  std::vector<float> scores_;
  std::vector<uint8_t> types_;
  int32_t pad_id_ = 0, eos_id_ = 0, bos_id_ = 0, unk_id_ = 0;
  bool add_dummy_ = false, byte_fallback_ = false;

  std::unordered_map<std::string, int32_t> p2id_;
  int32_t byte_id_[256];
  std::vector<std::string> markers_;  // USER_DEFINED pieces, sorted longest first

  std::vector<int32_t> bpe(const std::vector<std::string>& syms) const;
};

}  // namespace whistle