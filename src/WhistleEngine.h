#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "AudioLayout.h"
#include "CactArchive.h"
#include "Decoder.h"
#include "Encoder.h"
#include "Tokenizer.h"

// STT engine facade composing the .cact parser, weight layer, decoder forward
// pass and SentencePiece tokenizer. Platform-independent (no ESP-IDF deps).
//
// Usage (Arduino): map/download whistle.cact into a PSRAM buffer, then
//   WhistleEngine engine; engine.load(psram_ptr, size);
//   auto ids = engine.encode("prompt");           // -> token ids
//   ids.insert(ids.begin(), engine.bosId());
//   engine.logits(ids.data(), ids.size(), logits); // -> (T x vocab) float32
//   int32_t next = WhistleEngine::argmax(logits + (T-1)*vocab, vocab);
//   std::string text = engine.decode(&next, 1);    // -> " Fr"

namespace whistle {

class WhistleEngine {
 public:
  WhistleEngine() = default;
  ~WhistleEngine() = default;
  WhistleEngine(const WhistleEngine&) = delete;
  WhistleEngine& operator=(const WhistleEngine&) = delete;

  // Parses `data` (the mapped .cact) and prepares weights + decoder + tokenizer.
  // `data` must remain valid (mapped and alive) for the lifetime of this object.
  bool load(const uint8_t* data, size_t size);

  bool ready() const { return ready_; }

  uint32_t vocab() const;
  int32_t bosId() const;
  int32_t eosId() const;

  // SentencePiece encode (text -> token ids).
  std::vector<int32_t> encode(const std::string& text) const;
  // SentencePiece decode (token ids -> text); used for transcripts.
  std::string decode(const int32_t* ids, size_t n) const;

  // Decoder forward over `n` tokens; writes n*vocab() float32 logits row-major
  // [t][vocab]. Prepend bosId() yourself when building a prompt. Returns false
  // if not ready.
  bool logits(const int32_t* tokens, size_t n, float* out);

  // Argmax over a single logits row (vocab floats).
  static int32_t argmax(const float* logits, size_t vocab);

  // True when the archive carries the audio encoder and the speech path is
  // usable (Whistle, not Needle).
  bool audioReady() const { return audio_ready_; }

  // Full speech-to-text: log-mel -> stem -> encoder -> cross-attention decoder
  // greedy decode. `pcm` is 16 kHz mono in [-1,1]. Returns false on failure;
  // on success writes the transcript (possibly empty) to `text`.
  bool transcribe(const float* pcm, size_t num_samples, std::string* text);

  // Optional stage/progress hook (tags such as enc_start, enc_done, pe_done,
  // step0, pre, l0, head). nullptr disables.
  using ProgressFn = void (*)(void* udata, const char* tag);
  void setProgress(ProgressFn fn, void* udata) {
    progress_ = fn;
    progress_udata_ = udata;
    decoder_.progress = fn;
    decoder_.progressUdata = udata;
    decoder_.weights().setProgress(fn, udata);
    encoder_.setProgress(fn, udata);
  }

  // Read-only access to internals (geometry / weights) for diagnostics.
  const cactus::CactArchive& archive() const { return archive_; }
  const Decoder& decoder() const { return decoder_; }

  // Enables the memory-bounded decoder (embedding streamed, per-layer eviction),
  // required to fit a forward in the PSRAM left after staging the model.
  void setBounded(bool b) { decoder_.setBounded(b); }
  bool bounded() const { return decoder_.bounded(); }

  // Speech decoder ladder depth: 0 = full, otherwise 2..num_layers.
  void setDecoderDepth(uint32_t depth) { decoder_.setDepth(depth); }
  uint32_t decoderDepth() const { return decoder_.depth(); }

  // Incremental (one-token) decode with a KV cache. Enabled by default; falls
  // back to the batched full-prefix path if the state cannot be allocated or the
  // token cap is exceeded. Both paths are numerically identical.
  void setIncremental(bool b) { incremental_ = b; }
  bool incremental() const { return incremental_; }

  // Probe aid: with verification on, every incremental step is re-run on the
  // batched path and the largest |logit difference| and the number of argmax
  // mismatches are accumulated (costly; off by default).
  void setVerifyIncremental(bool b) { verify_inc_ = b; }
  float verifyDiff() const { return verify_diff_; }
  uint32_t verifyMismatches() const { return verify_mismatch_; }

  // Hard cap on generated tokens (the loop still stops at eos).
  void setMaxNewTokens(uint32_t n) { max_new_tokens_ = n ? n : 1; }

  // Language hint: 0..6 forces a language (en,de,fr,es,it,nl,pl), >= 7
  // auto-detects (the decoder's own choice among the 7 language tokens).
  void setLanguage(uint8_t lang) { lang_ = lang; }

  // Enables the FWHT-bucket CQ GEMV (dequant-free matmuls).
  void setBucket(bool b) { decoder_.setBucket(b); }
  bool bucket() const { return decoder_.bucket(); }

  // Enables the int8-activation CQ GEMV (fastest, fixed-point buckets).
  void setInt8(bool b) { decoder_.setInt8(b); }
  bool int8() const { return decoder_.int8(); }

 private:
  cactus::CactArchive archive_;
  Decoder decoder_;
  Encoder encoder_;
  Tokenizer tokenizer_;
  AudioLayout audio_layout_;
  bool audio_ready_ = false;
  bool ready_ = false;
  ProgressFn progress_ = nullptr;
  void* progress_udata_ = nullptr;
  uint32_t max_new_tokens_ = 128;
  uint8_t lang_ = 255;  // 0..6 = forced language, >=7 = auto-detect
  bool incremental_ = true;
  bool verify_inc_ = false;
  float verify_diff_ = 0.0f;
  uint32_t verify_mismatch_ = 0;
};

}  // namespace whistle
