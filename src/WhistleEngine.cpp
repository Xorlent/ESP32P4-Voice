#include "WhistleEngine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>



namespace whistle {
namespace {

void fakeQuantRows(float* x, size_t rows, size_t cols) {
  const float qmax = 127.0f;
  for (size_t r = 0; r < rows; ++r) {
    float* row = x + r * cols;
    float amax = 0.0f;
    for (size_t c = 0; c < cols; ++c) {
      const float a = std::fabs(row[c]);
      if (a > amax) amax = a;
    }
    const float scale = (amax > 0.0f) ? amax / qmax : 1.0f;
    for (size_t c = 0; c < cols; ++c) {
      float v = std::nearbyintf(row[c] / scale);
      if (v < -128.0f) v = -128.0f;
      else if (v > 127.0f) v = 127.0f;
      row[c] = v * scale;
    }
  }
}

}  // namespace


bool WhistleEngine::load(const uint8_t* data, size_t size) {
  ready_ = false;
  if (!archive_.parse(data, size)) return false;
  if (!decoder_.prepare(archive_)) return false;
  audio_ready_ = encoder_.prepare(archive_, decoder_.weights());
  if (audio_ready_) {
    audio_layout_ = encoder_.layout();
    decoder_.setInt8(true);
  }

  if (!tokenizer_.prepare(archive_)) return false;
  ready_ = true;
  return true;
}

uint32_t WhistleEngine::vocab() const {
  const ModelGeometry& g = decoder_.weights().geometry();
  return g.out_vocab ? g.out_vocab : g.vocab;
}

int32_t WhistleEngine::bosId() const { return tokenizer_.bosId(); }
int32_t WhistleEngine::eosId() const { return tokenizer_.eosId(); }

std::vector<int32_t> WhistleEngine::encode(const std::string& text) const {
  return tokenizer_.encode(text);
}

std::string WhistleEngine::decode(const int32_t* ids, size_t n) const {
  return tokenizer_.decode(ids, n);
}

bool WhistleEngine::logits(const int32_t* tokens, size_t n, float* out) {
  if (!ready_) return false;
  return decoder_.forward(tokens, n, out);
}

int32_t WhistleEngine::argmax(const float* logits, size_t vocab) {
  int32_t best = 0;
  float best_score = logits[0];
  for (size_t v = 1; v < vocab; ++v) {
    if (logits[v] > best_score) {
      best_score = logits[v];
      best = static_cast<int32_t>(v);
    }
  }
  return best;
}

bool WhistleEngine::transcribe(const float* pcm, size_t num_samples,
                               std::string* text) {
  if (!ready_ || !audio_ready_ || pcm == nullptr || text == nullptr)
    return false;

  if (progress_) progress_(progress_udata_, "enc_start");

  std::vector<float> mem;
  size_t frames = 0;
  if (!encoder_.encode(pcm, num_samples, &mem, &frames) || frames == 0)
    return false;
  if (progress_) progress_(progress_udata_, "enc_done");


  const uint32_t d = audio_layout_.d_model;
  float pe_gate = 0.0f;
  if (!decoder_.weights().readIndex(audio_layout_.pe_gate, &pe_gate, 1))
    return false;
  const float gate = 1.0f / (1.0f + std::exp(-pe_gate));
  const uint32_t half = d / 2;
  const float denom = static_cast<float>(half > 1 ? half - 1 : 1);
  for (size_t t = 0; t < frames; ++t)
    for (uint32_t j = 0; j < half; ++j) {
      const float ang = static_cast<float>(t) *
                        std::exp(-std::log(10000.0f) / denom *
                                 static_cast<float>(j));
      mem[t * d + j] += gate * std::sin(ang);
      mem[t * d + half + j] += gate * std::cos(ang);
    }
  if (progress_) progress_(progress_udata_, "pe_done");

  fakeQuantRows(mem.data(), frames, d);

  // Project the memory K/V once for the whole utterance; the greedy loop would
  // otherwise repeat it for every block at every step.
  const uint32_t cachedBlocks =
      decoder_.prepareCrossAttention(mem.data(), frames, audio_layout_);
  if (progress_) {
    char tag[16];
    std::snprintf(tag, sizeof(tag), "xkv%u", (unsigned)cachedBlocks);
    progress_(progress_udata_, tag);
  }

  const uint32_t vocab = decoder_.weights().geometry().vocab;
  const uint32_t text_vocab = vocab - 7;
  std::vector<int32_t> tokens;
  tokens.push_back(bosId());
  std::vector<float> logits;
  const size_t kMaxNew = max_new_tokens_;
  // Incremental steps keep per-block K/V state; the batched path stays as the
  // fallback and as the numerical oracle (both give identical logits).
  bool useInc = incremental_ && decoder_.beginSequence(audio_layout_);
  if (progress_) progress_(progress_udata_, useInc ? "inc" : "batched");
  for (size_t step = 0; step < kMaxNew; ++step) {
    const size_t T = tokens.size();
    if (progress_) {
      char tag[16];
      std::snprintf(tag, sizeof(tag), "step%u", (unsigned)step);
      progress_(progress_udata_, tag);
    }

    logits.assign(T * vocab, 0.0f);
    bool fwd = false;
    if (useInc) {
      // forwardStep writes the single new row; place it where the batched path
      // would put the last prefix row.
      fwd = decoder_.forwardStep(tokens.data(), T, mem.data(), frames,
                                 audio_layout_,
                                 logits.data() + (T - 1) * vocab);
      if (!fwd) {
        decoder_.endSequence();
        useInc = false;
      }
    }
    if (!useInc) {
      fwd = decoder_.forwardWithMemory(tokens.data(), T, mem.data(), frames,
                                       audio_layout_, logits.data(),
                                       static_cast<int32_t>(T - 1));
    }
    if (!fwd) {
      decoder_.endSequence();
      decoder_.dropCrossKv();
      if (progress_) progress_(progress_udata_, "fwd_fail");
      return false;
    }
    if (useInc && verify_inc_) {
      // Same prefix on the batched path: the two must agree bit-for-bit up to
      // float associativity, so any drift here points at the cached state.
      std::vector<float> ref(T * vocab, 0.0f);
      if (decoder_.forwardWithMemory(tokens.data(), T, mem.data(), frames,
                                     audio_layout_, ref.data(),
                                     static_cast<int32_t>(T - 1))) {
        const float* a = logits.data() + (T - 1) * vocab;
        const float* b = ref.data() + (T - 1) * vocab;
        float maxd = 0.0f;
        uint32_t ia = 0;
        uint32_t ib = 0;
        for (uint32_t v = 0; v < vocab; ++v) {
          const float dd = std::fabs(a[v] - b[v]);
          if (dd > maxd) maxd = dd;
          if (a[v] > a[ia]) ia = v;
          if (b[v] > b[ib]) ib = v;
        }
        if (maxd > verify_diff_) verify_diff_ = maxd;
        if (ia != ib) ++verify_mismatch_;
      }
    }
    const float* last = logits.data() + (T - 1) * vocab;
    int32_t best;
    if (T == 1 && lang_ < 7) {
      // Forced language hint: the 7 language tokens immediately follow the text
      // vocab (en,de,fr,es,it,nl,pl), so pin the first token instead of argmax.
      best = static_cast<int32_t>(text_vocab) + lang_;
    } else {
      const uint32_t lo = (T == 1) ? text_vocab : 0;
      const uint32_t hi = (T == 1) ? vocab : text_vocab;
      best = static_cast<int32_t>(lo);
      float best_score = last[lo];
      for (uint32_t v = lo + 1; v < hi; ++v) {
        if (last[v] > best_score) {
          best_score = last[v];
          best = static_cast<int32_t>(v);
        }
      }
    }
    if (best == eosId()) break;
    tokens.push_back(best);
  }

  size_t end = tokens.size();
  for (size_t i = 2; i < tokens.size(); ++i) {
    if (tokens[i] == eosId()) {
      end = i;
      break;
    }
  }
  if (end > 2)
    *text = tokenizer_.decode(tokens.data() + 2, end - 2);
  else
    text->clear();
  decoder_.endSequence();
  decoder_.dropCrossKv();
  return true;
}

}  // namespace whistle
