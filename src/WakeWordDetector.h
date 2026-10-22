#pragma once

#include <cstddef>
#include <cstdint>

// Hook for a future wake-word (keyword-spotting) stage.
//
// The library feeds contiguous 16 kHz mono PCM here while listening for the
// wake word; when triggered() goes true, capture switches from
// "listening for the wake word" to "recording the utterance" (ending at
// silence). Leaving the detector unset gives push-to-talk-only operation, so
// this interface is the only place a wake-word stage needs to plug in.
namespace whistle {

class WakeWordDetector {
 public:
  virtual ~WakeWordDetector() = default;

  virtual bool begin() = 0;

  // Feeds a chunk of captured PCM samples (int16, mono) to the detector.
  virtual void feed(const int16_t* pcm, size_t samples) = 0;

  // True once the wake word has been detected.
  virtual bool triggered() const = 0;

  // Clears internal state after a detection is consumed.
  virtual void reset() = 0;
};

}  // namespace whistle
