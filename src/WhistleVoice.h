#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

#include <ETH.h>

#include "WakeWordDetector.h"

// Public API for on-device voice commands on ESP32-P4.
//
// The library owns the I2S microphone and a background capture task, and
// converts a spoken command to text using the Cactus "Whistle" speech-to-text
// model held in PSRAM. A command begins either on a push-to-talk GPIO press
// or (once a wake-word detector is attached) on a spoken wake word; it ends
// on release (push-to-talk) or detected silence (wake word).
//
// Results are dispatched to the caller's callback from `update()`, so the
// callback runs in the safe context of the sketch's loop().
namespace whistle {

enum class Language : uint8_t {
  Auto = 255,  // detect from speech (default)
  English = 0,
  German,
  French,
  Spanish,
  Italian,
  Dutch,
  Polish,
};

enum class Transport : uint8_t {
  Ethernet = 0,  // USB stick first (when compiled in), then the network download
  Usb = 1,       // USB stick only; never touches the network
};

struct VoiceConfig {
  // Model source. Ethernet tries a USB stick first (when compiled in) and falls
  // back to the download; Usb skips the network entirely (no-Ethernet boards).
  Transport transport = Transport::Ethernet;

  // Ethernet PHY (used only when transport is Ethernet; USB loading ignores it).
  // Defaults target the TLK110 on the M5Stack Unit PoE-P4.
  eth_phy_type_t eth_phy_type = ETH_PHY_TLK110;
  eth_clock_mode_t eth_clk_mode = EMAC_CLK_EXT_IN;
  int eth_phy_addr = 1;
  int eth_phy_mdc = 31;
  int eth_phy_mdio = 52;
  int eth_phy_power = 51;

  // INMP441 I2S pins (board dependent - set for your hardware).
  int i2s_bclk = 0;
  int i2s_ws = 0;
  int i2s_data = 0;

  // Fixed linear gain applied to the microphone before the encoder. Derived on
  // hardware with the MicProbe example (11x lifts conversational speech into
  // range for the INMP441); adjust per board/acoustics. The model expects
  // roughly the level a 16-bit WAV reader produces for the same signal.
  float mic_gain = 11.0f;

  // Push-to-talk GPIO. Audio is recorded while this pin is asserted (see
  // ptt_active_high) and transcribed on release. Set to -1 to disable. The
  // library configures the resting pull automatically: active-high gets an
  // internal pull-down, active-low an internal pull-up.
  int ptt_pin = -1;
  bool ptt_active_high = true;

  // Silence (ms) that ends the utterance after a wake-word trigger.
  uint32_t silence_ms = 800;

  // Speech decoder ladder depth: 0 = full (8), otherwise 2..8. Rungs select
  // blocks in ladder order (both endpoints, bisected interiors), so depth 4 runs
  // blocks 0,3,5,7. Shallower is faster with some accuracy loss.
  uint32_t decoder_depth = 0;

  // Hard cap on generated transcript tokens (the decoder still stops at eos).
  uint32_t max_transcript_tokens = 128;

  // Default model source. Override to point at a self-hosted mirror.
  const char* model_url =
      "https://huggingface.co/Cactus-Compute/whistle/resolve/main/whistle.cact";

  Language language = Language::Auto;
};

class WhistleVoice {
 public:
  using TranscriptCallback = std::function<void(const char* text, bool ok)>;

  WhistleVoice() = default;
  ~WhistleVoice();

  WhistleVoice(const WhistleVoice&) = delete;
  WhistleVoice& operator=(const WhistleVoice&) = delete;

  // Initialises networking, downloads (or locates) the model into PSRAM,
  // starts the I2S capture task. Returns false on failure.
  bool begin(const VoiceConfig& cfg);

  // Registers the transcript callback.
  void onTranscript(TranscriptCallback cb);

  // Must be called from loop(); dispatches pending transcript results.
  void update();

  // Attach an optional wake-word detector (KWS). May be set before or after
  // begin(); when attached, the library also listens for the wake word.
  void setWakeWordDetector(WakeWordDetector* kws);

  // True once the model is loaded and the capture task is running.
  bool isReady() const;

  // Changes the transcription language at runtime.
  void setLanguage(Language lang);

  // Stops capture and frees the model and networking resources. A subsequent
  // begin() re-downloads the model.
  void end();

 private:
  struct Impl;
  Impl* impl_ = nullptr;
};

}  // namespace whistle
