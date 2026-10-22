#include "WhistleVoice.h"

#include <Arduino.h>
#include <cstdio>
#include <string>
#include <utility>

#include <vector>

#include "MicInput.h"
#include "ModelTransport.h"
#include "WhistleEngine.h"
#include "psram.h"

namespace whistle {

struct WhistleVoice::Impl {
  VoiceConfig cfg;
  ModelTransport transport;
  MicInput mic;
  WhistleEngine engine;
  TranscriptCallback on_transcript;
  WakeWordDetector* kws = nullptr;

  uint8_t* model = nullptr;
  size_t model_size = 0;

  bool ready = false;
  bool ptt_was_pressed = false;
  std::vector<float> audio;
};

namespace {

constexpr uint32_t kMaxCaptureSamples = 5u * 16000u;
constexpr size_t kChunkSamples = 640;  // 40 ms @ 16 kHz (small for loop-task stack)

}  // namespace

WhistleVoice::~WhistleVoice() {
  end();
  delete impl_;
  impl_ = nullptr;
}

bool WhistleVoice::begin(const VoiceConfig& cfg) {
  end();
  if (impl_ == nullptr) {
    impl_ = new Impl;
  }
  Impl& im = *impl_;
  im.cfg = cfg;

  // PTT GPIO: configure the resting pull automatically. Active-high logic rests
  // low (pulled HIGH to talk -> needs a pull-down so an open button reads idle);
  // active-low rests high -> pull-up.
  if (cfg.ptt_pin >= 0) {
    pinMode(cfg.ptt_pin, cfg.ptt_active_high ? INPUT_PULLDOWN : INPUT_PULLUP);
    im.ptt_was_pressed = digitalRead(cfg.ptt_pin) != 0;
  }

  MicConfig mic_cfg;
  mic_cfg.bclk = cfg.i2s_bclk;
  mic_cfg.ws = cfg.i2s_ws;
  mic_cfg.din = cfg.i2s_data;
  mic_cfg.gain = cfg.mic_gain;
  if (!im.mic.begin(mic_cfg)) {
    Serial.println("[WhistleVoice] mic init failed");
    return false;
  }

  // Model source: USB-only boards skip the network; otherwise a USB stick is
  // tried first (when compiled in) with the download as fallback.
  ModelTransportConfig tcfg;
  tcfg.phy_type = cfg.eth_phy_type;
  tcfg.clk_mode = cfg.eth_clk_mode;
  tcfg.phy_addr = cfg.eth_phy_addr;
  tcfg.phy_mdc = cfg.eth_phy_mdc;
  tcfg.phy_mdio = cfg.eth_phy_mdio;
  tcfg.phy_power = cfg.eth_phy_power;
  if (cfg.transport == Transport::Usb) {
    im.model = im.transport.fetchUsb(&im.model_size);
  } else {
    im.model = im.transport.load(&im.model_size, tcfg, cfg.model_url);
  }
  if (im.model == nullptr) {
    Serial.println(cfg.transport == Transport::Usb
                       ? "[WhistleVoice] USB model load failed"
                       : "[WhistleVoice] model load failed");
    return false;
  }

  im.engine.setBounded(true);  // required to fit a forward in device PSRAM
  im.engine.setDecoderDepth(cfg.decoder_depth);
  im.engine.setMaxNewTokens(cfg.max_transcript_tokens);
  im.engine.setLanguage(static_cast<uint8_t>(cfg.language));
  if (!im.engine.load(im.model, im.model_size)) {
    Serial.println("[WhistleVoice] engine load failed");
    psram::free(im.model);
    im.model = nullptr;
    return false;
  }

  if (im.kws != nullptr) {
    im.kws->begin();
  }

  im.ready = true;
  Serial.println("[WhistleVoice] ready (push-to-talk transcriber)");
  return true;
}

void WhistleVoice::end() {
  if (impl_ == nullptr) {
    return;
  }
  Impl& im = *impl_;
  im.mic.end();
  im.transport.end();
  if (im.model != nullptr) {
    psram::free(im.model);
    im.model = nullptr;
  }
  im.audio.clear();
  im.ready = false;
}

void WhistleVoice::onTranscript(TranscriptCallback cb) {
  if (impl_ != nullptr) {
    impl_->on_transcript = std::move(cb);
  }
}

void WhistleVoice::update() {
  if (impl_ == nullptr || !impl_->ready) {
    return;
  }
  Impl& im = *impl_;

  const bool has_ptt = im.cfg.ptt_pin >= 0;
  bool asserted = false;
  if (has_ptt) {
    const bool pressed = digitalRead(im.cfg.ptt_pin) != 0;
    asserted = im.cfg.ptt_active_high ? pressed : !pressed;

    if (asserted && !im.ptt_was_pressed) {
      im.audio.clear();  // press edge: start a fresh capture
    } else if (!asserted && im.ptt_was_pressed) {
      std::string text;
      const bool ok = !im.audio.empty() &&
                      im.engine.transcribe(im.audio.data(), im.audio.size(),
                                           &text);
      if (im.on_transcript) im.on_transcript(text.c_str(), ok);
    }
    im.ptt_was_pressed = pressed;
  }

  if (asserted) {
    float chunk[kChunkSamples];
    const size_t n = im.mic.read(chunk, kChunkSamples);
    if (n > 0 && im.audio.size() < kMaxCaptureSamples) {
      const size_t take = (n > kMaxCaptureSamples - im.audio.size())
                              ? kMaxCaptureSamples - im.audio.size()
                              : n;
      im.audio.insert(im.audio.end(), chunk, chunk + take);
    }
  }
}

void WhistleVoice::setWakeWordDetector(WakeWordDetector* kws) {
  if (impl_ != nullptr) {
    impl_->kws = kws;
  }
}

bool WhistleVoice::isReady() const {
  return impl_ != nullptr && impl_->ready;
}

void WhistleVoice::setLanguage(Language lang) {
  if (impl_ != nullptr) {
    impl_->cfg.language = lang;
    impl_->engine.setLanguage(static_cast<uint8_t>(lang));
  }
}

}  // namespace whistle