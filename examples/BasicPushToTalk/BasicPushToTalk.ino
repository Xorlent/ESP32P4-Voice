#include <WhistleVoice.h>

using namespace whistle;

// INMP441 I2S pins and push-to-talk GPIO - adjust for your board.
const int I2S_BCLK = 11;
const int I2S_WS = 12;
const int I2S_DATA = 13;
const int PTT_PIN = 8;

WhistleVoice voice;

void onTranscript(const char* text, bool ok) {
  if (ok) {
    Serial.printf("Command: %s\n", text);
  } else {
    Serial.println("Transcription failed");
  }
}

void setup() {
  Serial.begin(115200);

  VoiceConfig cfg;
  cfg.transport = Transport::Ethernet;
  cfg.i2s_bclk = I2S_BCLK;
  cfg.i2s_ws = I2S_WS;
  cfg.i2s_data = I2S_DATA;
  cfg.ptt_pin = PTT_PIN;
  cfg.ptt_active_high = true;
  cfg.mic_gain = 11.0f;  // calibrated with MicProbe for the INMP441
  cfg.language = Language::English;  // force English (default: Language::Auto detects it)

  if (!voice.begin(cfg)) {
    Serial.println("Failed to start voice engine");
    return;
  }
  voice.onTranscript(onTranscript);
  Serial.println("Ready. Hold the PTT button and speak.");
}

void loop() {
  voice.update();
  delay(1);
}
