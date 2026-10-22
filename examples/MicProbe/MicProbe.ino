#include <MicInput.h>

#include <cmath>

using namespace whistle;

// INMP441 I2S pins - adjust for your board. Calibrated gain for this wiring is
// 11x (bake into MicConfig::gain / VoiceConfig::mic_gain once confirmed).
const int I2S_BCLK = 12;
const int I2S_WS = 13;
const int I2S_DATA = 14;

// Captures a short burst every tick and prints level stats. Live-tune the gain
// from the serial monitor with "gain <multiplier>" (e.g. "gain 8"), then bake
// the winning value into MicConfig::gain / VoiceConfig::mic_gain.
MicInput mic;

void setup() {
  Serial.begin(115200);
  delay(500);

  MicConfig cfg;
  cfg.bclk = I2S_BCLK;
  cfg.ws = I2S_WS;
  cfg.din = I2S_DATA;
  cfg.gain = 1.0f;

  if (!mic.begin(cfg)) {
    Serial.println("Mic init failed - check pins and wiring.");
    return;
  }
  Serial.println("Mic ready. Send 'gain <n>' to tune. Idle = noise floor.");
}

double toDb(double v) {
  return 20.0 * log10(v > 1e-12 ? v : 1e-12);
}

void loop() {
  static float buf[1600];
  static uint32_t ticks = 0;

  size_t n = mic.read(buf, sizeof(buf) / sizeof(buf[0]));
  if (n > 0) {
    double sum = 0.0, sumSq = 0.0, peak = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double v = buf[i];
      sum += v;
      sumSq += v * v;
      double a = fabs(v);
      if (a > peak) peak = a;
    }
    double dc = sum / n;
    double rms = sqrt(sumSq / n);

    if ((ticks++ % 5) == 0) {
      Serial.printf("gain=%.1f n=%u peak=%6.1f dB  rms=%6.1f dB  dc=%+7.4f\n",
                    mic.config().gain, (unsigned)n, toDb(peak), toDb(rms), dc);
    }
  }

  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    line.toLowerCase();
    if (line.startsWith("gain")) {
      float g = line.substring(4).toFloat();
      if (g > 0.0f) {
        mic.setGain(g);
        Serial.printf("gain set to %.1f\n", g);
      } else {
        Serial.println("usage: gain <positive multiplier>");
      }
    }
  }

  delay(20);
}
