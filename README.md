# ESP32P4 Voice

On-device speech-to-text for ESP32-P4 boards. Speak a command, and it is
transcribed entirely on device using the Cactus Whistle model.

## What it does

- Records from an I2S MEMS microphone (INMP441) at 16 kHz while you hold a
  push-to-talk button, then transcribes on release.
- Runs the Cactus **Whistle** model (16.9 MB `whistle.cact`) from PSRAM.
- Returns the transcript in any of 7 languages (auto-detected; an optional hint
  is configurable via `Language`).

## What you need

- An **ESP32-P4** board with 32 MB PSRAM (tested on the **M5Stack Unit PoE-P4**).
- An INMP441 I2S microphone and a momentary push button.
- Ethernet for the one-time model download at boot OR a USB memory stick with [whistle.cact](https://huggingface.co/Cactus-Compute/whistle/resolve/main/whistle.cact).


## Wiring

| INMP441 | ESP32-P4 |
|---|---|
| VDD | 3V3 |
| GND | GND |
| SCK | I2S bit-clock pin |
| WS | I2S word-select pin |
| SD | I2S data pin |
| L/R | GND |

Button: one leg to the push-to-talk GPIO, the other to GND.

## Install

1. Install Arduino IDE 2.3.10
2. In **Boards Manager**, add esp32 by Espressif Systems version 3.3.12
3. Add this library via **Sketch -> Include Library -> Add .ZIP Library**, or
   copy this folder into your `Arduino/libraries/` directory.

## Quick start

```cpp
#include <WhistleVoice.h>
using namespace whistle;

WhistleVoice voice;

void onTranscript(const char* text, bool ok) {
  if (ok) Serial.printf("Command: %s\n", text);
}

void setup() {
  Serial.begin(115200);

  VoiceConfig cfg;
  cfg.transport = Transport::Ethernet;
  cfg.i2s_bclk  = 11;    // INMP441 SCK
  cfg.i2s_ws    = 12;    // INMP441 WS
  cfg.i2s_data  = 13;    // INMP441 SD
  cfg.ptt_pin   = 8;     // push-to-talk GPIO
  cfg.mic_gain  = 11.0f;

  if (!voice.begin(cfg)) {
    Serial.println("Failed to start (check Ethernet).");
    return;
  }
  voice.onTranscript(onTranscript);
  Serial.println("Ready. Hold the button and speak.");
}

void loop() {
  voice.update();
  delay(1);
}
```

Pin numbers and `mic_gain` above match `BasicPushToTalk` and are set for the
**M5Stack Unit PoE-P4**; adjust the pins for a different board, and tune the gain
with the `MicProbe` example. A ready-made sketch is at
**File -> Examples -> ESP32P4 Voice -> BasicPushToTalk**.

### Recording length

A single push-to-talk utterance is capped at **5 seconds**. Holding the button
longer is safe, but the extra audio is silently ignored, so keep commands to a
few seconds - 1 to 3 seconds is the range the examples are tuned around. Longer
clips also take proportionally longer to transcribe (the encoder processes the
whole utterance at once).

### Language

The model transcribes in any of its 7 languages and auto-detects the language
from speech by default. To force a language, set `VoiceConfig::language` (or call
`voice.setLanguage(...)` at runtime):

```cpp
VoiceConfig cfg;
cfg.language = Language::English;   // force English (default behaviour: Language::Auto)
```

`Language` values are `Auto` (default), `English`, `German`, `French`, `Spanish`,
`Italian`, `Dutch`, `Polish`. With `Auto`, the decoder picks the language token
itself after the first token.

### Configuring Ethernet (optional)

The PHY type, RMII clock mode and pins are fields on `VoiceConfig`, defaulting to
the TLK110 on the M5Stack Unit PoE-P4. For a different board or PHY, set them in
your sketch:

```cpp
VoiceConfig cfg;
cfg.eth_phy_type = ETH_PHY_LAN8720;    // PHY chip (eth_phy_type_t enum)
cfg.eth_clk_mode = EMAC_CLK_EXT_IN;     // RMII clock mode (eth_clock_mode_t)
cfg.eth_phy_addr = 0;
cfg.eth_phy_mdc  = 23;
cfg.eth_phy_mdio = 18;
cfg.eth_phy_power = -1;                 // -1 = no power pin
```

`eth_phy_type` takes an `eth_phy_type_t` (`ETH_PHY_TLK110`, `ETH_PHY_LAN8720`,
`ETH_PHY_KSZ8081`, ...) and `eth_clk_mode` an `eth_clock_mode_t`, both from the
esp32 core's `<ETH.h>`, so invalid values fail to compile. These are only used by
the network download; USB loading ignores them.

### Loading the model from a USB drive (optional)

Place `whistle.cact` in the root of a FAT32 USB stick and plug it into the Type-C
port on the side of the Unit-PoE-P4 to load the model locally instead of
downloading it. To turn the feature on, create a file named `build_opt.h` in
your sketch folder (next to the `.ino`) containing:

    -DWHISTLE_USB_MSC=1
    -DUSBHOST_MSC_MAX_SECTORS=64

If the stick cannot be read, the library falls back to the network download.
__Note: The USB stick feature uses approximately 49KB of additional RAM.__

For a board **without Ethernet**, set the transport to USB so the library skips
the network entirely (it never initializes the PHY or waits for DHCP):

```cpp
cfg.transport = Transport::Usb;
```

## Examples

The library ships three example sketches. **BasicPushToTalk** and
**TranscriptProbe** each include a `build_opt.h` that turns on USB memory-stick
loading (see above), so they load the model from a USB stick when one is
plugged in and fall back to the network download otherwise.

- **BasicPushToTalk** - the interactive transcriber. Hold the push-to-talk
  button, speak, release, and the transcript is printed over serial. This is
  the sketch to flash for a working voice command. USB loading: **enabled**;
  disable by editing `build_opt.h`: -DWHISTLE_USB_MSC=0

- **TranscriptProbe** - a diagnostics tool that runs a fixed 3 s audio clip
  through the full pipeline and prints per-stage timing plus the transcript, so
  you can check accuracy and latency against a known expected result.
  USB loading: **enabled**; disable by editing `build_opt.h`: -DWHISTLE_USB_MSC=0

- **MicProbe** - live microphone level and gain tuning. Send `gain <n>` over
  serial to find the right `mic_gain` for your mic and wiring before baking it
  into `VoiceConfig`.

## Credits

Whistle and its weights are by [Cactus Compute](https://huggingface.co/Cactus-Compute/whistle)
(Apache 2.0). This library is an independent ESP32-P4 implementation.
