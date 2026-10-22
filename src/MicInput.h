#pragma once

#include <cstddef>
#include <cstdint>

#include <driver/i2s_types.h>

namespace whistle {

// INMP441 I2S MEMS microphone capture for the ESP32-P4 (IDF5 i2s_std driver).
//
// Captures 16 kHz mono audio into float samples in roughly [-1, 1]. The INMP441
// drives 24 signal bits left-justified in a 32-bit slot; we keep the top 16
// bits and scale by 2^-15, so a unity-gain capture matches what a 16-bit WAV
// reader produces for the same signal. That is the level convention the model
// saw in training, which is why absolute gain matters even for log-mel input.
//
// The INMP441 is quiet: nominal -26 dBFS at 94 dB SPL, so conversational
// speech lands around -40..-60 dBFS. A fixed linear gain (config or
// setGain()) lifts that back into range; derive the value on hardware with the
// MicProbe example and then bake it into MicConfig::gain.
struct MicConfig {
  // I2S pins. All three are required (din is the INMP441 SD line).
  int bclk = -1;
  int ws = -1;
  int din = -1;

  uint32_t sample_rate = 16000;  // Whistle expects 16 kHz.

  // Fixed linear gain applied to the float/int16 output. 1.0 = unity.
  float gain = 1.0f;

  // Bits to discard from the raw 32-bit slot. 16 keeps the top 16 signal bits
  // (canonical INMP441 -> int16 path). Set 0 with the MicProbe example if the
  // captured DC/level suggests the data is not left-justified on your board.
  uint32_t bit_shift = 16;

  // DMA sizing. Larger counts tolerate longer stalls but consume more RAM.
  uint32_t dma_desc_num = 4;
  uint32_t dma_frame_num = 240;
};

class MicInput {
 public:
  MicInput() = default;
  ~MicInput();

  MicInput(const MicInput&) = delete;
  MicInput& operator=(const MicInput&) = delete;

  // Initialises the RX channel and starts the DMA. Returns false if a required
  // pin is unset or the driver rejects the config (error logged to Serial).
  bool begin(const MicConfig& cfg);

  // Stops and releases the channel.
  void end();

  bool running() const { return running_; }

  // Blocking read of up to max_samples mono float samples in roughly [-1, 1].
  // Returns the count actually produced; a short read (normally a driver
  // timeout with no data available) is not an error state.
  size_t read(float* dst, size_t max_samples);

  // Same, but as int16 PCM scaled by 2^15 (for the wake-word feed hook).
  size_t readS16(int16_t* dst, size_t max_samples);

  // Live gain tweak (same factor as MicConfig::gain). Use for bring-up tuning.
  void setGain(float gain) { cfg_.gain = gain; }

  const MicConfig& config() const { return cfg_; }

 private:
  MicConfig cfg_;
  i2s_chan_handle_t rx_ = nullptr;
  bool running_ = false;
  uint32_t* raw_ = nullptr;  // scratch, dma_frame_num words
};

}  // namespace whistle