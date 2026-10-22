#include "MicInput.h"

#include <new>

#include <Arduino.h>
#include <driver/i2s_common.h>
#include <driver/i2s_std.h>
#include <freertos/FreeRTOS.h>

#include <cmath>

namespace whistle {

namespace {
constexpr float kScale16 = 1.0f / 32768.0f;

void logError(const char* step, esp_err_t err) {
  Serial.printf("[MicInput] %s failed: 0x%x\n", step, static_cast<int>(err));
}

// Single blocking read into buf; returns the number of 32-bit words received.
size_t readChunk(i2s_chan_handle_t rx, uint32_t* buf, size_t max_words) {
  size_t bytes = 0;
  i2s_channel_read(rx, buf, max_words * sizeof(uint32_t), &bytes,
                   pdMS_TO_TICKS(250));
  return bytes / sizeof(uint32_t);
}
}  // namespace

MicInput::~MicInput() {
  end();
}

bool MicInput::begin(const MicConfig& cfg) {
  end();
  cfg_ = cfg;

  if (cfg_.bclk < 0 || cfg_.ws < 0 || cfg_.din < 0) {
    Serial.println("[MicInput] bclk/ws/din pins must all be set");
    return false;
  }

  raw_ = new (std::nothrow) uint32_t[cfg_.dma_frame_num];
  if (raw_ == nullptr) {
    Serial.println("[MicInput] failed to allocate DMA scratch");
    return false;
  }

  i2s_chan_config_t chan =
      I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  chan.dma_desc_num = cfg_.dma_desc_num;
  chan.dma_frame_num = cfg_.dma_frame_num;

  esp_err_t err = i2s_new_channel(&chan, nullptr, &rx_);
  if (err != ESP_OK) {
    logError("i2s_new_channel", err);
    end();
    return false;
  }

  i2s_std_config_t std_cfg = {};
  std_cfg.clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(cfg_.sample_rate);
  std_cfg.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
      I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO);
  // INMP441 L/R is tied low, so the signal is in the left slot.
  std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

  std_cfg.gpio_cfg = {
      .mclk = GPIO_NUM_NC,
      .bclk = static_cast<gpio_num_t>(cfg_.bclk),
      .ws = static_cast<gpio_num_t>(cfg_.ws),
      .dout = GPIO_NUM_NC,
      .din = static_cast<gpio_num_t>(cfg_.din),
      .invert_flags = {false, false, false},
  };

  err = i2s_channel_init_std_mode(rx_, &std_cfg);
  if (err != ESP_OK) {
    logError("i2s_channel_init_std_mode", err);
    end();
    return false;
  }

  err = i2s_channel_enable(rx_);
  if (err != ESP_OK) {
    logError("i2s_channel_enable", err);
    end();
    return false;
  }

  running_ = true;
  return true;
}

void MicInput::end() {
  if (rx_ != nullptr) {
    i2s_channel_disable(rx_);
    i2s_del_channel(rx_);
    rx_ = nullptr;
  }
  delete[] raw_;
  raw_ = nullptr;
  running_ = false;
}

size_t MicInput::read(float* dst, size_t max_samples) {
  if (!running_ || dst == nullptr) {
    return 0;
  }
  size_t out = 0;
  while (out < max_samples) {
    size_t want = max_samples - out;
    if (want > cfg_.dma_frame_num) {
      want = cfg_.dma_frame_num;
    }
    size_t got = readChunk(rx_, raw_, want);
    if (got == 0) {
      break;
    }
    for (size_t i = 0; i < got; ++i) {
      const int32_t s = static_cast<int32_t>(raw_[i]) >> cfg_.bit_shift;
      dst[out++] = static_cast<float>(s) * kScale16 * cfg_.gain;
    }
  }
  return out;
}

size_t MicInput::readS16(int16_t* dst, size_t max_samples) {
  if (!running_ || dst == nullptr) {
    return 0;
  }
  size_t out = 0;
  while (out < max_samples) {
    size_t want = max_samples - out;
    if (want > cfg_.dma_frame_num) {
      want = cfg_.dma_frame_num;
    }
    size_t got = readChunk(rx_, raw_, want);
    if (got == 0) {
      break;
    }
    for (size_t i = 0; i < got; ++i) {
      const int32_t s = static_cast<int32_t>(raw_[i]) >> cfg_.bit_shift;
      int32_t t = static_cast<int32_t>(lroundf(static_cast<float>(s) * cfg_.gain));
      if (t > 32767) {
        t = 32767;
      } else if (t < -32768) {
        t = -32768;
      }
      dst[out++] = static_cast<int16_t>(t);
    }
  }
  return out;
}

}  // namespace whistle