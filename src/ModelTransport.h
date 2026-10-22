#pragma once

#include <cstddef>
#include <cstdint>

#include <ETH.h>

namespace whistle {

struct ModelTransportConfig {
  // Model source (HTTP or HTTPS; HTTPS for the HuggingFace default).
  const char* url =
      "https://huggingface.co/Cactus-Compute/whistle/resolve/main/whistle.cact";

  eth_phy_type_t phy_type = ETH_PHY_TLK110;
  eth_clock_mode_t clk_mode = EMAC_CLK_EXT_IN;
  int phy_addr = 1;
  int phy_mdc = 31;
  int phy_mdio = 52;
  int phy_power = 51;

  uint32_t link_timeout_ms = 15000;   // wait for PHY link before DHCP
  uint32_t dhcp_timeout_ms = 30000;   // wait for a DHCP lease
  uint32_t http_timeout_ms = 30000;   // per-stream idle timeout (ms)
  const char* hostname = "esp32p4-voice";
};

// ESP32-specific model transport: boots Ethernet (TLK110 PHY on the M5Stack
// Unit PoE-P4) and HTTP(S)-downloads a .cact into a fresh PSRAM buffer that is
// handed to WhistleEngine::load. This is deliberately separate from the
// platform-independent decoder core (Decoder/Tokenizer/WhistleEngine), which
// has no ESP-IDF dependencies.
class ModelTransport {
 public:
  ~ModelTransport();

  // Initialises the PHY and waits for a DHCP lease. Returns false on timeout.
  bool begin(const ModelTransportConfig& cfg = ModelTransportConfig{});

  bool connected() const { return got_ip_; }
  const char* ipString();

  // Blocks: downloads the URL (following redirects) into PSRAM. On success
  // returns a PSRAM pointer (free with psram::free()) and sets *size. On
  // failure frees nothing and returns nullptr. `url` overrides cfg.url if set.
  uint8_t* fetch(size_t* size, const char* url = nullptr);

  // Loads the model from a FAT32 USB mass-storage drive (/whistle.cact at the
  // drive root). Returns a PSRAM buffer, or nullptr when no drive/file is found.
  // Does not need begin()/Ethernet, and is a no-op (nullptr) unless
  // WHISTLE_USB_MSC is defined to 1.
  uint8_t* fetchUsb(size_t* size);

  // One-shot load: tries the USB stick first (when compiled in), then falls back
  // to the Ethernet download, so callers never need to know the source. Returns
  // a PSRAM buffer (free with psram::free()) and sets *size, or nullptr on
  // failure. `url` overrides cfg.url for the network fallback.
  uint8_t* load(size_t* size,
                const ModelTransportConfig& cfg = ModelTransportConfig{},
                const char* url = nullptr);

  // Tears down Ethernet (frees the netif).
  void end();

 private:
  ModelTransportConfig cfg_;
  bool began_ = false;
  bool got_ip_ = false;
  char ipbuf_[20] = {0};

  bool splitUrl(const char* url, bool* https, char* host, size_t hostcap,
                uint16_t* port, char* path, size_t pathcap) const;
};

}  // namespace whistle