#include "ModelTransport.h"

#include <cstdio>
#include <cstring>

#include <Arduino.h>
#include <ETH.h>
#include <HTTPClient.h>
#include <Network.h>
#include <NetworkClientSecure.h>

#include "psram.h"

// Optional USB mass-storage model source: a FAT32 stick with whistle.cact in its
// root. Off by default. Enable it from the sketch with a "build_opt.h" file in
// the sketch folder containing "-DWHISTLE_USB_MSC=1" (and, for speed,
// "-DUSBHOST_MSC_MAX_SECTORS=64" - the core's host driver otherwise transfers
// one 512-byte sector per SCSI round-trip, which makes the model load very
// slow). The ESP32 core injects build_opt.h into every file in the build. Do not
// edit this file to change the default.
#ifndef WHISTLE_USB_MSC
#define WHISTLE_USB_MSC 0
#endif

#if WHISTLE_USB_MSC
#include <USBHost.h>
#include <USBHostMSC.h>
#include <USBMSCFS.h>
#endif

namespace whistle {

namespace {

bool s_got_ip = false;

#if WHISTLE_USB_MSC
constexpr const char* kUsbMountpoint = "/usb";
constexpr const char* kUsbModelPath = "/whistle.cact";
constexpr uint32_t kUsbTimeoutMs = 3000;
USBMSCFSClass s_usbFs;
#endif

// Runs on the Ethernet event task (not loop()); only touches volatile state.
void onEthEvent(arduino_event_id_t event) {
  switch (event) {
    case ARDUINO_EVENT_ETH_START:
      ETH.setHostname("esp32p4-voice");
      break;
    case ARDUINO_EVENT_ETH_GOT_IP:
      s_got_ip = true;
      break;
    case ARDUINO_EVENT_ETH_LOST_IP:
    case ARDUINO_EVENT_ETH_DISCONNECTED:
    case ARDUINO_EVENT_ETH_STOP:
      s_got_ip = false;
      break;
    default:
      break;
  }
}

}  // namespace

ModelTransport::~ModelTransport() { end(); }

bool ModelTransport::begin(const ModelTransportConfig& cfg) {
  cfg_ = cfg;
  s_got_ip = false;
  Network.onEvent(onEthEvent);

  const bool phy_ok = ETH.begin(cfg_.phy_type, cfg_.phy_addr, cfg_.phy_mdc,
                                cfg_.phy_mdio, cfg_.phy_power, cfg_.clk_mode);
  if (!phy_ok) return false;

  const uint32_t t0 = millis();
  while (!s_got_ip && (millis() - t0) < cfg_.dhcp_timeout_ms) {
    delay(50);
  }
  got_ip_ = s_got_ip;
  began_ = true;
  return got_ip_;
}

const char* ModelTransport::ipString() {
  if (!got_ip_) return "(no ip)";
  const String ip = ETH.localIP().toString();
  std::snprintf(ipbuf_, sizeof ipbuf_, "%s", ip.c_str());
  return ipbuf_;
}

bool ModelTransport::splitUrl(const char* url, bool* https, char* host,
                              size_t hostcap, uint16_t* port, char* path,
                              size_t pathcap) const {
  const char* p = url;
  if (std::strncmp(p, "https://", 8) == 0) {
    *https = true;
    p += 8;
    *port = 443;
  } else if (std::strncmp(p, "http://", 7) == 0) {
    *https = false;
    p += 7;
    *port = 80;
  } else {
    return false;
  }

  const char* slash = std::strchr(p, '/');
  const char* colon = std::strchr(p, ':');
  size_t hostlen;
  if (slash == nullptr) {
    hostlen = std::strlen(p);
  } else if (colon != nullptr && colon < slash) {
    hostlen = static_cast<size_t>(colon - p);
  } else {
    hostlen = static_cast<size_t>(slash - p);
  }
  if (hostlen == 0 || hostlen >= hostcap) return false;
  std::memcpy(host, p, hostlen);
  host[hostlen] = '\0';

  if (colon != nullptr && (slash == nullptr || colon < slash)) {
    *port = static_cast<uint16_t>(std::atoi(colon + 1));
  }
  if (slash != nullptr) {
    std::snprintf(path, pathcap, "%s", slash);
  } else {
    path[0] = '/';
    path[1] = '\0';
  }
  return true;
}

uint8_t* ModelTransport::fetch(size_t* size, const char* url) {
  if (!began_) return nullptr;
  if (size != nullptr) *size = 0;
  const char* target = url != nullptr ? url : cfg_.url;

  bool https;
  char host[256], path[1024];
  uint16_t port;
  if (!splitUrl(target, &https, host, sizeof host, &port, path, sizeof path)) {
    return nullptr;
  }

  HTTPClient http;
  http.setConnectTimeout(static_cast<int32_t>(cfg_.http_timeout_ms));
  http.setTimeout(static_cast<uint16_t>(cfg_.http_timeout_ms));
  http.setFollowRedirects(HTTPC_FORCE_FOLLOW_REDIRECTS);
  http.setRedirectLimit(10);

  NetworkClient plain;
  NetworkClientSecure secure;
  secure.setInsecure();  // bring-up only; swap to setCACert/ bundle later
  bool ok;
  if (https) {
    ok = http.begin(secure, target);
  } else {
    ok = http.begin(plain, target);
  }
  if (!ok) return nullptr;

  const int code = http.GET();
  if (code != HTTP_CODE_OK) {
    http.end();
    return nullptr;
  }

  const int content_length = http.getSize();
  if (content_length <= 0) {
    http.end();
    return nullptr;
  }
  const size_t want = static_cast<size_t>(content_length);

  uint8_t* buf = psram::alloc(want);
  if (buf == nullptr) {
    http.end();
    return nullptr;
  }

  NetworkClient* stream = http.getStreamPtr();
  stream->setTimeout(cfg_.http_timeout_ms);
  size_t got = 0;
  while (got < want) {
    const size_t avail = static_cast<size_t>(stream->available());
    if (avail > 0) {
      size_t chunk = avail;
      if (chunk > want - got) chunk = want - got;
      const size_t rd = stream->readBytes(buf + got, chunk);
      if (rd == 0) break;
      got += rd;
    } else if (!stream->connected()) {
      break;  // server closed and nothing buffered
    } else {
      delay(1);
    }
  }
  http.end();

  if (got != want) {
    psram::free(buf);
    return nullptr;
  }
  if (size != nullptr) *size = got;
  return buf;
}

uint8_t* ModelTransport::fetchUsb(size_t* size) {
#if WHISTLE_USB_MSC
  if (size != nullptr) *size = 0;

  // USBHost.begin() starts a background TinyUSB worker, so the blocking file
  // read below completes without loop() calling USBHost.task().
  Serial.println("[USB] probing for a mass-storage drive...");
  if (!USBHost.begin()) {
    Serial.println("[USB] USBHost.begin() failed");
    return nullptr;
  }

  const uint32_t t0 = millis();
  while (!USBHostMSC.mounted()) {
    USBHost.task();
    if ((millis() - t0) > kUsbTimeoutMs) {
      Serial.printf("[USB] no drive mounted within %u ms\n", (unsigned)kUsbTimeoutMs);
      return nullptr;
    }
    delay(10);
  }
  Serial.printf("[USB] drive mounted (block %u B, %u blocks)\n",
                (unsigned)USBHostMSC.blockSize(), (unsigned)USBHostMSC.blockCount());

  if (!s_usbFs.begin(kUsbMountpoint, 5, false)) {
    Serial.println("[USB] FAT mount failed (not FAT32?)");
    return nullptr;
  }

  File f = s_usbFs.open(kUsbModelPath);
  if (!f) {
    s_usbFs.end();
    Serial.printf("[USB] %s not found\n", kUsbModelPath);
    return nullptr;
  }
  const size_t total = static_cast<size_t>(f.size());
  Serial.printf("[USB] reading %s (%u bytes)\n", kUsbModelPath, (unsigned)total);
  if (total == 0) {
    f.close();
    s_usbFs.end();
    return nullptr;
  }
  uint8_t* buf = psram::alloc(total);
  if (buf == nullptr) {
    f.close();
    s_usbFs.end();
    Serial.println("[USB] PSRAM allocation failed");
    return nullptr;
  }
  size_t got = 0;
  while (got < total) {
    const size_t n = f.read(buf + got, total - got);
    if (n == 0) break;
    got += n;
  }
  f.close();
  s_usbFs.end();
  if (got != total) {
    psram::free(buf);
    Serial.println("[USB] short read");
    return nullptr;
  }
  if (size != nullptr) *size = got;
  Serial.println("[USB] model loaded from drive");
  return buf;
#else
  (void)size;
  return nullptr;
#endif
}

uint8_t* ModelTransport::load(size_t* size, const ModelTransportConfig& cfg,
                              const char* url) {
  uint8_t* m = fetchUsb(size);
  if (m != nullptr) return m;
  if (!begin(cfg)) return nullptr;
  return fetch(size, url);
}

void ModelTransport::end() {
  if (began_) {
    ETH.end();
    began_ = false;
    got_ip_ = false;
  }
}

}  // namespace whistle