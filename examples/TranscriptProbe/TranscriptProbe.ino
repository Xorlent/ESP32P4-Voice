#include <Arduino.h>
#include <esp_heap_caps.h>
#include <esp_task_wdt.h>
#include <esp_timer.h>

#include <cstring>
#include <string>
#include <vector>

#include <ModelTransport.h>
#include <RowParallel.h>
#include <WhistleEngine.h>
#include <psram.h>

#include "jfk_pcm.h"

using namespace whistle;

// The per-group fwht/buck tags are thousands of lines per run and their serial
// writes are part of the measured time; keep only the coarse stage tags.
#ifndef TRANSCRIPT_TRACE_GROUPS
#define TRANSCRIPT_TRACE_GROUPS 0
#endif

// 1 = install the two-core CQ GEMV row hook, 0 = single core (isolation build).
#ifndef PROBE_DUAL_CORE
#define PROBE_DUAL_CORE 1
#endif

// 1 = incremental (KV-cached) decoder, 0 = batched full-prefix (oracle build).
#ifndef PROBE_INCREMENTAL
#define PROBE_INCREMENTAL 1
#endif

// 1 = also re-run every incremental step on the batched path and report the
// largest logit difference + argmax mismatches (slow, diagnostic only).
#ifndef PROBE_VERIFY_INC
#define PROBE_VERIFY_INC 0
#endif

#define LOG(fmt, ...)                    \
  do {                                   \
    Serial.printf(fmt, ##__VA_ARGS__);   \
    Serial0.printf(fmt, ##__VA_ARGS__);  \
  } while (0)

static ModelTransport transport;
static uint8_t* model = nullptr;
static size_t model_size = 0;
static int64_t prev_us = 0;

// Internal SRAM here is 768 KB and uncached, and the default allocator serves
// requests <= 16 KB from it, so where a buffer ends up depends on how full
// internal SRAM was at that moment. Stage timings swung 2x between builds that
// differed only by one 786 KB allocation; print the state so the next swing can
// be tied to it rather than guessed at.
static void logHeap(const char* when) {
  LOG("heap %-6s int %u/%u KB, psram %u/%u KB\n", when,
      (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
      (unsigned)(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) / 1024),
      (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024),
      (unsigned)(heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) / 1024));
}

static void progressCb(void*, const char* tag) {
#if !TRANSCRIPT_TRACE_GROUPS
  if (std::strcmp(tag, "fwht") == 0 || std::strcmp(tag, "buck") == 0) return;
  // Per-block decoder tags (pre/eng/attn/hada/lN/head) are ~20 lines per step;
  // at 115200 baud that is measurable. The step totals are enough.
  if (std::strcmp(tag, "pre") == 0 || std::strcmp(tag, "eng") == 0 ||
      std::strcmp(tag, "attn") == 0 || std::strcmp(tag, "hada") == 0 ||
      std::strcmp(tag, "head") == 0 || tag[0] == 'l')
    return;
#endif
  const int64_t now = esp_timer_get_time();
  LOG("  [%-9s] +%lld ms\n", tag, (long long)((now - prev_us) / 1000));
  prev_us = now;
}

static void run(void*) {
  LOG("\n=== Whistle transcript probe (3 s JFK) ===\n");
  LOG("PSRAM total %u MB, largest free %u KB\n",
      (unsigned)(psram::totalSize() / (1024u * 1024u)),
      (unsigned)(psram::largestFree() / 1024u));

  const uint32_t td = millis();
  model = transport.load(&model_size);
  if (model == nullptr) {
    LOG("FAIL: model load\n");
    vTaskDelete(nullptr);
    return;
  }
  LOG("source: %s\n", transport.connected() ? transport.ipString() : "USB drive");
  LOG("loaded %u bytes in %u ms, free %u KB\n", (unsigned)model_size,
      (unsigned)(millis() - td), (unsigned)(psram::largestFree() / 1024u));

  WhistleEngine engine;
  if (!engine.load(model, model_size)) {
    LOG("FAIL: engine load\n");
    vTaskDelete(nullptr);
    return;
  }
  LOG("engine ready, audio=%d, decoder_layers=%u, free %u KB\n",
      engine.audioReady() ? 1 : 0, (unsigned)engine.decoder().weights().geometry().num_layers,
      (unsigned)(psram::largestFree() / 1024u));
  logHeap("ready");
  engine.setBounded(true);
  engine.setIncremental(PROBE_INCREMENTAL != 0);
  engine.setVerifyIncremental(PROBE_VERIFY_INC != 0);
  engine.setProgress(progressCb, nullptr);

  std::vector<float> pcm(kJfkSamples);
  for (unsigned i = 0; i < kJfkSamples; ++i)
    pcm[i] = static_cast<float>(kJfkPcm[i]) / 32768.0f;

  esp_task_wdt_deinit();

  // Warm-up: discard the first transcribe so the measured runs are not the
  // first-ever ones. It is NOT slower for clock/PLL reasons - CONFIG_PM_ENABLE is
  // off in this core, so the P4 runs at 400 MHz from reset - and the lazy
  // buffers are ~1.8 MB of first touch (a few ms). What actually differs is the
  // heap layout: today the first run measures ~25% FASTER than later ones
  // (develop.md 64), so discarding it keeps the A/B honest rather than
  // conservative.
  {
    engine.setDecoderDepth(0);
    engine.setMaxNewTokens(1);
    std::string warm;
    prev_us = esp_timer_get_time();
    const uint32_t t0 = millis();
    engine.transcribe(pcm.data(), pcm.size(), &warm);
    LOG("warm-up %u ms (discarded)\n", (unsigned)(millis() - t0));
    logHeap("warmup");
  }

  // Depth 0 is the full 8-block stack; depth 4 is the ladder rung (blocks
  // 0,3,5,7 - the "l<N>" tags in the trace show which blocks actually ran).
  // Add 2/3 here for a full accuracy/latency sweep.
  const uint32_t depths[] = {0, 4};
  const uint32_t kMaxTokens = 16;
  bool all_ok = true;
  for (uint32_t depth : depths) {
    engine.setDecoderDepth(depth);
    engine.setMaxNewTokens(kMaxTokens);
    LOG("\n--- depth=%u max_tokens=%u ---\n", (unsigned)depth, (unsigned)kMaxTokens);
    if (depth != 0) LOG("(ladder rung: blocks 0,3,5,7)\n");

    prev_us = esp_timer_get_time();
    const uint32_t t0 = millis();
    std::string text;
    const bool ok = engine.transcribe(pcm.data(), pcm.size(), &text);
    const uint32_t dt = millis() - t0;
    LOG("depth=%u ok=%d time=%u ms\n", (unsigned)depth, ok ? 1 : 0, (unsigned)dt);
    logHeap("run");
    LOG("transcript: \"%s\"\n", text.c_str());
    if (PROBE_VERIFY_INC)
      LOG("inc-vs-batched: max|dlogit| %.6f, argmax mismatches %u\n",
          engine.verifyDiff(), (unsigned)engine.verifyMismatches());
    if (!ok || text.empty()) all_ok = false;
  }

  LOG("\nexpected  : \"And so, my fellow Americans!\"\n");
  LOG("free after: %u KB\n", (unsigned)(psram::largestFree() / 1024u));
  LOG("%s\n", all_ok ? "TRANSCRIPT PROBE DONE" : "TRANSCRIPT PROBE FAIL");
  vTaskDelete(nullptr);
}

void setup() {
  Serial.begin(115200);
  Serial0.begin(115200, SERIAL_8N1, 38, 37);
  delay(1200);
  const bool rows2 = PROBE_DUAL_CORE ? enableDualCoreRows() : false;
  LOG("dual-core rows: %s\n", rows2 ? "on" : "off");
  xTaskCreatePinnedToCore(run, "transcript", 32768, nullptr, 5, nullptr, 0);
}

void loop() {}
