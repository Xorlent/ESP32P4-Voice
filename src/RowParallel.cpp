#include "RowParallel.h"

#if defined(ARDUINO)

#include <Arduino.h>
#include <esp_cache.h>
#include <esp_heap_caps.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>

#include <cstring>

#include "ModelWeights.h"

namespace whistle {
namespace {

// The two P4 cores have private L1 data caches, and esp_cache_msync rejects the
// unaligned flag for the memory-to-cache direction. So: the worker writes only
// to a dedicated staging buffer (never to the caller's output), ranges are
// cache-line aligned, and the caller merges after the join. That needs exactly
// one writeback + one invalidate of a whole dedicated range per group, with no
// sub-range boundaries and no false sharing with the caller's own half.
constexpr size_t kLine = 64;
constexpr size_t kStagingBytes = 160 * 1024;

float* g_staging = nullptr;
size_t g_stagingFloats = 0;

ModelWeights::RowFn g_fn = nullptr;
void* g_ctx = nullptr;
ModelWeights::RowsSync g_sync;
uint32_t g_r0 = 0;
uint32_t g_r1 = 0;
size_t g_outStride = 0;

TaskHandle_t g_worker = nullptr;
SemaphoreHandle_t g_go = nullptr;
SemaphoreHandle_t g_done = nullptr;

void syncRange(const void* p, size_t n, int flags) {
  if (p == nullptr || n == 0) return;
  const uintptr_t start = reinterpret_cast<uintptr_t>(p);
  const uintptr_t lo = start & ~(kLine - 1);
  const uintptr_t hi = (start + n + kLine - 1) & ~(kLine - 1);
  esp_cache_msync(reinterpret_cast<void*>(lo), static_cast<size_t>(hi - lo),
                  flags);
}

constexpr int kWriteBack =
    ESP_CACHE_MSYNC_FLAG_DIR_C2M | ESP_CACHE_MSYNC_FLAG_INVALIDATE;
constexpr int kInvalidate = ESP_CACHE_MSYNC_FLAG_DIR_M2C;

void workerTask(void*) {
  for (;;) {
    xSemaphoreTake(g_go, portMAX_DELAY);
    // Pick up the activations the caller published before dispatching.
    syncRange(g_sync.ro, g_sync.roBytes, kInvalidate);
    g_fn(g_ctx, g_r0, g_r1, g_staging, g_outStride);
    syncRange(g_staging, static_cast<size_t>(g_sync.rwRows) * g_outStride *
                             sizeof(float),
              kWriteBack);
    xSemaphoreGive(g_done);
  }
}

void rowsHook(ModelWeights::RowFn fn, void* ctx, uint32_t nrows,
              const ModelWeights::RowsSync& sync) {
  float* dst = static_cast<float*>(sync.rw);
  const uint32_t half = nrows / 2;
  const uint32_t wrows = nrows - half;
  if (g_worker == nullptr || g_staging == nullptr || nrows < 2 ||
      static_cast<size_t>(sync.rwRows) * wrows > g_stagingFloats) {
    fn(ctx, 0, nrows, dst, sync.rwStride);
    return;
  }

  g_fn = fn;
  g_ctx = ctx;
  g_sync = sync;
  g_r0 = half;
  g_r1 = nrows;
  g_outStride = wrows;
  const size_t wbytes =
      static_cast<size_t>(sync.rwRows) * wrows * sizeof(float);
  std::memset(g_staging, 0, wbytes);
  // Publish the transformed activations (writeback, whole dedicated range).
  syncRange(sync.ro, sync.roBytes, ESP_CACHE_MSYNC_FLAG_DIR_C2M);
  xSemaphoreGive(g_go);
  fn(ctx, 0, half, dst, sync.rwStride);
  xSemaphoreTake(g_done, portMAX_DELAY);
  syncRange(g_staging, wbytes, kInvalidate);
  // Merge the worker's rows; the caller's own half stayed in its cache intact.
  for (size_t mm = 0; mm < sync.rwRows; ++mm) {
    const float* src = g_staging + mm * wrows;
    float* drow = dst + mm * sync.rwStride + half;
    for (uint32_t i = 0; i < wrows; ++i) drow[i] += src[i];
  }
}

// ---- Frame/chunk parallel (Hadamard MLP) ----
// Independent rows of `rowStride` contiguous floats split across the two cores.
// The caller keeps [0, r0) and a second pinned worker produces [r0, nrows) into
// the same staging buffer, which is then memcpy'd into place (rows are written,
// not accumulated, so the merge is a plain copy rather than the GEMV's +=).
RowChunkFn g_cfn = nullptr;
void* g_cctx = nullptr;
uint32_t g_cr0 = 0, g_cr1 = 0;
size_t g_cstride = 0;
const void* g_cro = nullptr;
size_t g_croBytes = 0;

TaskHandle_t g_cworker = nullptr;
SemaphoreHandle_t g_cgo = nullptr;
SemaphoreHandle_t g_cdone = nullptr;

void chunkWorker(void*) {
  for (;;) {
    xSemaphoreTake(g_cgo, portMAX_DELAY);
    syncRange(g_cro, g_croBytes, kInvalidate);
    g_cfn(g_cctx, g_cr0, g_cr1, g_staging);
    syncRange(g_staging,
              static_cast<size_t>(g_cr1 - g_cr0) * g_cstride * sizeof(float),
              kWriteBack);
    xSemaphoreGive(g_cdone);
  }
}

}  // namespace

bool runRowParallel(RowChunkFn fn, void* ctx, uint32_t nrows, size_t rowStride,
                    float* out, const void* ro, size_t roBytes) {
  const uint32_t r0 = nrows / 2;
  const uint32_t wrows = nrows - r0;
  const size_t need = static_cast<size_t>(wrows) * rowStride;
  if (g_cworker == nullptr || g_staging == nullptr || nrows < 2 ||
      need > g_stagingFloats)
    return false;

  g_cfn = fn;
  g_cctx = ctx;
  g_cr0 = r0;
  g_cr1 = nrows;
  g_cstride = rowStride;
  g_cro = ro;
  g_croBytes = roBytes;

  // Publish the caller's input rows to PSRAM, then run the caller's own half
  // inline while the worker produces the second half.
  syncRange(ro, roBytes, ESP_CACHE_MSYNC_FLAG_DIR_C2M);
  xSemaphoreGive(g_cgo);
  fn(ctx, 0, r0, out);
  xSemaphoreTake(g_cdone, portMAX_DELAY);
  syncRange(g_staging, need * sizeof(float), kInvalidate);
  std::memcpy(out + static_cast<size_t>(r0) * rowStride, g_staging,
              need * sizeof(float));
  return true;
}

bool enableDualCoreRows() {
  if (g_worker != nullptr) return true;
  if (g_staging == nullptr) {
    g_staging = static_cast<float*>(
        heap_caps_aligned_alloc(kLine, kStagingBytes, MALLOC_CAP_SPIRAM));
    if (g_staging == nullptr) return false;
    g_stagingFloats = kStagingBytes / sizeof(float);
  }
  g_go = xSemaphoreCreateBinary();
  g_done = xSemaphoreCreateBinary();
  if (g_go == nullptr || g_done == nullptr) return false;
  // Worker on core 1; the inference task keeps its own core. 4 KB is ample for a
  // row worker (a 256-entry decode scratch plus the PIE dot).
  if (xTaskCreatePinnedToCore(workerTask, "cqrows", 4096, nullptr, 5, &g_worker,
                              1) != pdPASS) {
    g_worker = nullptr;
    return false;
  }
  g_cgo = xSemaphoreCreateBinary();
  g_cdone = xSemaphoreCreateBinary();
  if (g_cgo == nullptr || g_cdone == nullptr) return false;
  if (xTaskCreatePinnedToCore(chunkWorker, "cqchunks", 4096, nullptr, 5,
                              &g_cworker, 1) != pdPASS) {
    g_cworker = nullptr;
    return false;
  }
  ModelWeights::setRowsStaging(g_stagingFloats);
  ModelWeights::setRowsHook(&rowsHook);
  return true;
}

}  // namespace whistle

#else

namespace whistle {
bool enableDualCoreRows() { return false; }
bool runRowParallel(RowChunkFn, void*, uint32_t, size_t, float*, const void*,
                    size_t) {
  return false;
}
}  // namespace whistle

#endif


