#include "psram.h"

#include "esp_heap_caps.h"
#include "esp_memory_utils.h"
#include "esp_psram.h"

namespace whistle {
namespace psram {

bool isAvailable() {
  return esp_psram_is_initialized();
}

uint8_t* alloc(size_t bytes) {
  return static_cast<uint8_t*>(heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM));
}

void free(uint8_t* p) {
  if (p != nullptr) {
    heap_caps_free(p);
  }
}

bool inExternal(const void* p) {
  return p != nullptr && esp_ptr_external_ram(p);
}

size_t totalSize() {
  return esp_psram_is_initialized() ? static_cast<size_t>(esp_psram_get_size()) : 0;
}

size_t largestFree() {
  return static_cast<size_t>(heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
}

}  // namespace psram
}  // namespace whistle
