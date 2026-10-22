#pragma once

#include <cstddef>
#include <cstdint>

// PSRAM helpers. The model weights live in a single PSRAM buffer that is read
// in place (no second copy). We allocate with MALLOC_CAP_SPIRAM so the pointer
// is a cached, memory-mapped external-RAM address.
namespace whistle {
namespace psram {

bool isAvailable();

// Allocates `bytes` in PSRAM, or nullptr on failure. Alignment: 16 bytes.
uint8_t* alloc(size_t bytes);

void free(uint8_t* p);

// True if `p` points into external PSRAM. For p == nullptr, returns false.
bool inExternal(const void* p);

// Total installed PSRAM in bytes (0 if unavailable).
size_t totalSize();

// Largest free PSRAM block in bytes.
size_t largestFree();

}  // namespace psram
}  // namespace whistle
