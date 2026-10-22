#pragma once

#include <cstddef>
#include <cstdint>

// Xespv2p2 (ESP32-P4 "PIE") packed-SIMD kernels; the PIE unit is always present
// on the P4, so these are inline assembly unconditionally.

namespace whistle {
namespace kernels {

// Signed 16-bit in-place Walsh-Hadamard transform; `n` a power of two.
void fwht_s16(int16_t* x, size_t n);

// Signed int8 dot over `n` (multiple of 16) elements, full width, as int32.
int32_t dot_s8(const int8_t* a, const int8_t* b, size_t n);

// Signed 16-bit dot over `n` (multiple of 8) elements, as int64.
int64_t dot_s16(const int16_t* a, const int16_t* b, size_t n);

// One 128-element s8 group per iteration (8x VMULAS.S8), storing each group's
// sum, so a caller covers a whole weight row in one call instead of one per
// group (the M=1 decoder GEMVs are dominated by per-dot call overhead).
void dot_groups_s8(const int8_t* a, const int8_t* b, int64_t* out, size_t groups);

}  // namespace kernels
}  // namespace whistle
