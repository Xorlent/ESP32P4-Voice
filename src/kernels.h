#pragma once

#include <cstddef>
#include <cstdint>

// Xespv2p2 (ESP32-P4 "PIE") packed-SIMD kernels; the PIE unit is always present
// on the P4, so these are inline assembly unconditionally.
//
// dot_s8 is defined here rather than in kernels.cpp so callers can inline it: the
// CQ GEMV calls it hundreds of thousands of times per layer. Worth ~0.5%.

namespace whistle {
namespace kernels {

// Signed 16-bit in-place Walsh-Hadamard transform; `n` a power of two.
void fwht_s16(int16_t* x, size_t n);

// Signed int8 dot over `n` (multiple of 16) elements, full width, as int32.
// Unrolled 8 chunks with loads issued four chunks ahead of the VMULAS that
// consume them, so the serial XACC chain hides the load latency; the accumulate
// order is unchanged, so the sum is exact.
__attribute__((always_inline)) static inline int32_t dot_s8(const int8_t* a,
                                                            const int8_t* b,
                                                            size_t n) {
  int64_t result = 0;
  int64_t* out = &result;
  const size_t chunks = n >> 4;
  const size_t octs = chunks >> 3;
  const size_t rem = chunks & 7;
  // PIE VLD/VST bases must be in x8..x15, so park the pointers in a3/a4/a5; the
  // allocator picks a7/t1 for an inlined asm, and that does not assemble.
  asm volatile(
    "mv    a4, %[a]                 \n\t"
    "mv    a5, %[b]                 \n\t"
    "mv    a3, %[out]               \n\t"
    "ESP.ZERO.XACC                  \n\t"
    "mv    t0, %[octs]              \n\t"
    "beqz  t0, 3f                   \n\t"
    "1:                             \n\t"
    "ESP.VLD.128.IP q0, a4, 16      \n\t"
    "ESP.VLD.128.IP q1, a5, 16      \n\t"
    "ESP.VLD.128.IP q2, a4, 16      \n\t"
    "ESP.VLD.128.IP q3, a5, 16      \n\t"
    "ESP.VMULAS.S8.XACC q0, q1      \n\t"
    "ESP.VLD.128.IP q4, a4, 16      \n\t"
    "ESP.VLD.128.IP q5, a5, 16      \n\t"
    "ESP.VMULAS.S8.XACC q2, q3      \n\t"
    "ESP.VLD.128.IP q6, a4, 16      \n\t"
    "ESP.VLD.128.IP q7, a5, 16      \n\t"
    "ESP.VMULAS.S8.XACC q4, q5      \n\t"
    "ESP.VLD.128.IP q0, a4, 16      \n\t"
    "ESP.VLD.128.IP q1, a5, 16      \n\t"
    "ESP.VMULAS.S8.XACC q6, q7      \n\t"
    "ESP.VLD.128.IP q2, a4, 16      \n\t"
    "ESP.VLD.128.IP q3, a5, 16      \n\t"
    "ESP.VMULAS.S8.XACC q0, q1      \n\t"
    "ESP.VLD.128.IP q4, a4, 16      \n\t"
    "ESP.VLD.128.IP q5, a5, 16      \n\t"
    "ESP.VMULAS.S8.XACC q2, q3      \n\t"
    "ESP.VLD.128.IP q6, a4, 16      \n\t"
    "ESP.VLD.128.IP q7, a5, 16      \n\t"
    "ESP.VMULAS.S8.XACC q4, q5      \n\t"
    "ESP.VMULAS.S8.XACC q6, q7      \n\t"
    "addi  t0, t0, -1               \n\t"
    "bnez  t0, 1b                   \n\t"
    "3:                             \n\t"
    "mv    t0, %[rem]               \n\t"
    "2:                             \n\t"
    "beqz  t0, 4f                   \n\t"
    "ESP.VLD.128.IP q0, a4, 16      \n\t"
    "ESP.VLD.128.IP q1, a5, 16      \n\t"
    "ESP.VMULAS.S8.XACC q0, q1      \n\t"
    "addi  t0, t0, -1               \n\t"
    "bnez  t0, 2b                   \n\t"
    "4:                             \n\t"
    "ESP.ST.S.XACC.IP a3, 0         \n\t"
    :
    : [a] "r"(a), [b] "r"(b), [octs] "r"(octs), [rem] "r"(rem), [out] "r"(out)
    : "t0", "a3", "a4", "a5", "memory");
  return static_cast<int32_t>(result);
}

// Signed 16-bit dot over `n` (multiple of 8) elements, as int64.
int64_t dot_s16(const int16_t* a, const int16_t* b, size_t n);

// One 128-element s8 group per iteration (8x VMULAS.S8), storing each group's
// sum, so a caller covers a whole weight row in one call instead of one per
// group (the M=1 decoder GEMVs are dominated by per-dot call overhead).
void dot_groups_s8(const int8_t* a, const int8_t* b, int64_t* out, size_t groups);

}  // namespace kernels
}  // namespace whistle
