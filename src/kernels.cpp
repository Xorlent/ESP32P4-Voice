#include "kernels.h"

#include <cstring>

// Xespv2p2 packed-SIMD (PIE) inline assembly. XACC is the 40-bit accumulator:
//   ESP.ZERO.XACC               clear XACC
//   ESP.VMULAS.S8.XACC qx,qy    XACC += sum of 16 s8 products
//   ESP.VMULAS.S16.XACC qx,qy   XACC += sum of 8 s16 products
//   ESP.VLD.128.IP qu,rs,imm    load 16B, base post-increment by imm
//   ESP.ST.S.XACC.IP rs,imm     store sign-extended XACC; imm=0 -> no post-inc
//   ESP.VADD.S16 / ESP.VSUB.S16 / ESP.VST.128.IP

namespace whistle {
namespace kernels {

// Signed int8 dot (n multiple of 16). Unrolled 8 chunks with loads issued four
// chunks ahead of the VMULAS that consume them, so the serial XACC chain hides
// the load latency; the accumulate order is unchanged, so the sum is exact.
int32_t dot_s8(const int8_t* a, const int8_t* b, size_t n) {
  int64_t result = 0;
  int64_t* out = &result;
  const size_t chunks = n >> 4;
  const size_t octs = chunks >> 3;
  const size_t rem = chunks & 7;
  asm volatile(
    "ESP.ZERO.XACC                  \n\t"
    "mv    t0, %[octs]              \n\t"
    "beqz  t0, 3f                   \n\t"
    "1:                             \n\t"
    "ESP.VLD.128.IP q0, %[a], 16    \n\t"
    "ESP.VLD.128.IP q1, %[b], 16    \n\t"
    "ESP.VLD.128.IP q2, %[a], 16    \n\t"
    "ESP.VLD.128.IP q3, %[b], 16    \n\t"
    "ESP.VMULAS.S8.XACC q0, q1      \n\t"
    "ESP.VLD.128.IP q4, %[a], 16    \n\t"
    "ESP.VLD.128.IP q5, %[b], 16    \n\t"
    "ESP.VMULAS.S8.XACC q2, q3      \n\t"
    "ESP.VLD.128.IP q6, %[a], 16    \n\t"
    "ESP.VLD.128.IP q7, %[b], 16    \n\t"
    "ESP.VMULAS.S8.XACC q4, q5      \n\t"
    "ESP.VLD.128.IP q0, %[a], 16    \n\t"
    "ESP.VLD.128.IP q1, %[b], 16    \n\t"
    "ESP.VMULAS.S8.XACC q6, q7      \n\t"
    "ESP.VLD.128.IP q2, %[a], 16    \n\t"
    "ESP.VLD.128.IP q3, %[b], 16    \n\t"
    "ESP.VMULAS.S8.XACC q0, q1      \n\t"
    "ESP.VLD.128.IP q4, %[a], 16    \n\t"
    "ESP.VLD.128.IP q5, %[b], 16    \n\t"
    "ESP.VMULAS.S8.XACC q2, q3      \n\t"
    "ESP.VLD.128.IP q6, %[a], 16    \n\t"
    "ESP.VLD.128.IP q7, %[b], 16    \n\t"
    "ESP.VMULAS.S8.XACC q4, q5      \n\t"
    "ESP.VMULAS.S8.XACC q6, q7      \n\t"
    "addi  t0, t0, -1               \n\t"
    "bnez  t0, 1b                   \n\t"
    "3:                             \n\t"
    "mv    t0, %[rem]               \n\t"
    "2:                             \n\t"
    "beqz  t0, 4f                   \n\t"
    "ESP.VLD.128.IP q0, %[a], 16    \n\t"
    "ESP.VLD.128.IP q1, %[b], 16    \n\t"
    "ESP.VMULAS.S8.XACC q0, q1      \n\t"
    "addi  t0, t0, -1               \n\t"
    "bnez  t0, 2b                   \n\t"
    "4:                             \n\t"
    "ESP.ST.S.XACC.IP %[out], 0     \n\t"
    : [a] "+r"(a), [b] "+r"(b)
    : [octs] "r"(octs), [rem] "r"(rem), [out] "r"(out)
    : "t0", "memory");
  return static_cast<int32_t>(result);
}

// Signed 16-bit dot (n multiple of 8). Unrolled 4 chunks so the 8 loads issue
// ahead of the 4 serialized VMULAS.S16.XACC, hiding load latency on the short
// 128-element dots that dominate the CQ GEMV.
int64_t dot_s16(const int16_t* a, const int16_t* b, size_t n) {
  int64_t result = 0;
  int64_t* out = &result;
  const size_t chunks = n >> 3;
  const size_t quads = chunks >> 2;
  const size_t rem = chunks & 3;
  asm volatile(
    "ESP.ZERO.XACC                  \n\t"
    "mv    t0, %[quads]             \n\t"
    "beqz  t0, 3f                   \n\t"
    "1:                             \n\t"
    "ESP.VLD.128.IP q0, %[a], 16    \n\t"
    "ESP.VLD.128.IP q1, %[b], 16    \n\t"
    "ESP.VLD.128.IP q2, %[a], 16    \n\t"
    "ESP.VLD.128.IP q3, %[b], 16    \n\t"
    "ESP.VLD.128.IP q4, %[a], 16    \n\t"
    "ESP.VLD.128.IP q5, %[b], 16    \n\t"
    "ESP.VLD.128.IP q6, %[a], 16    \n\t"
    "ESP.VLD.128.IP q7, %[b], 16    \n\t"
    "ESP.VMULAS.S16.XACC q0, q1     \n\t"
    "ESP.VMULAS.S16.XACC q2, q3     \n\t"
    "ESP.VMULAS.S16.XACC q4, q5     \n\t"
    "ESP.VMULAS.S16.XACC q6, q7     \n\t"
    "addi  t0, t0, -1               \n\t"
    "bnez  t0, 1b                   \n\t"
    "3:                             \n\t"
    "mv    t0, %[rem]               \n\t"
    "2:                             \n\t"
    "beqz  t0, 4f                   \n\t"
    "ESP.VLD.128.IP q0, %[a], 16    \n\t"
    "ESP.VLD.128.IP q1, %[b], 16    \n\t"
    "ESP.VMULAS.S16.XACC q0, q1     \n\t"
    "addi  t0, t0, -1               \n\t"
    "bnez  t0, 2b                   \n\t"
    "4:                             \n\t"
    "ESP.ST.S.XACC.IP %[out], 0     \n\t"
    : [a] "+r"(a), [b] "+r"(b)
    : [quads] "r"(quads), [rem] "r"(rem), [out] "r"(out)
    : "t0", "memory");
  return result;
}

// In-place int16 FWHT, vectorized for butterfly lengths >= 8 (the 128-group
// Whistle case covers 4 of the 7 stages). Saturating two-vector add/sub is
// exact here because the FWHT of int8 activations stays within +-16384. The
// VLD/VST base register must be in the compressible set, so the pointers are
// moved into a4/a5 inside the loop rather than trusting the C allocation.
void fwht_s16(int16_t* x, size_t n) {
  for (size_t h = 1; h < n; h <<= 1) {
    if (h >= 8) {
      const size_t vecs = h >> 3;
      for (size_t i = 0; i < n; i += (h << 1)) {
        int16_t* pa = x + i;
        int16_t* pb = x + i + h;
        asm volatile(
          "mv   a4, %[pa]            \n\t"
          "mv   a5, %[pb]            \n\t"
          "mv   a3, %[vecs]          \n\t"
          "1:                        \n\t"
          "ESP.VLD.128.IP q0, a4, 16 \n\t"
          "ESP.VLD.128.IP q1, a5, 16 \n\t"
          "ESP.VADD.S16 q2, q0, q1   \n\t"
          "ESP.VSUB.S16 q3, q0, q1   \n\t"
          "addi a4, a4, -16          \n\t"
          "addi a5, a5, -16          \n\t"
          "ESP.VST.128.IP q2, a4, 16 \n\t"
          "ESP.VST.128.IP q3, a5, 16 \n\t"
          "addi a3, a3, -1           \n\t"
          "bnez a3, 1b               \n\t"
          :
          : [pa] "r"(pa), [pb] "r"(pb), [vecs] "r"(vecs)
          : "a3", "a4", "a5", "memory");
      }
    } else {
      for (size_t i = 0; i < n; i += (h << 1)) {
        for (size_t j = i; j < i + h; ++j) {
          const int a = x[j];
          const int b = x[j + h];
          x[j] = static_cast<int16_t>(a + b);
          x[j + h] = static_cast<int16_t>(a - b);
        }
      }
    }
  }
}

// One 128-element s8 group per iteration (8x VMULAS.S8), then that group's sum
// is stored and XACC cleared.
void dot_groups_s8(const int8_t* a, const int8_t* b, int64_t* out, size_t groups) {
  asm volatile(
    "ESP.ZERO.XACC                  \n\t"
    "mv    t0, %[n]                 \n\t"
    "beqz  t0, 2f                   \n\t"
    "1:                             \n\t"
    "ESP.VLD.128.IP q0, %[a], 16    \n\t"
    "ESP.VLD.128.IP q1, %[b], 16    \n\t"
    "ESP.VLD.128.IP q2, %[a], 16    \n\t"
    "ESP.VLD.128.IP q3, %[b], 16    \n\t"
    "ESP.VLD.128.IP q4, %[a], 16    \n\t"
    "ESP.VLD.128.IP q5, %[b], 16    \n\t"
    "ESP.VLD.128.IP q6, %[a], 16    \n\t"
    "ESP.VLD.128.IP q7, %[b], 16    \n\t"
    "ESP.VMULAS.S8.XACC q0, q1      \n\t"
    "ESP.VMULAS.S8.XACC q2, q3      \n\t"
    "ESP.VMULAS.S8.XACC q4, q5      \n\t"
    "ESP.VMULAS.S8.XACC q6, q7      \n\t"
    "ESP.VLD.128.IP q0, %[a], 16    \n\t"
    "ESP.VLD.128.IP q1, %[b], 16    \n\t"
    "ESP.VLD.128.IP q2, %[a], 16    \n\t"
    "ESP.VLD.128.IP q3, %[b], 16    \n\t"
    "ESP.VLD.128.IP q4, %[a], 16    \n\t"
    "ESP.VLD.128.IP q5, %[b], 16    \n\t"
    "ESP.VLD.128.IP q6, %[a], 16    \n\t"
    "ESP.VLD.128.IP q7, %[b], 16    \n\t"
    "ESP.VMULAS.S8.XACC q0, q1      \n\t"
    "ESP.VMULAS.S8.XACC q2, q3      \n\t"
    "ESP.VMULAS.S8.XACC q4, q5      \n\t"
    "ESP.VMULAS.S8.XACC q6, q7      \n\t"
    "ESP.ST.S.XACC.IP %[o], 0       \n\t"
    "addi  %[o], %[o], 8            \n\t"
    "ESP.ZERO.XACC                  \n\t"
    "addi  t0, t0, -1               \n\t"
    "bnez  t0, 1b                   \n\t"
    "2:                             \n\t"
    : [a] "+r"(a), [b] "+r"(b), [o] "+r"(out)
    : [n] "r"(groups)
    : "t0", "memory");
}

}  // namespace kernels
}  // namespace whistle
