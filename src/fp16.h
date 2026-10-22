#pragma once

#include <cstdint>
#include <cstring>

// Portable IEEE 754 half <-> float conversion. The .cact stores norms and
// elementwise parameters as fp16, read as float32 here.

namespace whistle {

inline float fp16ToF32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t man = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      // subnormal: value = man * 2^-24. Normalize until bit 10 (0x400) is set.
      uint32_t m = man;
      int k = 0;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        ++k;
      }
      // value = (1 + frac/1024) * 2^(-14-k); float exp field = 113 - k.
      const uint32_t frac = (m & 0x3FFu) << 13;
      bits = sign | (static_cast<uint32_t>(113 - k) << 23) | frac;
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (man << 13);  // inf / nan
  } else {
    // value = (1 + man/1024) * 2^(exp - 15); float exp field = exp + 112.
    bits = sign | ((exp + 112u) << 23) | (man << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

}  // namespace whistle