#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

// Fast (~5x) expf for the Whistle hot path (silu/softmax/Sinkhorn/cond): the
// ESP32-P4 has no libm expf acceleration. Set WHISTLE_FAST_EXP 0 for libm.

#ifndef WHISTLE_FAST_EXP
#define WHISTLE_FAST_EXP 1
#endif

namespace whistle {
namespace fastmath {

// exp(x), range-reduced to 2^k * 2^f with f in [-0.5, 0.5]: 2^f is a degree-5
// minimax polynomial and 2^k is assembled directly into the exponent field.
// Accurate to ~1e-7 relative over the useful range (matches the independent
// ESP32-S3 whistle port's nd_expf).
inline float expf(float x) {
#if WHISTLE_FAST_EXP
  if (x > 88.0f) return 3.4028235e38f;
  if (x < -88.0f) return 0.0f;

  const float z = x * 1.44269504f;  // x / ln 2
  const int k = static_cast<int>(z + (z >= 0.0f ? 0.5f : -0.5f));
  const float f = z - static_cast<float>(k);

  float p = 0.0013333f;
  p = p * f + 0.0096181f;
  p = p * f + 0.0555041f;
  p = p * f + 0.2402265f;
  p = p * f + 0.6931472f;
  p = p * f + 1.0f;

  const uint32_t bits = static_cast<uint32_t>(k + 127) << 23;  // 2^k
  float scale;
  std::memcpy(&scale, &bits, sizeof(scale));
  return p * scale;
#else
  return std::expf(x);
#endif
}

inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

inline float silu(float x) { return x * sigmoid(x); }

}  // namespace fastmath
}  // namespace whistle
