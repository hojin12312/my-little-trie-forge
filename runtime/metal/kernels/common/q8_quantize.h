#pragma once

#include "metal/abi/KernelABI.h"

// Group-64 symmetric int8 activation quantization, shared by the standalone
// quantize kernels, the fused RMS-norm kernels and the A8 up_silu output
// side channel. The calling lane holds the group's elements k = lane and
// k = lane + 32; the pair is quantized against the simdgroup-wide max.
// Returns {sx, sx * qxsum}; sx = 1 on an all-zero group so the terms stay
// finite, and qxsum is exact in int32 (|q| <= 127, 64 elements).
inline float2 q8_quantize_pair(float a, float b, thread int &qa,
                               thread int &qb) {
  const float amax = simd_max(max(fabs(a), fabs(b)));
  const float sx = amax > 0.0f ? amax / 127.0f : 1.0f;
  qa = clamp(int(rint(a / sx)), -127, 127);
  qb = clamp(int(rint(b / sx)), -127, 127);
  const int qsum = simd_sum(qa + qb);
  return float2(sx, sx * float(qsum));
}
