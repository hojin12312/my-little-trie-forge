#pragma once

#include "metal/abi/KernelABI.h"

// Inverse RMS of one row, reduced across the 256-thread group and returned to
// every thread.
inline float rms_inverse(device const bfloat *row_input, uint width,
                         threadgroup float *reductions, uint thread_index,
                         uint lane, uint simd_group) {
  float sum = 0.0f;
  for (uint column = thread_index; column < width; column += 256) {
    float value = float(row_input[column]);
    sum += value * value;
  }
  sum = simd_sum(sum);
  if (lane == 0)
    reductions[simd_group] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i)
      total += reductions[i];
    reductions[0] = rsqrt(total / width + 1e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return reductions[0];
}
