#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/rms_inverse.h"
#include "metal/kernels/common/q8_quantize.h"

kernel void
prefill_norm_rms_sums32(device const bfloat *input [[buffer(0)]],
                           device const bfloat *weight [[buffer(1)]],
                           device bfloat *output [[buffer(2)]],
                           device float *sums [[buffer(3)]],
                           constant uint &width [[buffer(4)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint thread_index [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint TileM = 32;
  threadgroup float reductions[8];
  const float inverse_rms = rms_inverse(input + row * width, width, reductions,
                                        thread_index, lane, simd_group);
  const uint quant_groups = width / 64;
  const uint row_tile = row / TileM;
  const uint row_in_tile = row % TileM;
  for (uint quant_group = simd_group; quant_group < quant_groups;
       quant_group += 8) {
    uint origin = row * width + quant_group * 64 + lane;
    bfloat first = bfloat(float(input[origin]) * inverse_rms *
                          float(weight[quant_group * 64 + lane]));
    bfloat second = bfloat(float(input[origin + 32]) * inverse_rms *
                           float(weight[quant_group * 64 + lane + 32]));
    output[origin] = first;
    output[origin + 32] = second;
    float group_sum = simd_sum(float(first) + float(second));
    if (lane == 0) {
      sums[(ulong(row_tile) * quant_groups + quant_group) * TileM +
           row_in_tile] = group_sum;
    }
  }
}

// Fused RMS normalization plus dense Q8 A8 operands for packed prefill: the
// ordinary bf16 output stays for non-A8 consumers while the projection reads
// the row-major int8 activations and tiled [group][row] float2 terms.
kernel void
prefill_norm_rms_q8a8(device const bfloat *input [[buffer(0)]],
                      device const bfloat *weight [[buffer(1)]],
                      device bfloat *output [[buffer(2)]],
                      device int8_t *qx [[buffer(3)]],
                      device float2 *ax [[buffer(4)]],
                      constant Q8QuantizeParams &params [[buffer(5)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint thread_index [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint TileM = 32;
  const uint width = params.input_size;
  const uint quant_groups = width / 64;
  const uint row_tile = row / TileM;
  const uint row_in_tile = row % TileM;
  // The projection reads the whole 32-row tile, so the tail past the real row
  // count is zeroed instead of keeping whatever an earlier command left there
  // (the sibling quantize kernel zeroes it the same way).
  if (row >= params.rows) {
    for (uint quant_group = simd_group; quant_group < quant_groups;
         quant_group += 8) {
      const uint origin = row * width + quant_group * 64 + lane;
      output[origin] = bfloat(0.0f);
      output[origin + 32] = bfloat(0.0f);
      qx[origin] = 0;
      qx[origin + 32] = 0;
      if (lane == 0)
        ax[(ulong(row_tile) * quant_groups + quant_group) * TileM +
           row_in_tile] = float2(0.0f);
    }
    return;
  }
  threadgroup float reductions[8];
  const float inverse_rms = rms_inverse(input + row * width, width, reductions,
                                        thread_index, lane, simd_group);
  for (uint quant_group = simd_group; quant_group < quant_groups;
       quant_group += 8) {
    uint origin = row * width + quant_group * 64 + lane;
    bfloat first = bfloat(float(input[origin]) * inverse_rms *
                          float(weight[quant_group * 64 + lane]));
    bfloat second = bfloat(float(input[origin + 32]) * inverse_rms *
                           float(weight[quant_group * 64 + lane + 32]));
    output[origin] = first;
    output[origin + 32] = second;
    int qa, qb;
    const float2 term =
        q8_quantize_pair(float(first), float(second), qa, qb);
    qx[origin] = int8_t(qa);
    qx[origin + 32] = int8_t(qb);
    if (lane == 0) {
      ax[(ulong(row_tile) * quant_groups + quant_group) * TileM +
         row_in_tile] = term;
    }
  }
}
