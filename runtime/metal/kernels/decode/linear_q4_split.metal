#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/q4_mpp_tiles.h"

// Split-K decode projections for one lane (rows == 8). A threadgroup holds
// Parts partitions of Simdgroups simdgroups each and launches with
// Parts * Simdgroups * 32 threads; the partitions stream equal K ranges of one
// 8 x TileN tile into threadgroup partials (q4_mpp_tile_split), then the whole
// threadgroup reduces the partials and applies the epilogue. A projection
// with fewer 256-wide tiles than the GPU has cores cannot fill the cores with
// the sequential kernels; splitting K multiplies the simdgroups per tile
// instead of narrowing the tile further. No device scratch, extra dispatch or
// weight-layout change. Requires input_size % 1024 == 0 (four 256-input
// ranges) and, as dispatched by ops::Q4Linear, one threadgroup per tile.
//
// The buffer contracts equal the decode kernels in linear_q4.metal: Affine
// (input, weights, scales, biases, output, params), Residual (..., residual,
// output, params) and GateUp (gate stream, output, up stream, params).
template <ushort TileN, ushort Simdgroups, bool Residual, bool GateUp = false>
inline void q4_split(device bfloat *input, device uchar *weights,
                     device bfloat *scales, device bfloat *biases,
                     device bfloat *residual, device bfloat *output,
                     device uchar *upWeights, device bfloat *upScales,
                     device bfloat *upBiases, constant Q4Params &p, uint group,
                     uint lane, uint simd, threadgroup float *sums,
                     threadgroup float *partials) {
  constexpr uint Parts = 4;
  uint partition = simd / Simdgroups;
  for (uint tile = group; tile < p.output_size / TileN;
       tile += p.persistent_groups) {
    q4_mpp_tile_split<TileN, GateUp, 256, true, Simdgroups, Parts>(
        input, weights, scales, biases, partials, upWeights, upScales,
        upBiases, p.input_size, sums + partition * 64, tile * TileN, lane,
        simd % Simdgroups, partition);
    // Same epilogue as q4_mpp_tile: one bf16 rounding of the projection,
    // then the residual add or the SiLU gate, then the output rounding.
    for (uint i = simd * 32 + lane; i < 8 * TileN;
         i += Parts * Simdgroups * 32) {
      float value = 0;
      for (uint part = 0; part < Parts; ++part)
        value += partials[part * 8 * TileN + i];
      uint index = (i / TileN) * p.output_size + tile * TileN + i % TileN;
      value = float(bfloat(value));
      if constexpr (GateUp) {
        float up = 0;
        for (uint part = 0; part < Parts; ++part)
          up += partials[(Parts + part) * 8 * TileN + i];
        value = value / (1.0f + fast::exp2(-1.44269504089f * value)) *
                float(bfloat(up));
      }
      if constexpr (Residual)
        value += float(residual[index]);
      output[index] = bfloat(value);
    }
    // The next tile's partials overwrite this reduction's inputs.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define Q4_SPLIT_AFFINE(Name, TileN, Simdgroups)                               \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *output [[buffer(4)]],                        \
                   constant Q4Params &params [[buffer(5)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float sums[4 * 64], partials[4 * 8 * TileN];                   \
    q4_split<TileN, Simdgroups, false>(input, weights, scales, biases, output, \
                                       output, weights, scales, biases,        \
                                       params, group, lane, simd, sums,        \
                                       partials);                              \
  }

#define Q4_SPLIT_RESIDUAL(Name, TileN, Simdgroups)                             \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *residual [[buffer(4)]],                      \
                   device bfloat *output [[buffer(5)]],                        \
                   constant Q4Params &params [[buffer(6)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint simd [[simdgroup_index_in_threadgroup]]) {             \
    threadgroup float sums[4 * 64], partials[4 * 8 * TileN];                   \
    q4_split<TileN, Simdgroups, true>(input, weights, scales, biases,          \
                                      residual, output, weights, scales,       \
                                      biases, params, group, lane, simd, sums, \
                                      partials);                               \
  }

// Threads per threadgroup = 4 partitions x Simdgroups x 32. Only N32 and
// N64 are instantiated: these are the tiles selected by the split policy.
// A wider tile would require separate register-pressure and timing evidence.
Q4_SPLIT_AFFINE(decode_linear_q4_n32_split4, 32, 1)        // 128 threads
Q4_SPLIT_AFFINE(decode_linear_q4_n64_split4, 64, 2)        // 256 threads
Q4_SPLIT_RESIDUAL(decode_linear_q4_n32_split4_residual, 32, 1)
Q4_SPLIT_RESIDUAL(decode_linear_q4_n64_split4_residual, 64, 2)
#undef Q4_SPLIT_AFFINE
#undef Q4_SPLIT_RESIDUAL

// 128 threads: four single-simdgroup partitions, two weight streams.
kernel void decode_linear_q4_n32_split4_gate_up(
    device bfloat *input [[buffer(0)]], device uchar *weights_0 [[buffer(1)]],
    device bfloat *scales_0 [[buffer(2)]],
    device bfloat *biases_0 [[buffer(3)]], device bfloat *output [[buffer(4)]],
    device uchar *weights_1 [[buffer(5)]], device bfloat *scales_1 [[buffer(6)]],
    device bfloat *biases_1 [[buffer(7)]], constant Q4Params &params [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[4 * 64], partials[2 * 4 * 8 * 32];
  q4_split<32, 1, false, true>(input, weights_0, scales_0, biases_0, output,
                               output, weights_1, scales_1, biases_1, params,
                               group, lane, simd, sums, partials);
}
