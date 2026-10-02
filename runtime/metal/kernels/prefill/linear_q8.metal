#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/q8_mpp_tiles.h"

// Dense Q8 prefill projections. A16 kernels take bf16 activations and the
// fp32 group sums produced by prefill_linear_q4_sums32 or the fused norm.
// A8 kernels take the int8 activations and float2 {sx, sx*qxsum} terms that
// prefill_linear_q8a8_quantize or prefill_norm_rms_q8a8 produce.

// Quantize the bf16 input of an A8 projection: row-major int8 qx plus the
// tiled [group][row] float2 terms the epilogue reads. Rows past the real row
// count are zeroed so the padded tail of each 32-row tile is deterministic.
kernel void prefill_linear_q8a8_quantize(
    device const bfloat *input [[buffer(0)]],
    device int8_t *qx [[buffer(1)]],
    device float2 *ax [[buffer(2)]],
    constant Q8QuantizeParams &params [[buffer(3)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint TileM = 32;
  const uint quant_groups = params.input_size / 64;
  input += ulong(tile) * TileM * params.input_size;
  qx += ulong(tile) * TileM * params.input_size;
  ax += ulong(tile) * TileM * quant_groups;
  for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
    for (uint row = simd_group; row < TileM; row += 8) {
      const uint origin =
          row * params.input_size + quant_group * 64 + simd_lane;
      if (tile * TileM + row >= params.rows) {
        qx[origin] = 0;
        qx[origin + 32] = 0;
        if (simd_lane == 0)
          ax[quant_group * TileM + row] = float2(0.0f);
        continue;
      }
      int qa, qb;
      const float2 term =
          q8_quantize_pair(float(input[origin]), float(input[origin + 32]),
                           qa, qb);
      qx[origin] = int8_t(qa);
      qx[origin + 32] = int8_t(qb);
      if (simd_lane == 0)
        ax[quant_group * TileM + row] = term;
    }
  }
}

// Eight simdgroups stage terms in threadgroup memory; four read them from
// device memory directly. Both paths use the same accumulation order.
#define Q8_PREFILL(Name, Param, Activation, InputT, TermT, TileN, Simdgroups)  \
  kernel void Name(                                                          \
      device InputT *input [[buffer(0)]], device int8_t *weights [[buffer(1)]],\
      device Param *scales [[buffer(2)]], device Param *biases [[buffer(3)]],  \
      device bfloat *output [[buffer(4)]],                                   \
      device const TermT *terms [[buffer(5)]],                               \
      constant Q8PrefillParams &params [[buffer(6)]],                        \
      uint2 group [[threadgroup_position_in_grid]],                          \
      uint simd_lane [[thread_index_in_simdgroup]],                          \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                  \
    constexpr ushort TileM = 32;                                             \
    constexpr ushort Batch = Q8Operand<Activation>::TermBatch;               \
    threadgroup TermT input_terms[Simdgroups == 8 ? TileM * Batch : 1];      \
    uint row_tile = group.x;                                                 \
    uint output_tile = group.y;                                              \
    terms += ulong(row_tile) * TileM * (params.input_size / 64);             \
    input += ulong(row_tile) * TileM * params.input_size;                    \
    output += ulong(row_tile) * TileM * params.output_size;                  \
    q8_mpp_prefill_tile<Activation, TileM, TileN, Simdgroups, false, false>( \
        input, weights, scales, biases, output, output, params.output_size,  \
        params.input_size, terms, output_tile * TileN, simd_lane,            \
        simd_group, input_terms);                                            \
  }

#define Q8_PREFILL_SG4(Name, Param, Activation, InputT, TermT, TileN)          \
  kernel void Name(                                                          \
      device InputT *input [[buffer(0)]], device int8_t *weights [[buffer(1)]],\
      device Param *scales [[buffer(2)]], device Param *biases [[buffer(3)]],  \
      device bfloat *output [[buffer(4)]],                                   \
      device const TermT *terms [[buffer(5)]],                               \
      constant Q8PrefillParams &params [[buffer(6)]],                        \
      uint2 group [[threadgroup_position_in_grid]],                          \
      uint simd_lane [[thread_index_in_simdgroup]],                          \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                  \
    constexpr ushort TileM = 32;                                             \
    uint row_tile = group.x;                                                 \
    uint output_tile = group.y;                                              \
    terms += ulong(row_tile) * TileM * (params.input_size / 64);             \
    input += ulong(row_tile) * TileM * params.input_size;                    \
    output += ulong(row_tile) * TileM * params.output_size;                  \
    q8_mpp_prefill_tile<Activation, TileM, TileN, 4, false, false>(          \
        input, weights, scales, biases, output, output, params.output_size,  \
        params.input_size, terms, output_tile * TileN, simd_lane,            \
        simd_group);                                                         \
  }

#define Q8_PREFILL_RESIDUAL(Name, Param, Activation, InputT, TermT, TileN,     \
                            Simdgroups)                                       \
  kernel void Name(                                                          \
      device InputT *input [[buffer(0)]], device int8_t *weights [[buffer(1)]],\
      device Param *scales [[buffer(2)]], device Param *biases [[buffer(3)]],  \
      device bfloat *residual [[buffer(4)]],                                 \
      device bfloat *output [[buffer(5)]],                                   \
      device const TermT *terms [[buffer(6)]],                               \
      constant Q8PrefillParams &params [[buffer(7)]],                        \
      uint2 group [[threadgroup_position_in_grid]],                          \
      uint simd_lane [[thread_index_in_simdgroup]],                          \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                  \
    constexpr ushort TileM = 32;                                             \
    constexpr ushort Batch = Q8Operand<Activation>::TermBatch;               \
    threadgroup TermT input_terms[Simdgroups == 8 ? TileM * Batch : 1];      \
    uint row_tile = group.x;                                                 \
    uint output_tile = group.y;                                              \
    ulong input_offset = ulong(row_tile) * TileM * params.input_size;        \
    ulong output_offset = ulong(row_tile) * TileM * params.output_size;      \
    terms += ulong(row_tile) * TileM * (params.input_size / 64);             \
    q8_mpp_prefill_tile<Activation, TileM, TileN, Simdgroups, true, false>(  \
        input + input_offset, weights, scales, biases,                       \
        output + output_offset, residual + output_offset,                    \
        params.output_size, params.input_size, terms, output_tile * TileN,   \
        simd_lane, simd_group, input_terms);                                 \
  }

// The fused up projection applies SiLU(gate) and emits the next projection's
// input terms: fp32 sums for A16 (same signature as the Q4 kernel), or the
// quantized int8 input plus float2 terms for A8.
#define Q8_PREFILL_UP_SILU(Name, Param, TileN, Simdgroups)                     \
  kernel void Name(                                                          \
      device bfloat *input [[buffer(0)]],                                    \
      device int8_t *weights [[buffer(1)]],                                  \
      device Param *scales [[buffer(2)]], device Param *biases [[buffer(3)]],  \
      device bfloat *gate [[buffer(4)]], device bfloat *output [[buffer(5)]], \
      device const float *sums [[buffer(6)]],                                \
      device float *output_sums [[buffer(7)]],                               \
      constant Q8PrefillParams &params [[buffer(8)]],                        \
      uint2 group [[threadgroup_position_in_grid]],                          \
      uint simd_lane [[thread_index_in_simdgroup]],                          \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                  \
    constexpr ushort TileM = 32;                                             \
    constexpr ushort Batch = Q8Operand<Q8Activation::A16>::TermBatch;        \
    threadgroup float input_sums[Simdgroups == 8 ? TileM * Batch : 1];       \
    uint row_tile = group.x;                                                 \
    uint output_tile = group.y;                                              \
    ulong input_offset = ulong(row_tile) * TileM * params.input_size;        \
    ulong output_offset = ulong(row_tile) * TileM * params.output_size;      \
    sums += ulong(row_tile) * TileM * (params.input_size / 64);              \
    output_sums += ulong(row_tile) * TileM * (params.output_size / 64);      \
    q8_mpp_prefill_tile<Q8Activation::A16, TileM, TileN, Simdgroups, false,  \
                        true>(                                               \
        input + input_offset, weights, scales, biases,                       \
        output + output_offset, gate + output_offset, params.output_size,    \
        params.input_size, sums, output_tile * TileN, simd_lane,             \
        simd_group, input_sums);                                             \
    q8_prefill_write_output_sums<TileM, TileN, Simdgroups>(                  \
        output + output_offset, output_sums, params.output_size,             \
        output_tile * TileN, simd_lane, simd_group);                         \
  }

#define Q8A8_PREFILL_UP_SILU(Name, Param, TileN, Simdgroups)                 \
  kernel void Name(                                                          \
      device int8_t *input [[buffer(0)]],                                    \
      device int8_t *weights [[buffer(1)]],                                  \
      device Param *scales [[buffer(2)]], device Param *biases [[buffer(3)]],  \
      device bfloat *gate [[buffer(4)]], device bfloat *output [[buffer(5)]], \
      device const float2 *ax [[buffer(6)]],                                 \
      device int8_t *down_input [[buffer(7)]],                               \
      device float2 *down_terms [[buffer(8)]],                               \
      constant Q8PrefillParams &params [[buffer(9)]],                        \
      uint2 group [[threadgroup_position_in_grid]],                          \
      uint simd_lane [[thread_index_in_simdgroup]],                          \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                  \
    constexpr ushort TileM = 32;                                             \
    constexpr ushort Batch = Q8Operand<Q8Activation::A8>::TermBatch;         \
    threadgroup float2 input_terms[Simdgroups == 8 ? TileM * Batch : 1];     \
    uint row_tile = group.x;                                                 \
    uint output_tile = group.y;                                              \
    ulong input_offset = ulong(row_tile) * TileM * params.input_size;        \
    ulong output_offset = ulong(row_tile) * TileM * params.output_size;      \
    ax += ulong(row_tile) * TileM * (params.input_size / 64);                \
    down_input += output_offset;                                             \
    down_terms += ulong(row_tile) * TileM * (params.output_size / 64);       \
    q8_mpp_prefill_tile<Q8Activation::A8, TileM, TileN, Simdgroups, false,   \
                        true>(                                               \
        input + input_offset, weights, scales, biases,                       \
        output + output_offset, gate + output_offset, params.output_size,    \
        params.input_size, ax, output_tile * TileN, simd_lane, simd_group,   \
        input_terms);                                                        \
    q8_prefill_quantize_output<TileM, TileN, Simdgroups>(                    \
        output + output_offset, down_input, down_terms, params.output_size,  \
        output_tile * TileN, simd_lane, simd_group);                         \
  }

// A16: bf16 activations x int8 weights, fp32 group sums. Instantiated for
// fp32 (s, b') parameters and, under the _bf16p names, compact bf16 (s, z).
#define Q8_A16_PREFILL_KERNELS(Suffix, Param)                                  \
  Q8_PREFILL(prefill_linear_q8_n128##Suffix, Param, Q8Activation::A16,         \
             bfloat, float, 128, 8)                                            \
  Q8_PREFILL(prefill_linear_q8_n256##Suffix, Param, Q8Activation::A16,         \
             bfloat, float, 256, 8)                                            \
  Q8_PREFILL_SG4(prefill_linear_q8_n128_sg4##Suffix, Param,                    \
                 Q8Activation::A16, bfloat, float, 128)                        \
  Q8_PREFILL_RESIDUAL(prefill_linear_q8_n128_residual##Suffix, Param,          \
                      Q8Activation::A16, bfloat, float, 128, 8)                \
  Q8_PREFILL_RESIDUAL(prefill_linear_q8_n256_residual##Suffix, Param,          \
                      Q8Activation::A16, bfloat, float, 256, 8)                \
  Q8_PREFILL_RESIDUAL(prefill_linear_q8_n128_residual_sg4##Suffix, Param,      \
                      Q8Activation::A16, bfloat, float, 128, 4)                \
  Q8_PREFILL_UP_SILU(prefill_linear_q8_n256_up_silu_sums##Suffix, Param, 256,  \
                     8)                                                        \
  Q8_PREFILL_UP_SILU(prefill_linear_q8_n128_up_silu_sums_sg4##Suffix, Param,   \
                     128, 4)

Q8_A16_PREFILL_KERNELS(, float)
Q8_A16_PREFILL_KERNELS(_bf16p, bfloat)
#undef Q8_A16_PREFILL_KERNELS

// A8: int8 activations x int8 weights, float2 {sx, sx*qxsum} terms. Same
// two parameter forms as A16; decode A8 stays fp32 only.
#define Q8_A8_PREFILL_KERNELS(Suffix, Param)                                   \
  Q8_PREFILL(prefill_linear_q8a8_n128##Suffix, Param, Q8Activation::A8,        \
             int8_t, float2, 128, 8)                                           \
  Q8_PREFILL(prefill_linear_q8a8_n256##Suffix, Param, Q8Activation::A8,        \
             int8_t, float2, 256, 8)                                           \
  Q8_PREFILL_SG4(prefill_linear_q8a8_n128_sg4##Suffix, Param,                  \
                 Q8Activation::A8, int8_t, float2, 128)                        \
  Q8_PREFILL_RESIDUAL(prefill_linear_q8a8_n128_residual##Suffix, Param,        \
                      Q8Activation::A8, int8_t, float2, 128, 8)                \
  Q8_PREFILL_RESIDUAL(prefill_linear_q8a8_n256_residual##Suffix, Param,        \
                      Q8Activation::A8, int8_t, float2, 256, 8)                \
  Q8_PREFILL_RESIDUAL(prefill_linear_q8a8_n128_residual_sg4##Suffix, Param,    \
                      Q8Activation::A8, int8_t, float2, 128, 4)                \
  Q8A8_PREFILL_UP_SILU(prefill_linear_q8a8_n256_up_silu_quantize##Suffix,      \
                       Param, 256, 8)                                          \
  Q8A8_PREFILL_UP_SILU(prefill_linear_q8a8_n128_up_silu_quantize_sg4##Suffix,  \
                       Param, 128, 4)

Q8_A8_PREFILL_KERNELS(, float)
Q8_A8_PREFILL_KERNELS(_bf16p, bfloat)
#undef Q8_A8_PREFILL_KERNELS

// P1 candidate: one parameter producer per output column and group, consumed
// by the cooperative output tile. The 1 KiB staging region has a one-group
// lifetime, unlike the earlier all-group register hoist.
#define Q8_STAGE_PARAMETERS(Name, Param)                                      \
kernel void Name(                                                             \
    device int8_t *input [[buffer(0)]], device int8_t *weights [[buffer(1)]], \
    device Param *scales [[buffer(2)]], device Param *biases [[buffer(3)]],    \
    device bfloat *output [[buffer(4)]],                                    \
    device const float2 *terms [[buffer(5)]],                                \
    constant Q8PrefillParams &params [[buffer(6)]],                           \
    uint2 group [[threadgroup_position_in_grid]],                            \
    uint simd_lane [[thread_index_in_simdgroup]],                            \
    uint simd_group [[simdgroup_index_in_threadgroup]]) {                    \
  constexpr ushort TileM = 32;                                               \
  constexpr ushort TileN = 128;                                              \
  threadgroup float2 input_terms[1];                                        \
  threadgroup float2 staged_parameters[TileN];                               \
  const uint row_tile = group.x;                                              \
  const uint output_tile = group.y;                                           \
  terms += ulong(row_tile) * TileM * (params.input_size / 64);               \
  input += ulong(row_tile) * TileM * params.input_size;                       \
  output += ulong(row_tile) * TileM * params.output_size;                     \
  q8_mpp_prefill_tile<Q8Activation::A8, TileM, TileN, 4, false, false,     \
                      Param, false, true>(                                   \
      input, weights, scales, biases, output, output, params.output_size,    \
      params.input_size, terms, output_tile * TileN, simd_lane, simd_group, \
      input_terms, staged_parameters);                                        \
}
Q8_STAGE_PARAMETERS(prefill_linear_q8a8_n128_stagep, float)
Q8_STAGE_PARAMETERS(prefill_linear_q8a8_n128_stagep_bf16p, bfloat)
#undef Q8_STAGE_PARAMETERS

// P3-B synthetic runtime-only {scale, folded_bias} view. The permanent
// package is untouched; buffer 2 points to an interleaved float2 array and
// buffer 3 remains bound only to preserve the ordinary projection ABI.
kernel void prefill_linear_q8a8_n128_packedparams(
    device int8_t *input [[buffer(0)]], device int8_t *weights [[buffer(1)]],
    device float *paired [[buffer(2)]], device float *unused_biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const float2 *terms [[buffer(5)]],
    constant Q8PrefillParams &params [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr ushort TileM = 32, TileN = 128;
  threadgroup float2 input_terms[1];
  const uint row_tile = group.x, output_tile = group.y;
  terms += ulong(row_tile) * TileM * (params.input_size / 64);
  input += ulong(row_tile) * TileM * params.input_size;
  output += ulong(row_tile) * TileM * params.output_size;
  q8_mpp_prefill_tile<Q8Activation::A8, TileM, TileN, 4, false, false,
                      float, false, false, true>(
      input, weights, paired, unused_biases, output, output,
      params.output_size, params.input_size, terms, output_tile * TileN,
      simd_lane, simd_group, input_terms);
}

#undef Q8_PREFILL
#undef Q8_PREFILL_SG4
#undef Q8_PREFILL_RESIDUAL
#undef Q8_PREFILL_UP_SILU
#undef Q8A8_PREFILL_UP_SILU
