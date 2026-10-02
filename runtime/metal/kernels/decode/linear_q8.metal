#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/q8_mpp_tiles.h"

template <ushort Rows, ushort Simdgroups = 8>
inline void q8_split2_partial_tile(device bfloat *input, device int8_t *weights,
                                    device bfloat *scales, device bfloat *zeros,
                                    device float *partials, constant SplitQ8Params &params,
                                    threadgroup float *input_sums, uint2 group,
                                    uint simd_lane, uint simd_group,
                                    uint row_base = 0, uint scratch_rows = Rows) {
  constexpr ushort TileN = 128, StorageN = 256;
  const uint quant_groups = params.input_size / 64;
  const uint groups_per_split = (quant_groups + 1) / 2;
  const uint first = group.y * groups_per_split;
  const uint last = min(first + groups_per_split, quant_groups);
  const uint origin = group.x * TileN;
  input += ulong(row_base) * params.input_size;
  auto a = tensor(input, dextents<int, 2>{int(params.input_size), Rows},
                  array<int, 2>{1, int(params.input_size)});
  constexpr auto descriptor = matmul2d_descriptor(Rows, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  auto a0 = a.template slice<64, Rows>(0, 0);
  device int8_t *tile_weights = weights +
      (ulong(origin / StorageN) * quant_groups * StorageN + origin % StorageN) * 64;
  tensor<device int8_t, dextents<int, 2>, tensor_inline> first_b(
      tile_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
  auto b0 = first_b.slice<64, TileN>(0, 0);
  auto accumulated = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), float>();
  const bool full = uint(accumulated.get_capacity()) * Simdgroups * 32 == Rows * TileN;
  const auto traversal = full ? Q4Traversal::All : q4_traversal(accumulated);
  q4_visit(accumulated, traversal, [&](ushort i) { accumulated[i] = 0.0f; });
  for (uint base = first; base < last; base += 4) {
    q4_store_input_sums<Rows, Simdgroups>(input, params.input_size, base * 64,
                                          input_sums, 0, simd_lane, simd_group);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = base; g < min(base + 4, last); ++g) {
      auto a_slice = a.template slice<64, Rows>(g * 64, 0);
      tensor<device int8_t, dextents<int, 2>, tensor_inline> b(
          tile_weights + ulong(g) * StorageN * 64,
          dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
      auto b_slice = b.slice<64, TileN>(0, 0);
      auto dot = operation.template get_destination_cooperative_tensor<
          decltype(a_slice), decltype(b_slice), float>();
      operation.run(a_slice, b_slice, dot);
      q4_visit(accumulated, traversal, [&](ushort i) {
        const auto index = accumulated.get_multidimensional_index(i);
        const ulong parameter = (ulong(origin / StorageN) * quant_groups + g) *
                                StorageN + origin % StorageN + index[0];
        accumulated[i] = q8a16_accumulate(
            accumulated[i], dot[i], input_sums[(g - base) * Rows + index[1]],
            float(scales[parameter]), q8_folded_bias(scales, zeros, parameter));
      });
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  q4_visit(accumulated, traversal, [&](ushort i) {
    const auto index = accumulated.get_multidimensional_index(i);
    partials[(ulong(group.y) * scratch_rows + row_base + index[1]) *
                 params.output_size + origin +
             index[0]] = accumulated[i];
  });
}

#define Q8_SPLIT2(Name, Rows)                                                  \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device int8_t *weights [[buffer(1)]],                       \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *zeros [[buffer(3)]],                         \
                   device float *partials [[buffer(4)]],                       \
                   constant SplitQ8Params &params [[buffer(5)]],               \
                   uint2 group [[threadgroup_position_in_grid]],              \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[4 * Rows];                                    \
    q8_split2_partial_tile<Rows>(input, weights, scales, zeros, partials,      \
                                 params, input_sums, group, simd_lane, simd_group); \
  }
Q8_SPLIT2(decode_q8_split2_partial_m8, 8)
Q8_SPLIT2(decode_q8_split2_partial_m16, 16)
Q8_SPLIT2(decode_q8_split2_partial_m24, 24)
Q8_SPLIT2(decode_q8_split2_partial_m32, 32)
// M64 (native B8): all eight lanes of verify rows in one cooperative tile.
Q8_SPLIT2(decode_q8_split2_partial_m64, 64)
#undef Q8_SPLIT2

kernel void decode_q8_split2_partial_m64_sg16(
    device bfloat *input [[buffer(0)]], device int8_t *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *zeros [[buffer(3)]],
    device float *partials [[buffer(4)]],
    constant SplitQ8Params &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float input_sums[4 * 64];
  q8_split2_partial_tile<64, 16>(input, weights, scales, zeros, partials,
                                  params, input_sums, group, simd_lane,
                                  simd_group);
}

// B5 screen winner: compute 40 active rows; retain the 64-row split pitch.
kernel void decode_q8_split2_partial_m40_sg16(
    device bfloat *input [[buffer(0)]], device int8_t *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *zeros [[buffer(3)]],
    device float *partials [[buffer(4)]],
    constant SplitQ8Params &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float input_sums[4 * 40];
  q8_split2_partial_tile<40, 16>(input, weights, scales, zeros, partials,
      params, input_sums, group, simd_lane, simd_group, 0, params.rows);
}

kernel void decode_q8_split2_reduce_rounded_active(
    device const float *partials [[buffer(0)]],
    device const bfloat *residual [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant ActiveSplitQ8ReduceParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.active_rows * params.output_size) return;
  const float sum = partials[index] +
      partials[ulong(params.partial_pitch_rows) * params.output_size + index];
  output[index] = bfloat(float(bfloat(sum)) + float(residual[index]));
}

// M32 validation-gate capability probe: distribute the same full 32-row
// cooperative tensor over 16 SIMD groups instead of the shipped eight.
kernel void decode_q8_split2_partial_m32_sg16(
    device bfloat *input [[buffer(0)]], device int8_t *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *zeros [[buffer(3)]],
    device float *partials [[buffer(4)]],
    constant SplitQ8Params &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float input_sums[4 * 32];
  q8_split2_partial_tile<32, 16>(input, weights, scales, zeros, partials,
                                  params, input_sums, group, simd_lane,
                                  simd_group);
}

// Development release-gate candidate: each M32 command is two independent
// M16 cooperative tiles while retaining the same global split2 scratch ABI.
kernel void decode_q8_split2_partial_m32_safem16(
    device bfloat *input [[buffer(0)]], device int8_t *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *zeros [[buffer(3)]],
    device float *partials [[buffer(4)]],
    constant SplitQ8Params &params [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (params.rows != 32 || group.z >= 2) return;
  threadgroup float input_sums[4 * 16];
  q8_split2_partial_tile<16>(input, weights, scales, zeros, partials,
                              params, input_sums, group.xy, simd_lane,
                              simd_group, group.z * 16, params.rows);
}

// Wide-decode slab candidate: two M32 row slabs in one dispatch; group.z
// selects the 32-row input/partial window while the scratch layout stays
// [split][params.rows][output_size] for the shared rounded reduce.
kernel void decode_q8_split2_partial_m64_slab(
    device bfloat *input [[buffer(0)]], device int8_t *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *zeros [[buffer(3)]],
    device float *partials [[buffer(4)]],
    constant SplitQ8Params &params [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (params.rows != 64 || group.z >= 2) return;
  threadgroup float input_sums[4 * 32];
  q8_split2_partial_tile<32>(input, weights, scales, zeros, partials,
                              params, input_sums, group.xy, simd_lane,
                              simd_group, group.z * 32, params.rows);
}

// Wide-decode loop candidate: each tile visit computes the two 32-row slabs
// back to back inside one threadgroup, so the weight tile streams from DRAM
// once while each slab keeps the M32 cooperative scope and accumulation
// order. The barrier isolates the shared input_sums staging between slabs.
kernel void decode_q8_split2_partial_m64_loop(
    device bfloat *input [[buffer(0)]], device int8_t *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *zeros [[buffer(3)]],
    device float *partials [[buffer(4)]],
    constant SplitQ8Params &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (params.rows != 64) return;
  threadgroup float input_sums[4 * 32];
  q8_split2_partial_tile<32>(input, weights, scales, zeros, partials,
                              params, input_sums, group, simd_lane, simd_group,
                              0, params.rows);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  q8_split2_partial_tile<32>(input, weights, scales, zeros, partials,
                              params, input_sums, group, simd_lane, simd_group,
                              32, params.rows);
}

// Shared-weight split2 partial candidate: same dual-fragment structure as
// q8_mpp_tile_dual - one threadgroup runs the 32-row fragment and the Rows1
// tail fragment against each weight group slice once, writing both slabs of
// the shared [split][rows][output] partial scratch. Per-row accumulation
// order is unchanged, so active rows stay bit-identical.
template <ushort Rows1>
inline void q8_split2_partial_tile_dual(
    device bfloat *input, device int8_t *weights, device bfloat *scales,
    device bfloat *zeros, device float *partials,
    constant SplitQ8Params &params, threadgroup float *sums_0,
    threadgroup float *sums_1, uint2 group, uint simd_lane, uint simd_group) {
  constexpr ushort Rows0 = 32, Simdgroups = 8;
  constexpr ushort TileN = 128, StorageN = 256;
  const uint quant_groups = params.input_size / 64;
  const uint groups_per_split = (quant_groups + 1) / 2;
  const uint first = group.y * groups_per_split;
  const uint last = min(first + groups_per_split, quant_groups);
  const uint origin = group.x * TileN;
  device bfloat *input_1 = input + ulong(Rows0) * params.input_size;
  auto a_0 = tensor(input, dextents<int, 2>{int(params.input_size), Rows0},
                    array<int, 2>{1, int(params.input_size)});
  auto a_1 = tensor(input_1, dextents<int, 2>{int(params.input_size), Rows1},
                    array<int, 2>{1, int(params.input_size)});
  constexpr auto descriptor_0 = matmul2d_descriptor(Rows0, TileN, 64, false, true, false);
  constexpr auto descriptor_1 = matmul2d_descriptor(Rows1, TileN, 64, false, true, false);
  matmul2d<descriptor_0, execution_simdgroups<Simdgroups>> operation_0;
  matmul2d<descriptor_1, execution_simdgroups<Simdgroups>> operation_1;
  auto a0 = a_0.template slice<64, Rows0>(0, 0);
  auto a1 = a_1.template slice<64, Rows1>(0, 0);
  device int8_t *tile_weights = weights +
      (ulong(origin / StorageN) * quant_groups * StorageN + origin % StorageN) * 64;
  tensor<device int8_t, dextents<int, 2>, tensor_inline> first_b(
      tile_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
  auto b0 = first_b.slice<64, TileN>(0, 0);
  auto accumulated_0 = operation_0.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), float>();
  auto accumulated_1 = operation_1.template get_destination_cooperative_tensor<
      decltype(a1), decltype(b0), float>();
  const bool full_0 = uint(accumulated_0.get_capacity()) * Simdgroups * 32 == Rows0 * TileN;
  const bool full_1 = uint(accumulated_1.get_capacity()) * Simdgroups * 32 == Rows1 * TileN;
  const auto traversal_0 = full_0 ? Q4Traversal::All : q4_traversal(accumulated_0);
  const auto traversal_1 = full_1 ? Q4Traversal::All : q4_traversal(accumulated_1);
  q4_visit(accumulated_0, traversal_0, [&](ushort i) { accumulated_0[i] = 0.0f; });
  q4_visit(accumulated_1, traversal_1, [&](ushort i) { accumulated_1[i] = 0.0f; });
  for (uint base = first; base < last; base += 4) {
    q4_store_input_sums<Rows0, Simdgroups>(input, params.input_size, base * 64,
                                         sums_0, 0, simd_lane, simd_group);
    q4_store_input_sums<Rows1, Simdgroups>(input_1, params.input_size,
                                           base * 64, sums_1, 0, simd_lane,
                                           simd_group);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = base; g < min(base + 4, last); ++g) {
      auto a0_slice = a_0.template slice<64, Rows0>(g * 64, 0);
      auto a1_slice = a_1.template slice<64, Rows1>(g * 64, 0);
      tensor<device int8_t, dextents<int, 2>, tensor_inline> b(
          tile_weights + ulong(g) * StorageN * 64,
          dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
      auto b_slice = b.slice<64, TileN>(0, 0);
      auto dot_0 = operation_0.template get_destination_cooperative_tensor<
          decltype(a0_slice), decltype(b_slice), float>();
      auto dot_1 = operation_1.template get_destination_cooperative_tensor<
          decltype(a1_slice), decltype(b_slice), float>();
      operation_0.run(a0_slice, b_slice, dot_0);
      operation_1.run(a1_slice, b_slice, dot_1);
      q4_visit(accumulated_0, traversal_0, [&](ushort i) {
        const auto index = accumulated_0.get_multidimensional_index(i);
        const ulong parameter = (ulong(origin / StorageN) * quant_groups + g) *
                                StorageN + origin % StorageN + index[0];
        accumulated_0[i] = q8a16_accumulate(
            accumulated_0[i], dot_0[i], sums_0[(g - base) * Rows0 + index[1]],
            float(scales[parameter]), q8_folded_bias(scales, zeros, parameter));
      });
      q4_visit(accumulated_1, traversal_1, [&](ushort i) {
        const auto index = accumulated_1.get_multidimensional_index(i);
        const ulong parameter = (ulong(origin / StorageN) * quant_groups + g) *
                                StorageN + origin % StorageN + index[0];
        accumulated_1[i] = q8a16_accumulate(
            accumulated_1[i], dot_1[i], sums_1[(g - base) * Rows1 + index[1]],
            float(scales[parameter]), q8_folded_bias(scales, zeros, parameter));
      });
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint scratch_rows = params.rows;
  q4_visit(accumulated_0, traversal_0, [&](ushort i) {
    const auto index = accumulated_0.get_multidimensional_index(i);
    partials[(ulong(group.y) * scratch_rows + index[1]) * params.output_size +
             origin + index[0]] = accumulated_0[i];
  });
  q4_visit(accumulated_1, traversal_1, [&](ushort i) {
    const auto index = accumulated_1.get_multidimensional_index(i);
    partials[(ulong(group.y) * scratch_rows + Rows0 + index[1]) *
             params.output_size + origin + index[0]] = accumulated_1[i];
  });
}

#define Q8_SPLIT2_DUAL(Name, Rows)                                             \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device int8_t *weights [[buffer(1)]],                       \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *zeros [[buffer(3)]],                         \
                   device float *partials [[buffer(4)]],                       \
                   constant SplitQ8Params &params [[buffer(5)]],               \
                   uint2 group [[threadgroup_position_in_grid]],               \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float sums_0[4 * 32];                                          \
    threadgroup float sums_1[4 * Rows];                                        \
    q8_split2_partial_tile_dual<Rows>(input, weights, scales, zeros,          \
                                    partials, params, sums_0, sums_1, group,  \
                                    simd_lane, simd_group);                    \
  }
Q8_SPLIT2_DUAL(decode_q8_split2_partial_d40, 8)
Q8_SPLIT2_DUAL(decode_q8_split2_partial_d48, 16)
Q8_SPLIT2_DUAL(decode_q8_split2_partial_d56, 24)
Q8_SPLIT2_DUAL(decode_q8_split2_partial_d64, 32)
#undef Q8_SPLIT2_DUAL

kernel void decode_q8_split2_reduce_rounded(
    device const float *partials [[buffer(0)]],
    device const bfloat *residual [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant SplitQ8Params &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.rows * params.output_size) return;
  const uint row = index / params.output_size;
  const uint col = index % params.output_size;
  const float sum = partials[row * params.output_size + col] +
                    partials[(params.rows + row) * params.output_size + col];
  output[index] = bfloat(float(bfloat(sum)) + float(residual[index]));
}

// Dense Q8 decode projections: persistent threadgroups stride over
// TileN-wide output tiles. A16 kernels mirror the Q4 buffer layout with a
// bf16 input and in-kernel group sums; A8 kernels take the quantized int8
// input plus the precomputed [group][row] float2 terms (written by
// decode_linear_q8a8_quantize or norm_rms_q8a8_decode) and need no
// threadgroup staging.

// Quantize one row per threadgroup: row-major int8 qx plus [group][row]
// float2 terms.
kernel void decode_linear_q8a8_quantize(
    device const bfloat *input [[buffer(0)]],
    device int8_t *qx [[buffer(1)]],
    device float2 *ax [[buffer(2)]],
    constant Q8QuantizeParams &params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint quant_groups = params.input_size / 64;
  for (uint quant_group = simd_group; quant_group < quant_groups;
       quant_group += 8) {
    const uint origin =
        row * params.input_size + quant_group * 64 + simd_lane;
    int qa, qb;
    const float2 term = q8_quantize_pair(
        float(input[origin]), float(input[origin + 32]), qa, qb);
    qx[origin] = int8_t(qa);
    qx[origin + 32] = int8_t(qb);
    if (simd_lane == 0)
      ax[quant_group * params.rows + row] = term;
  }
}

#define Q8_DECODE_AFFINE(Name, Param, TileCall, Sums, TileN)                   \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device int8_t *weights [[buffer(1)]],                       \
                   device Param *scales [[buffer(2)]],                         \
                   device Param *biases [[buffer(3)]],                         \
                   device bfloat *output [[buffer(4)]],                        \
                   constant Q8DecodeParams &params [[buffer(5)]],              \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      TileCall(input, weights, scales, biases, output, weights, scales,        \
               biases, output, params.output_size, params.input_size,          \
               input_sums, nullptr, tile * TileN, simd_lane, simd_group);      \
    }                                                                          \
  }

#define Q8_DECODE_AUXILIARY(Name, Param, Auxiliary, TileCall, Sums, TileN)     \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device int8_t *weights [[buffer(1)]],                       \
                   device Param *scales [[buffer(2)]],                         \
                   device Param *biases [[buffer(3)]],                         \
                   device bfloat *Auxiliary [[buffer(4)]],                     \
                   device bfloat *output [[buffer(5)]],                        \
                   constant Q8DecodeParams &params [[buffer(6)]],              \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      TileCall(input, weights, scales, biases, output, weights, scales,        \
               biases, Auxiliary, params.output_size, params.input_size,       \
               input_sums, nullptr, tile * TileN, simd_lane, simd_group);      \
    }                                                                          \
  }

#define Q8_DECODE_GATE_UP(Name, Param, TileCall, Sums, TileN)                  \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device int8_t *weights_0 [[buffer(1)]],                     \
                   device Param *scales_0 [[buffer(2)]],                       \
                   device Param *biases_0 [[buffer(3)]],                       \
                   device bfloat *output [[buffer(4)]],                        \
                   device int8_t *weights_1 [[buffer(5)]],                     \
                   device Param *scales_1 [[buffer(6)]],                       \
                   device Param *biases_1 [[buffer(7)]],                       \
                   constant Q8DecodeParams &params [[buffer(8)]],              \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      TileCall(input, weights_0, scales_0, biases_0, output, weights_1,        \
               scales_1, biases_1, output, params.output_size,                 \
               params.input_size, input_sums, nullptr, tile * TileN,           \
               simd_lane, simd_group);                                         \
    }                                                                          \
  }

// The A8 forms take the quantized input and the device term table; the last
// buffer before the parameters is always the float2 ax terms.
#define Q8A8_DECODE_AFFINE(Name, TileCall, TileN)                              \
  kernel void Name(device int8_t *input [[buffer(0)]],                         \
                   device int8_t *weights [[buffer(1)]],                       \
                   device float *scales [[buffer(2)]],                         \
                   device float *biases [[buffer(3)]],                         \
                   device bfloat *output [[buffer(4)]],                        \
                   device const float2 *ax [[buffer(5)]],                      \
                   constant Q8DecodeParams &params [[buffer(6)]],              \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      TileCall(input, weights, scales, biases, output, weights, scales,        \
               biases, output, params.output_size, params.input_size,          \
               nullptr, ax, tile * TileN, simd_lane, simd_group);              \
    }                                                                          \
  }

#define Q8A8_DECODE_AUXILIARY(Name, Auxiliary, TileCall, TileN)                \
  kernel void Name(device int8_t *input [[buffer(0)]],                         \
                   device int8_t *weights [[buffer(1)]],                       \
                   device float *scales [[buffer(2)]],                         \
                   device float *biases [[buffer(3)]],                         \
                   device bfloat *Auxiliary [[buffer(4)]],                     \
                   device bfloat *output [[buffer(5)]],                        \
                   device const float2 *ax [[buffer(6)]],                      \
                   constant Q8DecodeParams &params [[buffer(7)]],              \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      TileCall(input, weights, scales, biases, output, weights, scales,        \
               biases, Auxiliary, params.output_size, params.input_size,       \
               nullptr, ax, tile * TileN, simd_lane, simd_group);              \
    }                                                                          \
  }

#define Q8A8_DECODE_GATE_UP(Name, TileCall, TileN)                             \
  kernel void Name(device int8_t *input [[buffer(0)]],                         \
                   device int8_t *weights_0 [[buffer(1)]],                     \
                   device float *scales_0 [[buffer(2)]],                       \
                   device float *biases_0 [[buffer(3)]],                       \
                   device bfloat *output [[buffer(4)]],                        \
                   device int8_t *weights_1 [[buffer(5)]],                     \
                   device float *scales_1 [[buffer(6)]],                       \
                   device float *biases_1 [[buffer(7)]],                       \
                   device const float2 *ax [[buffer(8)]],                      \
                   constant Q8DecodeParams &params [[buffer(9)]],              \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      TileCall(input, weights_0, scales_0, biases_0, output, weights_1,        \
               scales_1, biases_1, output, params.output_size,                 \
               params.input_size, nullptr, ax, tile * TileN, simd_lane,        \
               simd_group);                                                    \
    }                                                                          \
  }

// Row-slab wide decode candidates: one dispatch carries independent 32-row
// cooperative tiles selected by group.z. Each slab runs the identical M32
// tile math on a 32-row window of the shared activation/output/residual
// buffers, so active rows stay bit-identical to the m32 kernels while one
// command covers the whole logical width.
#define Q8_DECODE_SLAB_AFFINE(Name, Param, TileCall, Sums, TileN)              \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device int8_t *weights [[buffer(1)]],                       \
                   device Param *scales [[buffer(2)]],                         \
                   device Param *biases [[buffer(3)]],                         \
                   device bfloat *output [[buffer(4)]],                        \
                   constant Q8DecodeParams &params [[buffer(5)]],              \
                   uint3 group [[threadgroup_position_in_grid]],               \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    input += ulong(group.z) * 32 * params.input_size;                          \
    output += ulong(group.z) * 32 * params.output_size;                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group.x; tile < tiles; tile += params.persistent_groups) { \
      TileCall(input, weights, scales, biases, output, weights, scales,        \
               biases, output, params.output_size, params.input_size,          \
               input_sums, nullptr, tile * TileN, simd_lane, simd_group);      \
    }                                                                          \
  }

#define Q8_DECODE_SLAB_AUXILIARY(Name, Param, Auxiliary, TileCall, Sums,       \
                                 TileN)                                        \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device int8_t *weights [[buffer(1)]],                       \
                   device Param *scales [[buffer(2)]],                         \
                   device Param *biases [[buffer(3)]],                         \
                   device bfloat *Auxiliary [[buffer(4)]],                     \
                   device bfloat *output [[buffer(5)]],                        \
                   constant Q8DecodeParams &params [[buffer(6)]],              \
                   uint3 group [[threadgroup_position_in_grid]],               \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    input += ulong(group.z) * 32 * params.input_size;                          \
    Auxiliary += ulong(group.z) * 32 * params.output_size;                     \
    output += ulong(group.z) * 32 * params.output_size;                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group.x; tile < tiles; tile += params.persistent_groups) { \
      TileCall(input, weights, scales, biases, output, weights, scales,        \
               biases, Auxiliary, params.output_size, params.input_size,       \
               input_sums, nullptr, tile * TileN, simd_lane, simd_group);      \
    }                                                                          \
  }

// Interleaved slab candidates (shared-weight campaign C0): identical tile math
// to the slab kernels, but a 1D grid of 2 * persistent_groups threadgroups
// where adjacent threadgroup ids are the two slabs of the same weight tile
// (slab = gid & 1), so both slabs stream the tile at nearly the same time.
#define Q8_DECODE_SLABI_AFFINE(Name, Param, TileCall, Sums, TileN)             \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device int8_t *weights [[buffer(1)]],                       \
                   device Param *scales [[buffer(2)]],                         \
                   device Param *biases [[buffer(3)]],                         \
                   device bfloat *output [[buffer(4)]],                        \
                   constant Q8DecodeParams &params [[buffer(5)]],              \
                   uint gid [[threadgroup_position_in_grid]],                  \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    input += ulong(gid & 1) * 32 * params.input_size;                          \
    output += ulong(gid & 1) * 32 * params.output_size;                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = gid >> 1; tile < tiles; tile += params.persistent_groups) { \
      TileCall(input, weights, scales, biases, output, weights, scales,        \
               biases, output, params.output_size, params.input_size,          \
               input_sums, nullptr, tile * TileN, simd_lane, simd_group);      \
    }                                                                          \
  }

#define Q8_DECODE_SLABI_AUXILIARY(Name, Param, Auxiliary, TileCall, Sums,      \
                                  TileN)                                       \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device int8_t *weights [[buffer(1)]],                       \
                   device Param *scales [[buffer(2)]],                         \
                   device Param *biases [[buffer(3)]],                         \
                   device bfloat *Auxiliary [[buffer(4)]],                     \
                   device bfloat *output [[buffer(5)]],                        \
                   constant Q8DecodeParams &params [[buffer(6)]],              \
                   uint gid [[threadgroup_position_in_grid]],                  \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    input += ulong(gid & 1) * 32 * params.input_size;                          \
    Auxiliary += ulong(gid & 1) * 32 * params.output_size;                     \
    output += ulong(gid & 1) * 32 * params.output_size;                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = gid >> 1; tile < tiles; tile += params.persistent_groups) { \
      TileCall(input, weights, scales, biases, output, weights, scales,        \
               biases, Auxiliary, params.output_size, params.input_size,       \
               input_sums, nullptr, tile * TileN, simd_lane, simd_group);      \
    }                                                                          \
  }

// Shared-weight dual-fragment kernels (campaign C1): one threadgroup runs the
// 32-row fragment and a Rows1-row tail fragment per weight tile, 1D grid of
// persistent_groups threadgroups. Rows0 = 32, active rows = 32 + Rows1.
#define Q8_DECODE_DUAL_AFFINE(Name, Param, Rows1, TileN, SG)                     \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device int8_t *weights [[buffer(1)]],                       \
                   device Param *scales [[buffer(2)]],                         \
                   device Param *biases [[buffer(3)]],                         \
                   device bfloat *output [[buffer(4)]],                        \
                   constant Q8DecodeParams &params [[buffer(5)]],              \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float sums_0[256];                                             \
    threadgroup float sums_1[8 * Rows1];                                       \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      q8_mpp_tile_dual<Rows1, TileN, false, TileN, Param, SG>(                            \
          input, weights, scales, biases, output, output, params.output_size,  \
          params.input_size, sums_0, sums_1, tile * TileN, simd_lane,          \
          simd_group);                                                         \
    }                                                                          \
  }

#define Q8_DECODE_DUAL_SILU(Name, Param, Rows1, TileN, SG)                     \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device int8_t *weights [[buffer(1)]],                       \
                   device Param *scales [[buffer(2)]],                         \
                   device Param *biases [[buffer(3)]],                         \
                   device bfloat *gate [[buffer(4)]],                          \
                   device bfloat *output [[buffer(5)]],                        \
                   constant Q8DecodeParams &params [[buffer(6)]],              \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float sums_0[256];                                             \
    threadgroup float sums_1[8 * Rows1];                                       \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      q8_mpp_tile_dual<Rows1, TileN, true, TileN, Param, SG>(                             \
          input, weights, scales, biases, output, gate, params.output_size,    \
          params.input_size, sums_0, sums_1, tile * TileN, simd_lane,          \
          simd_group);                                                         \
    }                                                                          \
  }

// Asymmetric slab-pair candidates (campaign C-family): one dispatch, 1D grid
// of 2 * persistent_groups threadgroups; even gid runs the 32-row fragment,
// odd gid runs the Rows1-row tail fragment for the same weight tile index, so
// the tail's DRAM fetch overlaps the main fragment's stream instead of
// serializing a full second pass. Per-fragment math is identical to the m32 /
// m<Rows1> batched tiles, so active rows are bit-identical.
#define Q8_DECODE_ASLAB_AFFINE(Name, Param, Rows1, TileN)                      \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device int8_t *weights [[buffer(1)]],                       \
                   device Param *scales [[buffer(2)]],                         \
                   device Param *biases [[buffer(3)]],                         \
                   device bfloat *output [[buffer(4)]],                        \
                   constant Q8DecodeParams &params [[buffer(5)]],              \
                   uint gid [[threadgroup_position_in_grid]],                  \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float sums_0[256];                                             \
    threadgroup float sums_1[8 * Rows1];                                       \
    const uint base = gid & 1;                                                 \
    device bfloat *frag_input = input + ulong(base) * 32 * params.input_size;  \
    device bfloat *frag_output = output + ulong(base) * 32 *                   \
                                 params.output_size;                           \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = gid >> 1; tile < tiles; tile += params.persistent_groups) { \
      if (base)                                                                \
        q8_mpp_tile_batched<Q8Activation::A16, Rows1, TileN, false, false,     \
                            256>(frag_input, weights, scales, biases,          \
                                 frag_output, weights, scales, biases,         \
                                 frag_output, params.output_size,              \
                                 params.input_size, sums_1, nullptr,           \
                                 tile * TileN, simd_lane, simd_group);         \
      else                                                                     \
        q8_mpp_tile_batched<Q8Activation::A16, 32, TileN, false, false,        \
                            256>(frag_input, weights, scales, biases,          \
                                 frag_output, weights, scales, biases,         \
                                 frag_output, params.output_size,              \
                                 params.input_size, sums_0, nullptr,           \
                                 tile * TileN, simd_lane, simd_group);         \
    }                                                                          \
  }

#define Q8_DECODE_ASLAB_SILU(Name, Param, Rows1, TileN)                        \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device int8_t *weights [[buffer(1)]],                       \
                   device Param *scales [[buffer(2)]],                         \
                   device Param *biases [[buffer(3)]],                         \
                   device bfloat *gate [[buffer(4)]],                          \
                   device bfloat *output [[buffer(5)]],                        \
                   constant Q8DecodeParams &params [[buffer(6)]],              \
                   uint gid [[threadgroup_position_in_grid]],                  \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float sums_0[256];                                             \
    threadgroup float sums_1[8 * Rows1];                                       \
    const uint base = gid & 1;                                                 \
    device bfloat *frag_input = input + ulong(base) * 32 * params.input_size;  \
    device bfloat *frag_output = output + ulong(base) * 32 *                   \
                                 params.output_size;                           \
    device bfloat *frag_gate = gate + ulong(base) * 32 * params.output_size;   \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = gid >> 1; tile < tiles; tile += params.persistent_groups) { \
      if (base)                                                                \
        q8_mpp_tile_batched<Q8Activation::A16, Rows1, TileN, false, false,     \
                            256, true>(frag_input, weights, scales, biases,    \
                                       frag_output, weights, scales, biases,   \
                                       frag_gate, params.output_size,          \
                                       params.input_size, sums_1, nullptr,     \
                                       tile * TileN, simd_lane, simd_group);   \
      else                                                                     \
        q8_mpp_tile_batched<Q8Activation::A16, 32, TileN, false, false,        \
                            256, true>(frag_input, weights, scales, biases,    \
                                       frag_output, weights, scales, biases,   \
                                       frag_gate, params.output_size,          \
                                       params.input_size, sums_0, nullptr,     \
                                       tile * TileN, simd_lane, simd_group);   \
    }                                                                          \
  }

// Wide-decode loop candidates: each tile visit runs two sequential 32-row
// cooperative tiles inside one threadgroup, so the weight tile streams once
// while both slabs keep the M32 tile math and accumulation order.
#define Q8_DECODE_LOOP_AFFINE(Name, Param, TileCall, Sums, TileN)              \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device int8_t *weights [[buffer(1)]],                       \
                   device Param *scales [[buffer(2)]],                         \
                   device Param *biases [[buffer(3)]],                         \
                   device bfloat *output [[buffer(4)]],                        \
                   constant Q8DecodeParams &params [[buffer(5)]],              \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      TileCall(input, weights, scales, biases, output, weights, scales,        \
               biases, output, params.output_size, params.input_size,          \
               input_sums, nullptr, tile * TileN, simd_lane, simd_group);      \
      threadgroup_barrier(mem_flags::mem_threadgroup);                         \
      TileCall(input + ulong(32) * params.input_size, weights, scales,         \
               biases, output + ulong(32) * params.output_size, weights,       \
               scales, biases, output + ulong(32) * params.output_size,        \
               params.output_size, params.input_size, input_sums, nullptr,     \
               tile * TileN, simd_lane, simd_group);                           \
      threadgroup_barrier(mem_flags::mem_threadgroup);                         \
    }                                                                          \
  }

#define Q8_DECODE_LOOP_AUXILIARY(Name, Param, Auxiliary, TileCall, Sums,       \
                                 TileN)                                        \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device int8_t *weights [[buffer(1)]],                       \
                   device Param *scales [[buffer(2)]],                         \
                   device Param *biases [[buffer(3)]],                         \
                   device bfloat *Auxiliary [[buffer(4)]],                     \
                   device bfloat *output [[buffer(5)]],                        \
                   constant Q8DecodeParams &params [[buffer(6)]],              \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      TileCall(input, weights, scales, biases, output, weights, scales,        \
               biases, Auxiliary, params.output_size, params.input_size,       \
               input_sums, nullptr, tile * TileN, simd_lane, simd_group);      \
      threadgroup_barrier(mem_flags::mem_threadgroup);                         \
      TileCall(input + ulong(32) * params.input_size, weights, scales,         \
               biases, output + ulong(32) * params.output_size, weights,       \
               scales, biases,                                                 \
               Auxiliary + ulong(32) * params.output_size, params.output_size, \
               params.input_size, input_sums, nullptr, tile * TileN,           \
               simd_lane, simd_group);                                         \
      threadgroup_barrier(mem_flags::mem_threadgroup);                         \
    }                                                                          \
  }

// A16: bf16 activations x int8 weights; in-kernel group sums like Q4.
// Instantiated twice: fp32 (s, b') parameters under the plain names and the
// compact bf16 (s, z) parameters under the _bf16p names.
#define Q8_A16_DECODE_KERNELS(Suffix, Param)                                    \
Q8_DECODE_AFFINE(decode_linear_q8_n128##Suffix, Param,                         \
                 (q8_mpp_tile<Q8Activation::A16, 128, false, false, 256>),     \
                 64, 128)                                                      \
Q8_DECODE_AFFINE(decode_linear_q8_n128_m16##Suffix, Param,                     \
                 (q8_mpp_tile_batched<Q8Activation::A16, 16, 128, false, false, \
                                      256>),                                   \
                 128, 128)                                                     \
Q8_DECODE_AFFINE(decode_linear_q8_n128_m24##Suffix, Param,                     \
                 (q8_mpp_tile_batched<Q8Activation::A16, 24, 128, false, false, \
                                      256>),                                   \
                 192, 128)                                                     \
Q8_DECODE_AFFINE(decode_linear_q8_n128_m24_sg4##Suffix, Param,                 \
                 (q8_mpp_tile_batched<Q8Activation::A16, 24, 128, false, false, \
                                      256, false, 4>),                         \
                 192, 128)                                                     \
Q8_DECODE_AFFINE(decode_linear_q8_n256_m16##Suffix, Param,                     \
                 (q8_mpp_tile_batched<Q8Activation::A16, 16, 256, false,       \
                                      false>),                                 \
                 128, 256)                                                     \
Q8_DECODE_AFFINE(decode_linear_q8_n256_m24##Suffix, Param,                     \
                 (q8_mpp_tile_batched<Q8Activation::A16, 24, 256, false,       \
                                      false>),                                 \
                 192, 256)                                                     \
Q8_DECODE_AFFINE(decode_linear_q8_n256##Suffix, Param,                         \
                 (q8_mpp_tile<Q8Activation::A16, 256, false, false>), 64, 256) \
Q8_DECODE_AFFINE(decode_linear_q8_n128_paired##Suffix, Param,                  \
                 (q8_mpp_tile<Q8Activation::A16, 128, false, false, 256,       \
                              true>),                                          \
                 64, 128)                                                      \
Q8_DECODE_AFFINE(decode_linear_q8_n256_paired_sg4##Suffix, Param,              \
                 (q8_mpp_tile<Q8Activation::A16, 256, false, false, 256, true, \
                              4>),                                             \
                 64, 256)                                                      \
Q8_DECODE_AUXILIARY(decode_linear_q8_n128_residual_paired##Suffix, Param, residual, \
                    (q8_mpp_tile<Q8Activation::A16, 128, false, true, 256,     \
                                 true>),                                       \
                    64, 128)                                                   \
Q8_DECODE_AUXILIARY(decode_linear_q8_n128_residual##Suffix, Param, residual,   \
                    (q8_mpp_tile<Q8Activation::A16, 128, false, true, 256>),   \
                    64, 128)                                                   \
Q8_DECODE_GATE_UP(decode_linear_q8_n256_gate_up##Suffix, Param,                \
                  (q8_mpp_tile<Q8Activation::A16, 256, true, false>), 64, 256) \
Q8_DECODE_AUXILIARY(decode_linear_q8_n128_residual_m16##Suffix, Param, residual, \
                    (q8_mpp_tile_batched<Q8Activation::A16, 16, 128, false,    \
                                         true, 256>),                          \
                    128, 128)                                                  \
Q8_DECODE_AUXILIARY(decode_linear_q8_n128_residual_m24##Suffix, Param, residual, \
                    (q8_mpp_tile_batched<Q8Activation::A16, 24, 128, false,    \
                                         true, 256>),                          \
                    192, 128)                                                  \
Q8_DECODE_AUXILIARY(decode_linear_q8_n128_residual_m24_sg4##Suffix, Param, residual, \
                    (q8_mpp_tile_batched<Q8Activation::A16, 24, 128, false,    \
                                         true, 256, false, 4>),                \
                    192, 128)                                                  \
Q8_DECODE_GATE_UP(decode_linear_q8_n256_gate_up_m16##Suffix, Param,            \
                  (q8_mpp_tile_batched<Q8Activation::A16, 16, 256, true,       \
                                       false>),                                \
                  128, 256)                                                    \
Q8_DECODE_AFFINE(decode_linear_q8_n128_m32##Suffix, Param,                     \
                 (q8_mpp_tile_batched<Q8Activation::A16, 32, 128, false, false, \
                                      256>),                                   \
                 256, 128)                                                     \
Q8_DECODE_AFFINE(decode_linear_q8_n256_m32##Suffix, Param,                     \
                 (q8_mpp_tile_batched<Q8Activation::A16, 32, 256, false,       \
                                      false>),                                 \
                 256, 256)                                                     \
Q8_DECODE_AUXILIARY(decode_linear_q8_n128_residual_m32##Suffix, Param, residual, \
                    (q8_mpp_tile_batched<Q8Activation::A16, 32, 128, false,    \
                                         true, 256>),                          \
                    256, 128)                                                  \
Q8_DECODE_AUXILIARY(decode_linear_q8_n256_up_silu_m32##Suffix, Param, gate,    \
                    (q8_mpp_tile_batched<Q8Activation::A16, 32, 256, false,    \
                                         false, 256, true>),                   \
                    256, 256)                                                  \
Q8_DECODE_AUXILIARY(decode_linear_q8_n256_up_silu_m24##Suffix, Param, gate,    \
                    (q8_mpp_tile_batched<Q8Activation::A16, 24, 256, false,    \
                                         false, 256, true>),                   \
                    192, 256)                                                  \
Q8_DECODE_AUXILIARY(decode_linear_q8_n64_residual_m24##Suffix, Param, residual, \
                    (q8_mpp_tile_batched<Q8Activation::A16, 24, 64, false,     \
                                         true, 256>),                          \
                    192, 64)                                                   \
Q8_DECODE_AUXILIARY(decode_linear_q8_n64_residual_m32##Suffix, Param, residual, \
                    (q8_mpp_tile_batched<Q8Activation::A16, 32, 64, false,     \
                                         true, 256>),                          \
                    256, 64)                                                   \
Q8_DECODE_AUXILIARY(decode_linear_q8_n128_residual_m16_paired##Suffix, Param,  \
                    residual,                                                  \
                    (q8_mpp_tile_batched<Q8Activation::A16, 16, 128, false,    \
                                         true, 256, false, 8, Param, true>),   \
                    128, 128)                                                  \
Q8_DECODE_AUXILIARY(decode_linear_q8_n64_residual_m32_paired##Suffix, Param,   \
                    residual,                                                  \
                    (q8_mpp_tile_batched<Q8Activation::A16, 32, 64, false,     \
                                         true, 256, false, 8, Param, true>),   \
                    256, 64) \
Q8_DECODE_AFFINE(decode_linear_q8_n128_m64##Suffix, Param,                     \
                 (q8_mpp_tile_batched<Q8Activation::A16, 64, 128, false, false, \
                                      256>),                                   \
                 512, 128)                                                     \
Q8_DECODE_AFFINE(decode_linear_q8_n256_m64##Suffix, Param,                     \
                 (q8_mpp_tile_batched<Q8Activation::A16, 64, 256, false,       \
                                      false>),                                 \
                 512, 256)                                                     \
Q8_DECODE_AUXILIARY(decode_linear_q8_n128_residual_m64##Suffix, Param, residual, \
                    (q8_mpp_tile_batched<Q8Activation::A16, 64, 128, false,    \
                                         true, 256>),                          \
                    512, 128)                                                  \
Q8_DECODE_AUXILIARY(decode_linear_q8_n64_residual_m64##Suffix, Param, residual, \
                    (q8_mpp_tile_batched<Q8Activation::A16, 64, 64, false,     \
                                         true, 256>),                          \
                    512, 64)                                                   \
Q8_DECODE_AUXILIARY(decode_linear_q8_n256_up_silu_m64##Suffix, Param, gate,    \
                    (q8_mpp_tile_batched<Q8Activation::A16, 64, 256, false,    \
                                         false, 256, true>),                   \
                    512, 256)                                                  \
Q8_DECODE_SLAB_AFFINE(decode_linear_q8_n128_m64_slab##Suffix, Param,           \
                      (q8_mpp_tile_batched<Q8Activation::A16, 32, 128, false,  \
                                           false, 256>),                      \
                      256, 128)                                                \
Q8_DECODE_SLAB_AFFINE(decode_linear_q8_n256_m64_slab##Suffix, Param,           \
                      (q8_mpp_tile_batched<Q8Activation::A16, 32, 256, false,  \
                                           false>),                           \
                      256, 256)                                                \
Q8_DECODE_SLAB_AUXILIARY(decode_linear_q8_n128_residual_m64_slab##Suffix,      \
                         Param, residual,                                     \
                         (q8_mpp_tile_batched<Q8Activation::A16, 32, 128,     \
                                              false, true, 256>),              \
                         256, 128)                                              \
Q8_DECODE_SLAB_AUXILIARY(decode_linear_q8_n64_residual_m64_slab##Suffix,       \
                         Param, residual,                                     \
                         (q8_mpp_tile_batched<Q8Activation::A16, 32, 64,      \
                                              false, true, 256>),              \
                         256, 64)                                               \
Q8_DECODE_SLAB_AUXILIARY(decode_linear_q8_n256_up_silu_m64_slab##Suffix,       \
                         Param, gate,                                         \
                         (q8_mpp_tile_batched<Q8Activation::A16, 32, 256,     \
                                              false, false, 256, true>),       \
                         256, 256)                                              \
Q8_DECODE_AFFINE(decode_linear_q8_n256_m40##Suffix, Param,                 \
                 (q8_mpp_tile_batched<Q8Activation::A16, 40, 256, false,       \
                                      false>),                                 \
                 320, 256)                                                     \
Q8_DECODE_AUXILIARY(decode_linear_q8_n256_up_silu_m40##Suffix, Param, gate,    \
                    (q8_mpp_tile_batched<Q8Activation::A16, 40, 256, false,    \
                                         false, 256, true>),                    \
                    320, 256)                                                  \
Q8_DECODE_AFFINE(decode_linear_q8_n256_m48##Suffix, Param,                 \
                 (q8_mpp_tile_batched<Q8Activation::A16, 48, 256, false,       \
                                      false>),                                 \
                 384, 256)                                                     \
Q8_DECODE_AUXILIARY(decode_linear_q8_n256_up_silu_m48##Suffix, Param, gate,    \
                    (q8_mpp_tile_batched<Q8Activation::A16, 48, 256, false,    \
                                         false, 256, true>),                    \
                    384, 256)                                                  \
Q8_DECODE_AFFINE(decode_linear_q8_n256_m56##Suffix, Param,                 \
                 (q8_mpp_tile_batched<Q8Activation::A16, 56, 256, false,       \
                                      false>),                                 \
                 448, 256)                                                     \
Q8_DECODE_AUXILIARY(decode_linear_q8_n256_up_silu_m56##Suffix, Param, gate,    \
                    (q8_mpp_tile_batched<Q8Activation::A16, 56, 256, false,    \
                                         false, 256, true>),                    \
                    448, 256)                                                  \
Q8_DECODE_SLABI_AFFINE(decode_linear_q8_n256_m64_slabi##Suffix, Param,         \
                       (q8_mpp_tile_batched<Q8Activation::A16, 32, 256, false, \
                                            false>),                          \
                       256, 256)                                               \
Q8_DECODE_SLABI_AUXILIARY(decode_linear_q8_n256_up_silu_m64_slabi##Suffix,     \
                          Param, gate,                                        \
                          (q8_mpp_tile_batched<Q8Activation::A16, 32, 256,    \
                                               false, false, 256, true>),      \
                          256, 256)                                            \
Q8_DECODE_DUAL_AFFINE(decode_linear_q8_n256_d40##Suffix, Param, 8, 256, 8)   \
Q8_DECODE_DUAL_SILU(decode_linear_q8_n256_up_silu_d40##Suffix, Param, 8, 256, 8) \
Q8_DECODE_DUAL_AFFINE(decode_linear_q8_n256_d48##Suffix, Param, 16, 256, 8)   \
Q8_DECODE_DUAL_SILU(decode_linear_q8_n256_up_silu_d48##Suffix, Param, 16, 256, 8) \
Q8_DECODE_DUAL_AFFINE(decode_linear_q8_n256_d56##Suffix, Param, 24, 256, 8)   \
Q8_DECODE_DUAL_SILU(decode_linear_q8_n256_up_silu_d56##Suffix, Param, 24, 256, 8) \
Q8_DECODE_DUAL_AFFINE(decode_linear_q8_n256_d64##Suffix, Param, 32, 256, 8)   \
Q8_DECODE_DUAL_SILU(decode_linear_q8_n256_up_silu_d64##Suffix, Param, 32, 256, 8) \
Q8_DECODE_DUAL_AFFINE(decode_linear_q8_n128_d40##Suffix, Param, 8, 128, 8)   \
Q8_DECODE_DUAL_SILU(decode_linear_q8_n128_up_silu_d40##Suffix, Param, 8, 128, 8) \
Q8_DECODE_DUAL_AFFINE(decode_linear_q8_n128_d48##Suffix, Param, 16, 128, 8)   \
Q8_DECODE_DUAL_SILU(decode_linear_q8_n128_up_silu_d48##Suffix, Param, 16, 128, 8) \
Q8_DECODE_DUAL_AFFINE(decode_linear_q8_n128_d56##Suffix, Param, 24, 128, 8)   \
Q8_DECODE_DUAL_SILU(decode_linear_q8_n128_up_silu_d56##Suffix, Param, 24, 128, 8) \
Q8_DECODE_DUAL_AFFINE(decode_linear_q8_n128_d64##Suffix, Param, 32, 128, 8)   \
Q8_DECODE_DUAL_SILU(decode_linear_q8_n128_up_silu_d64##Suffix, Param, 32, 128, 8) \
Q8_DECODE_DUAL_AFFINE(decode_linear_q8_n256_d40_s16##Suffix, Param, 8, 256, 16) \
Q8_DECODE_DUAL_SILU(decode_linear_q8_n256_up_silu_d40_s16##Suffix, Param, 8, 256, 16) \
Q8_DECODE_DUAL_AFFINE(decode_linear_q8_n256_d48_s16##Suffix, Param, 16, 256, 16) \
Q8_DECODE_DUAL_SILU(decode_linear_q8_n256_up_silu_d48_s16##Suffix, Param, 16, 256, 16) \
Q8_DECODE_DUAL_AFFINE(decode_linear_q8_n256_d56_s16##Suffix, Param, 24, 256, 16) \
Q8_DECODE_DUAL_SILU(decode_linear_q8_n256_up_silu_d56_s16##Suffix, Param, 24, 256, 16) \
Q8_DECODE_DUAL_AFFINE(decode_linear_q8_n256_d64_s16##Suffix, Param, 32, 256, 16) \
Q8_DECODE_DUAL_SILU(decode_linear_q8_n256_up_silu_d64_s16##Suffix, Param, 32, 256, 16) \
Q8_DECODE_ASLAB_AFFINE(decode_linear_q8_n256_a40##Suffix, Param, 8, 256) \
Q8_DECODE_ASLAB_SILU(decode_linear_q8_n256_up_silu_a40##Suffix, Param, 8, 256) \
Q8_DECODE_ASLAB_AFFINE(decode_linear_q8_n256_a48##Suffix, Param, 16, 256) \
Q8_DECODE_ASLAB_SILU(decode_linear_q8_n256_up_silu_a48##Suffix, Param, 16, 256) \
Q8_DECODE_ASLAB_AFFINE(decode_linear_q8_n256_a56##Suffix, Param, 24, 256) \
Q8_DECODE_ASLAB_SILU(decode_linear_q8_n256_up_silu_a56##Suffix, Param, 24, 256) \
Q8_DECODE_ASLAB_AFFINE(decode_linear_q8_n256_a64##Suffix, Param, 32, 256) \
Q8_DECODE_ASLAB_SILU(decode_linear_q8_n256_up_silu_a64##Suffix, Param, 32, 256) \
Q8_DECODE_LOOP_AFFINE(decode_linear_q8_n128_m64_loop##Suffix, Param,           \
                      (q8_mpp_tile_batched<Q8Activation::A16, 32, 128, false,  \
                                           false, 256>),                      \
                      256, 128)                                                \
Q8_DECODE_LOOP_AFFINE(decode_linear_q8_n256_m64_loop##Suffix, Param,           \
                      (q8_mpp_tile_batched<Q8Activation::A16, 32, 256, false,  \
                                           false>),                           \
                      256, 256)                                                \
Q8_DECODE_LOOP_AUXILIARY(decode_linear_q8_n128_residual_m64_loop##Suffix,      \
                         Param, residual,                                     \
                         (q8_mpp_tile_batched<Q8Activation::A16, 32, 128,     \
                                              false, true, 256>),              \
                         256, 128)                                              \
Q8_DECODE_LOOP_AUXILIARY(decode_linear_q8_n64_residual_m64_loop##Suffix,       \
                         Param, residual,                                     \
                         (q8_mpp_tile_batched<Q8Activation::A16, 32, 64,      \
                                              false, true, 256>),              \
                         256, 64)                                               \
Q8_DECODE_LOOP_AUXILIARY(decode_linear_q8_n256_up_silu_m64_loop##Suffix,       \
                         Param, gate,                                         \
                         (q8_mpp_tile_batched<Q8Activation::A16, 32, 256,     \
                                              false, false, 256, true>),       \
                         256, 256)

Q8_A16_DECODE_KERNELS(, float)
Q8_A16_DECODE_KERNELS(_bf16p, bfloat)
#undef Q8_A16_DECODE_KERNELS

// A8: int8 activations x int8 weights; exact int32 dots with fp32 epilogues.
Q8A8_DECODE_AFFINE(decode_linear_q8a8_n128,
                   (q8_mpp_tile<Q8Activation::A8, 128, false, false, 256>),
                   128)
Q8A8_DECODE_AFFINE(decode_linear_q8a8_n128_m16,
                   (q8_mpp_tile_batched<Q8Activation::A8, 16, 128, false,
                                        false, 256>),
                   128)
Q8A8_DECODE_AFFINE(decode_linear_q8a8_n128_m24,
                   (q8_mpp_tile_batched<Q8Activation::A8, 24, 128, false,
                                        false, 256>),
                   128)
Q8A8_DECODE_AFFINE(decode_linear_q8a8_n128_m24_sg4,
                   (q8_mpp_tile_batched<Q8Activation::A8, 24, 128, false,
                                        false, 256, false, 4>),
                   128)
Q8A8_DECODE_AFFINE(decode_linear_q8a8_n256_m16,
                   (q8_mpp_tile_batched<Q8Activation::A8, 16, 256, false,
                                        false>),
                   256)
Q8A8_DECODE_AFFINE(decode_linear_q8a8_n256_m24,
                   (q8_mpp_tile_batched<Q8Activation::A8, 24, 256, false,
                                        false>),
                   256)
Q8A8_DECODE_AFFINE(decode_linear_q8a8_n256,
                   (q8_mpp_tile<Q8Activation::A8, 256, false, false>), 256)
Q8A8_DECODE_AFFINE(decode_linear_q8a8_n128_paired,
                   (q8_mpp_tile<Q8Activation::A8, 128, false, false, 256,
                                true>),
                   128)
Q8A8_DECODE_AFFINE(decode_linear_q8a8_n256_paired_sg4,
                   (q8_mpp_tile<Q8Activation::A8, 256, false, false, 256,
                                true, 4>),
                   256)
Q8A8_DECODE_AUXILIARY(decode_linear_q8a8_n128_residual_paired, residual,
                      (q8_mpp_tile<Q8Activation::A8, 128, false, true, 256,
                                   true>),
                      128)
Q8A8_DECODE_AUXILIARY(decode_linear_q8a8_n128_residual, residual,
                      (q8_mpp_tile<Q8Activation::A8, 128, false, true, 256>),
                      128)
Q8A8_DECODE_GATE_UP(decode_linear_q8a8_n256_gate_up,
                    (q8_mpp_tile<Q8Activation::A8, 256, true, false>), 256)
Q8A8_DECODE_AUXILIARY(decode_linear_q8a8_n128_residual_m16, residual,
                      (q8_mpp_tile_batched<Q8Activation::A8, 16, 128, false,
                                           true, 256>),
                      128)
Q8A8_DECODE_AUXILIARY(decode_linear_q8a8_n128_residual_m24, residual,
                      (q8_mpp_tile_batched<Q8Activation::A8, 24, 128, false,
                                           true, 256>),
                      128)
Q8A8_DECODE_AUXILIARY(decode_linear_q8a8_n128_residual_m24_sg4, residual,
                      (q8_mpp_tile_batched<Q8Activation::A8, 24, 128, false,
                                           true, 256, false, 4>),
                      128)
Q8A8_DECODE_GATE_UP(decode_linear_q8a8_n256_gate_up_m16,
                    (q8_mpp_tile_batched<Q8Activation::A8, 16, 256, true,
                                         false>),
                    256)
Q8A8_DECODE_AFFINE(decode_linear_q8a8_n128_m32,
                   (q8_mpp_tile_batched<Q8Activation::A8, 32, 128, false,
                                        false, 256>),
                   128)
Q8A8_DECODE_AFFINE(decode_linear_q8a8_n256_m32,
                   (q8_mpp_tile_batched<Q8Activation::A8, 32, 256, false,
                                        false>),
                   256)
Q8A8_DECODE_AUXILIARY(decode_linear_q8a8_n128_residual_m32, residual,
                      (q8_mpp_tile_batched<Q8Activation::A8, 32, 128, false,
                                           true, 256>),
                      128)
Q8A8_DECODE_AUXILIARY(decode_linear_q8a8_n256_up_silu_m32, gate,
                      (q8_mpp_tile_batched<Q8Activation::A8, 32, 256, false,
                                           false, 256, true>),
                      256)
Q8A8_DECODE_AUXILIARY(decode_linear_q8a8_n256_up_silu_m24, gate,
                      (q8_mpp_tile_batched<Q8Activation::A8, 24, 256, false,
                                           false, 256, true>),
                      256)

#undef Q8_DECODE_AFFINE
#undef Q8_DECODE_AUXILIARY
#undef Q8_DECODE_GATE_UP
#undef Q8_DECODE_SLAB_AFFINE
#undef Q8_DECODE_SLAB_AUXILIARY
#undef Q8_DECODE_SLABI_AFFINE
#undef Q8_DECODE_SLABI_AUXILIARY
#undef Q8_DECODE_ASLAB_AFFINE
#undef Q8_DECODE_ASLAB_SILU
#define Q8_DECODE_DUAL_AFFINE_SEQ(Name, Param, Rows1, TileN, SG)                     \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device int8_t *weights [[buffer(1)]],                       \
                   device Param *scales [[buffer(2)]],                         \
                   device Param *biases [[buffer(3)]],                         \
                   device bfloat *output [[buffer(4)]],                        \
                   constant Q8DecodeParams &params [[buffer(5)]],              \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float sums_0[256];                                             \
    threadgroup float sums_1[8 * Rows1];                                       \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      q8_mpp_tile_dual<Rows1, TileN, false, TileN, Param, SG, true>(                            \
          input, weights, scales, biases, output, output, params.output_size,  \
          params.input_size, sums_0, sums_1, tile * TileN, simd_lane,          \
          simd_group);                                                         \
    }                                                                          \
  }

#define Q8_DECODE_DUAL_SILU_SEQ(Name, Param, Rows1, TileN, SG)                     \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device int8_t *weights [[buffer(1)]],                       \
                   device Param *scales [[buffer(2)]],                         \
                   device Param *biases [[buffer(3)]],                         \
                   device bfloat *gate [[buffer(4)]],                          \
                   device bfloat *output [[buffer(5)]],                        \
                   constant Q8DecodeParams &params [[buffer(6)]],              \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float sums_0[256];                                             \
    threadgroup float sums_1[8 * Rows1];                                       \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      q8_mpp_tile_dual<Rows1, TileN, true, TileN, Param, SG, true>(                             \
          input, weights, scales, biases, output, gate, params.output_size,    \
          params.input_size, sums_0, sums_1, tile * TileN, simd_lane,          \
          simd_group);                                                         \
    }                                                                          \
  }

// Asymmetric slab-pair candidates (campaign C-family): one dispatch, 1D grid
// of 2 * persistent_groups threadgroups; even gid runs the 32-row fragment,
// odd gid runs the Rows1-row tail fragment for the same weight tile index, so
// the tail's DRAM fetch overlaps the main fragment's stream instead of
// serializing a full second pass. Per-fragment math is identical to the m32 /
// m<Rows1> batched tiles, so active rows are bit-identical.

Q8_DECODE_DUAL_AFFINE_SEQ(decode_linear_q8_n256_d40_seq_bf16p, bfloat, 8, 256, 8)
Q8_DECODE_DUAL_SILU_SEQ(decode_linear_q8_n256_up_silu_d40_seq_bf16p, bfloat, 8, 256, 8)
Q8_DECODE_DUAL_AFFINE_SEQ(decode_linear_q8_n256_d48_seq_bf16p, bfloat, 16, 256, 8)
Q8_DECODE_DUAL_SILU_SEQ(decode_linear_q8_n256_up_silu_d48_seq_bf16p, bfloat, 16, 256, 8)
#undef Q8_DECODE_DUAL_AFFINE_SEQ
#undef Q8_DECODE_DUAL_SILU_SEQ

#undef Q8_DECODE_DUAL_AFFINE
#undef Q8_DECODE_DUAL_SILU
#undef Q8_DECODE_LOOP_AFFINE
#undef Q8_DECODE_LOOP_AUXILIARY
#undef Q8A8_DECODE_AFFINE
#undef Q8A8_DECODE_AUXILIARY
#undef Q8A8_DECODE_GATE_UP
