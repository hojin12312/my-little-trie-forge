#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/q4_mpp_tiles.h"
#include "metal/kernels/common/q8_quantize.h"

// Dense Q8 (8-bit, group 64, StorageN=256) tiles for QKV/MLP projections.
// Weights are signed int8: the package's affine uint8 codes are shifted by
// -128 at pack time and the offset folded into the fp32 bias, so every
// matmul is signed x signed (MPP has no i8 x ui8 path). Scales and the
// folded biases are fp32, indexed [(n / 256) * groups + g] * 256 + n % 256.
// A16 tiles also take the compact bf16 (s, z) form at the same index
// (q8_folded_bias).
//
// Two activation modes share the tile structure:
//   A16: bf16 rows; the epilogue is acc += partial * s + xsum * b', exactly
//        the Q4 form with int8 weights and fp32 parameters.
//   A8:  int8 rows produced by group-64 symmetric quantization; each group's
//        float2 term is {sx, sx * qxsum} and the epilogue is
//        acc += float(idot) * (sx * s) + sxq * b'. The int32 dot is exact,
//        so all error is activation quantization plus fp32 recombination.
enum class Q8Activation : ushort { A16, A8 };

// One A8 group step in a single fixed rounding order. Written as a plain
// expression, fast math lets the compiler pick a different contraction for
// different cooperative tensor slots, so an identical row rounded differently
// at tile rows 0-7, 8-15 and 16-31 and packed prefill lanes holding the same
// prompt disagreed bitwise. Explicit fma calls pin the order for every slot.
inline float q8a8_accumulate(float accumulated, int partial, float2 ax,
                             float scale, float bias) {
  return fma(float(partial), ax.x * scale, fma(ax.y, bias, accumulated));
}

// One A16 group step, pinned the same way. As a plain expression the
// decode_linear_q8_n128_m24 and _residual_m24 tiles contracted it as
// fma(p, s, fma(t, b, acc)) while every other decode tile compiled
// fma(t, b, fma(p, s, acc)), so one lane's rows rounded differently at batch
// width 3 than at widths 1, 2 and 4 and greedy decode forked. This is the
// order the other tiles already used, so only the m24 tiles change. Every
// A16 prefill tile also compiled this order (linear-q8-plan
// --prefill-epilogue-order), so pinning prefill with it changed no bytes.
inline float q8a16_accumulate(float accumulated, float partial, float term,
                              float scale, float bias) {
  return fma(term, bias, fma(partial, scale, accumulated));
}

// Decode tiles read the group parameters in one of two layouts at the same
// index. The package's fp32 form stores s and the folded b' = z + 128 * s.
// The compact form stores the checkpoint's bf16 s and zero point z, and the
// tile rebuilds b' as fma(128, s, z): 128 * s is exact, so the single rounding
// equals the converter's fp32 z + 128 * s and b' matches the fp32 package bit
// for bit. Both values are formed before the pinned epilogue helper sees them.
inline float q8_scale(device const float *scales, ulong parameter) {
  return scales[parameter];
}
inline float q8_scale(device const bfloat *scales, ulong parameter) {
  return float(scales[parameter]);
}
inline float q8_folded_bias(device const float *, device const float *biases,
                            ulong parameter) {
  return biases[parameter];
}
inline float q8_folded_bias(device const bfloat *scales,
                            device const bfloat *zeros, ulong parameter) {
  return fma(128.0f, float(scales[parameter]), float(zeros[parameter]));
}

template <Q8Activation> struct Q8Operand;
template <> struct Q8Operand<Q8Activation::A16> {
  using Input = bfloat;
  using Term = float;
  using Partial = float;
  // fp32 sums stage 256 groups in the 32 KiB threadgroup budget.
  enum : ushort { TermBatch = 256 };
};
template <> struct Q8Operand<Q8Activation::A8> {
  using Input = int8_t;
  using Term = float2;
  using Partial = int;
  // float2 terms stage 128 groups in the same 32 KiB budget.
  enum : ushort { TermBatch = 128 };
};

// Decode tiles: persistent threadgroups stride over TileN-wide output tiles.
// A16 computes bf16 input sums in kernel (the Q4 cadence: four staged groups
// in two threadgroup regions). A8 reads its precomputed terms straight from
// device memory, so the group loop needs no staging or barriers.
template <Q8Activation A, ushort TileN, bool GateUp, bool AddResidual,
          ushort StorageN = TileN, bool Pipelined = false, ushort Simdgroups = 8,
          typename P = float>
inline void q8_mpp_tile(
    device typename Q8Operand<A>::Input *input, device int8_t *weights_0,
    device P *scales_0, device P *biases_0, device bfloat *output_0,
    device int8_t *weights_1, device P *scales_1, device P *biases_1,
    device bfloat *residual, uint output_size, uint input_size,
    threadgroup typename Q8Operand<A>::Term *input_terms,
    device const typename Q8Operand<A>::Term *terms, uint output_origin,
    uint simd_lane, uint simd_group) {
  using Partial = typename Q8Operand<A>::Partial;
  auto a = tensor(input, dextents<int, 2>{int(input_size), 8},
                  array<int, 2>{1, int(input_size)});
  constexpr auto descriptor =
      matmul2d_descriptor(8, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  auto a0 = a.template slice<64, 8>(0, 0);
  uint quant_groups = input_size / 64;
  uint tile = output_origin / StorageN;
  uint tile_offset = output_origin % StorageN;
  device int8_t *tile_weights_0 =
      weights_0 + ulong(tile) * quant_groups * StorageN * 64;
  device int8_t *tile_weights_1 =
      weights_1 + ulong(tile) * quant_groups * StorageN * 64;
  tensor<device int8_t, dextents<int, 2>, tensor_inline> first_b0(
      tile_weights_0 + tile_offset * 64, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  tensor<device int8_t, dextents<int, 2>, tensor_inline> first_b1(
      tile_weights_1 + tile_offset * 64, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  auto b00 = first_b0.slice<64, TileN>(0, 0);
  auto b10 = first_b1.slice<64, TileN>(0, 0);
  auto accumulated_0 = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b00), float>();
  auto accumulated_1 = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b10), float>();
  const bool fullyOccupied =
      uint(accumulated_0.get_capacity()) * (uint(Simdgroups) * 32u) ==
      8u * TileN;
  const auto traversal =
      fullyOccupied ? Q4Traversal::All : q4_traversal(accumulated_0);
  q4_visit(accumulated_0, traversal, [&](ushort i) {
    accumulated_0[i] = 0.0f;
    if constexpr (GateUp)
      accumulated_1[i] = 0.0f;
  });
  using ASlice = decltype(a0);
  using BSlice = decltype(b00);
  using PartialTensor = decltype(operation.template
      get_destination_cooperative_tensor<ASlice, BSlice, Partial>());

  if constexpr (A == Q8Activation::A16) {
    q4_store_input_sums<8, Simdgroups>(input, input_size, 0, input_terms, 0,
                                     simd_lane, simd_group);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  auto run_group = [&](uint quant_group, thread PartialTensor &partial_0,
                       thread PartialTensor &partial_1) {
    uint input_origin = quant_group * 64;
    auto a_slice = a.template slice<64, 8>(input_origin, 0);
    device int8_t *group_weights_0 =
        tile_weights_0 + (ulong(quant_group) * StorageN + tile_offset) * 64;
    tensor<device int8_t, dextents<int, 2>, tensor_inline> b0(
        group_weights_0, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b0_slice = b0.slice<64, TileN>(0, 0);
    operation.run(a_slice, b0_slice, partial_0);
    device int8_t *group_weights_1 =
        tile_weights_1 + (ulong(quant_group) * StorageN + tile_offset) * 64;
    tensor<device int8_t, dextents<int, 2>, tensor_inline> b1(
        group_weights_1, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b1_slice = b1.slice<64, TileN>(0, 0);
    if constexpr (GateUp)
      operation.run(a_slice, b1_slice, partial_1);
  };
  auto finish_group = [&](uint quant_group, thread PartialTensor &partial_0,
                          thread PartialTensor &partial_1) {
    q4_visit(accumulated_0, traversal,
             [&](ushort i) __attribute__((always_inline)) {
      auto index = accumulated_0.get_multidimensional_index(i);
      uint row = index[1];
      ulong parameter = (ulong(tile) * quant_groups + quant_group) * StorageN +
                        tile_offset + index[0];
      if constexpr (A == Q8Activation::A16) {
        const uint sum_offset =
            ((quant_group >> 2) & 1) * 32 + (quant_group & 3) * 8;
        const float term = input_terms[sum_offset + row];
        accumulated_0[i] = q8a16_accumulate(
            accumulated_0[i], partial_0[i], term,
            q8_scale(scales_0, parameter),
            q8_folded_bias(scales_0, biases_0, parameter));
        if constexpr (GateUp)
          accumulated_1[i] = q8a16_accumulate(
              accumulated_1[i], partial_1[i], term,
              q8_scale(scales_1, parameter),
              q8_folded_bias(scales_1, biases_1, parameter));
      } else {
        const float2 ax = terms[quant_group * 8 + row];
        static_assert(is_same_v<P, float>,
                      "compact Q8 parameters are A16 only");
        accumulated_0[i] =
            q8a8_accumulate(accumulated_0[i], partial_0[i], ax,
                            scales_0[parameter], biases_0[parameter]);
        if constexpr (GateUp)
          accumulated_1[i] =
              q8a8_accumulate(accumulated_1[i], partial_1[i], ax,
                              scales_1[parameter], biases_1[parameter]);
      }
    });
    if constexpr (A == Q8Activation::A16) {
      if ((quant_group & 3) == 3 && quant_group + 1 < quant_groups) {
        uint next_group = (quant_group + 1) >> 2;
        q4_store_input_sums<8, Simdgroups>(input, input_size,
                                           quant_group * 64 + 64, input_terms,
                                           (next_group & 1) * 32, simd_lane,
                                           simd_group);
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  };
  if constexpr (Pipelined) {
    uint quant_group = 0;
    for (; quant_group + 1 < quant_groups; quant_group += 2) {
      PartialTensor first_0, second_0;
      PartialTensor first_1, second_1;
      run_group(quant_group, first_0, second_0);
      run_group(quant_group + 1, first_1, second_1);
      finish_group(quant_group, first_0, second_0);
      finish_group(quant_group + 1, first_1, second_1);
    }
    if (quant_group < quant_groups) {
      PartialTensor partial_0, partial_1;
      run_group(quant_group, partial_0, partial_1);
      finish_group(quant_group, partial_0, partial_1);
    }
  } else {
    for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
      PartialTensor partial_0, partial_1;
      run_group(quant_group, partial_0, partial_1);
      finish_group(quant_group, partial_0, partial_1);
    }
  }

  q4_visit(accumulated_0, traversal, [&](ushort i) {
    auto index = accumulated_0.get_multidimensional_index(i);
    uint output_index = index[1] * output_size + output_origin + index[0];
    float value;
    if constexpr (GateUp) {
      float gate = float(bfloat(accumulated_0[i]));
      float up = float(bfloat(accumulated_1[i]));
      value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * up;
    } else {
      value = float(bfloat(accumulated_0[i]));
    }
    if constexpr (AddResidual)
      value += float(residual[output_index]);
    output_0[output_index] = bfloat(value);
  });
  // A8 shares no threadgroup state between tiles; the A16 sums scratch needs
  // the same end-of-tile barrier as the Q4 kernel.
  if constexpr (A == Q8Activation::A16)
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Pipelined issues the MPP runs of groups g and g + 1 before either group's
// epilogue (the one-lane paired tile's cadence) and still folds them in group
// order, so each column keeps its sequential chain and the output bytes of
// the plain loop. It exists for the A16 single-projection form only.
template <Q8Activation A, ushort Rows, ushort TileN, bool GateUp,
          bool AddResidual, ushort StorageN = TileN,
          bool MultiplySiluGate = false, ushort Simdgroups = 8,
          typename P = float, bool Pipelined = false>
inline void q8_mpp_tile_batched(
    device typename Q8Operand<A>::Input *input, device int8_t *weights_0,
    device P *scales_0, device P *biases_0, device bfloat *output_0,
    device int8_t *weights_1, device P *scales_1, device P *biases_1,
    device bfloat *residual, uint output_size, uint input_size,
    threadgroup typename Q8Operand<A>::Term *input_terms,
    device const typename Q8Operand<A>::Term *terms, uint output_origin,
    uint simd_lane, uint simd_group) {
  using Partial = typename Q8Operand<A>::Partial;
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows},
                  array<int, 2>{1, int(input_size)});
  constexpr auto descriptor =
      matmul2d_descriptor(Rows, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  uint quant_groups = input_size / 64;
  uint tile = output_origin / StorageN;
  uint tile_offset = output_origin % StorageN;
  device int8_t *tile_weights_0 =
      weights_0 + ulong(tile) * quant_groups * StorageN * 64;
  device int8_t *tile_weights_1 =
      weights_1 + ulong(tile) * quant_groups * StorageN * 64;
  tensor<device int8_t, dextents<int, 2>, tensor_inline> first_b0(
      tile_weights_0 + tile_offset * 64, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  tensor<device int8_t, dextents<int, 2>, tensor_inline> first_b1(
      tile_weights_1 + tile_offset * 64, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  auto a0 = a.template slice<64, Rows>(0, 0);
  auto b00 = first_b0.slice<64, TileN>(0, 0);
  auto b10 = first_b1.slice<64, TileN>(0, 0);
  auto accumulated_0 = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b00), float>();
  auto accumulated_1 = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b10), float>();
  const bool fullyOccupied =
      uint(accumulated_0.get_capacity()) * (uint(Simdgroups) * 32u) ==
      uint(Rows) * TileN;
  const auto traversal =
      fullyOccupied ? Q4Traversal::All : q4_traversal(accumulated_0);
  q4_visit(accumulated_0, traversal, [&](ushort i) {
    accumulated_0[i] = 0.0f;
    if constexpr (GateUp)
      accumulated_1[i] = 0.0f;
  });
  if constexpr (A == Q8Activation::A16) {
    q4_store_input_sums<Rows, Simdgroups>(input, input_size, 0, input_terms, 0,
                                        simd_lane, simd_group);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if constexpr (Pipelined) {
    static_assert(A == Q8Activation::A16 && !GateUp,
                  "pipelined batched Q8 tiles are A16 single projections");
    using PartialTensor = decltype(operation.template
        get_destination_cooperative_tensor<decltype(a0), decltype(b00),
                                           Partial>());
    auto run_group = [&](uint quant_group, thread PartialTensor &partial) {
      auto a_slice = a.template slice<64, Rows>(quant_group * 64, 0);
      tensor<device int8_t, dextents<int, 2>, tensor_inline> b0(
          tile_weights_0 + (ulong(quant_group) * StorageN + tile_offset) * 64,
          dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
      auto b0_slice = b0.slice<64, TileN>(0, 0);
      operation.run(a_slice, b0_slice, partial);
    };
    auto finish_group = [&](uint quant_group, thread PartialTensor &partial) {
      q4_visit(accumulated_0, traversal,
               [&](ushort i) __attribute__((always_inline)) {
        auto index = accumulated_0.get_multidimensional_index(i);
        uint row = index[1];
        ulong parameter =
            (ulong(tile) * quant_groups + quant_group) * StorageN +
            tile_offset + index[0];
        const uint sum_offset =
            ((quant_group >> 2) & 1) * (4 * Rows) + (quant_group & 3) * Rows;
        accumulated_0[i] = q8a16_accumulate(
            accumulated_0[i], partial[i], input_terms[sum_offset + row],
            q8_scale(scales_0, parameter),
            q8_folded_bias(scales_0, biases_0, parameter));
      });
      if ((quant_group & 3) == 3 && quant_group + 1 < quant_groups) {
        uint next_group = (quant_group + 1) >> 2;
        q4_store_input_sums<Rows, Simdgroups>(
            input, input_size, quant_group * 64 + 64, input_terms,
            (next_group & 1) * (4 * Rows), simd_lane, simd_group);
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    };
    uint quant_group = 0;
    for (; quant_group + 1 < quant_groups; quant_group += 2) {
      PartialTensor first, second;
      run_group(quant_group, first);
      run_group(quant_group + 1, second);
      finish_group(quant_group, first);
      finish_group(quant_group + 1, second);
    }
    if (quant_group < quant_groups) {
      PartialTensor partial;
      run_group(quant_group, partial);
      finish_group(quant_group, partial);
    }
  } else
  for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
    uint input_origin = quant_group * 64;
    auto a_slice = a.template slice<64, Rows>(input_origin, 0);
    device int8_t *group_weights_0 =
        tile_weights_0 + (ulong(quant_group) * StorageN + tile_offset) * 64;
    tensor<device int8_t, dextents<int, 2>, tensor_inline> b0(
        group_weights_0, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b0_slice = b0.slice<64, TileN>(0, 0);
    auto partial_0 = operation.template get_destination_cooperative_tensor<
        decltype(a_slice), decltype(b0_slice), Partial>();
    operation.run(a_slice, b0_slice, partial_0);
    device int8_t *group_weights_1 =
        tile_weights_1 + (ulong(quant_group) * StorageN + tile_offset) * 64;
    tensor<device int8_t, dextents<int, 2>, tensor_inline> b1(
        group_weights_1, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b1_slice = b1.slice<64, TileN>(0, 0);
    auto partial_1 = operation.template get_destination_cooperative_tensor<
        decltype(a_slice), decltype(b1_slice), Partial>();
    if constexpr (GateUp)
      operation.run(a_slice, b1_slice, partial_1);
    q4_visit(accumulated_0, traversal,
             [&](ushort i) __attribute__((always_inline)) {
      auto index = accumulated_0.get_multidimensional_index(i);
      uint row = index[1];
      ulong parameter = (ulong(tile) * quant_groups + quant_group) * StorageN +
                        tile_offset + index[0];
      if constexpr (A == Q8Activation::A16) {
        const uint sum_offset =
            ((quant_group >> 2) & 1) * (4 * Rows) + (quant_group & 3) * Rows;
        const float term = input_terms[sum_offset + row];
        accumulated_0[i] = q8a16_accumulate(
            accumulated_0[i], partial_0[i], term,
            q8_scale(scales_0, parameter),
            q8_folded_bias(scales_0, biases_0, parameter));
        if constexpr (GateUp)
          accumulated_1[i] = q8a16_accumulate(
              accumulated_1[i], partial_1[i], term,
              q8_scale(scales_1, parameter),
              q8_folded_bias(scales_1, biases_1, parameter));
      } else {
        const float2 ax = terms[quant_group * Rows + row];
        static_assert(is_same_v<P, float>,
                      "compact Q8 parameters are A16 only");
        accumulated_0[i] =
            q8a8_accumulate(accumulated_0[i], partial_0[i], ax,
                            scales_0[parameter], biases_0[parameter]);
        if constexpr (GateUp)
          accumulated_1[i] =
              q8a8_accumulate(accumulated_1[i], partial_1[i], ax,
                              scales_1[parameter], biases_1[parameter]);
      }
    });
    if constexpr (A == Q8Activation::A16) {
      if ((quant_group & 3) == 3 && quant_group + 1 < quant_groups) {
        uint next_group = (quant_group + 1) >> 2;
        q4_store_input_sums<Rows, Simdgroups>(input, input_size,
                                            input_origin + 64, input_terms,
                                            (next_group & 1) * (4 * Rows),
                                            simd_lane, simd_group);
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }
  q4_visit(accumulated_0, traversal, [&](ushort i) {
    auto index = accumulated_0.get_multidimensional_index(i);
    uint output_index = index[1] * output_size + output_origin + index[0];
    float value;
    if constexpr (GateUp) {
      float gate = float(bfloat(accumulated_0[i]));
      float up = float(bfloat(accumulated_1[i]));
      value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * up;
    } else if constexpr (MultiplySiluGate) {
      float gate = float(residual[output_index]);
      value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) *
              float(bfloat(accumulated_0[i]));
    } else {
      value = float(bfloat(accumulated_0[i]));
    }
    if constexpr (AddResidual)
      value += float(residual[output_index]);
    output_0[output_index] = bfloat(value);
  });
  if constexpr (A == Q8Activation::A16)
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Shared-weight dual-fragment decode tile (wide-decode shared-weight
// campaign): one threadgroup computes a 32-row fragment and a Rows1-row tail
// fragment against the same weight group slice in one k loop, so each 64-wide
// Both fragments reference the same immutable group slice inside one TG.
// Actual cache/DRAM transactions require counters and are not inferred here.
// Each fragment keeps the M32 / M<Rows1> cooperative scope, its own MPP
// operation and its own per-row accumulation chain, so active rows are
// bit-identical to the standalone batched tiles. A16 single projection only.
template <ushort Rows1, ushort TileN, bool MultiplySiluGate,
          ushort StorageN = TileN, typename P = float, ushort Simdgroups = 8, bool SequentialDots = false>
inline void q8_mpp_tile_dual(
    device bfloat *input, device int8_t *weights, device P *scales,
    device P *biases, device bfloat *output, device bfloat *gate,
    uint output_size, uint input_size, threadgroup float *sums_0,
    threadgroup float *sums_1, uint output_origin, uint simd_lane,
    uint simd_group) {
  constexpr ushort Rows0 = 32;
  auto a_0 = tensor(input, dextents<int, 2>{int(input_size), Rows0},
                    array<int, 2>{1, int(input_size)});
  device bfloat *input_1 = input + ulong(Rows0) * input_size;
  auto a_1 = tensor(input_1, dextents<int, 2>{int(input_size), Rows1},
                    array<int, 2>{1, int(input_size)});
  constexpr auto descriptor_0 =
      matmul2d_descriptor(Rows0, TileN, 64, false, true, false);
  constexpr auto descriptor_1 =
      matmul2d_descriptor(Rows1, TileN, 64, false, true, false);
  matmul2d<descriptor_0, execution_simdgroups<Simdgroups>> operation_0;
  matmul2d<descriptor_1, execution_simdgroups<Simdgroups>> operation_1;
  uint quant_groups = input_size / 64;
  uint tile = output_origin / StorageN;
  uint tile_offset = output_origin % StorageN;
  device int8_t *tile_weights =
      weights + ulong(tile) * quant_groups * StorageN * 64;
  tensor<device int8_t, dextents<int, 2>, tensor_inline> first_b(
      tile_weights + tile_offset * 64, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  auto b_first = first_b.slice<64, TileN>(0, 0);
  auto a0_first = a_0.template slice<64, Rows0>(0, 0);
  auto a1_first = a_1.template slice<64, Rows1>(0, 0);
  auto accumulated_0 = operation_0.template get_destination_cooperative_tensor<
      decltype(a0_first), decltype(b_first), float>();
  auto accumulated_1 = operation_1.template get_destination_cooperative_tensor<
      decltype(a1_first), decltype(b_first), float>();
  const bool full_0 = uint(accumulated_0.get_capacity()) *
                          (uint(Simdgroups) * 32u) == uint(Rows0) * TileN;
  const bool full_1 = uint(accumulated_1.get_capacity()) *
                          (uint(Simdgroups) * 32u) == uint(Rows1) * TileN;
  const auto traversal_0 =
      full_0 ? Q4Traversal::All : q4_traversal(accumulated_0);
  const auto traversal_1 =
      full_1 ? Q4Traversal::All : q4_traversal(accumulated_1);
  q4_visit(accumulated_0, traversal_0, [&](ushort i) { accumulated_0[i] = 0.0f; });
  q4_visit(accumulated_1, traversal_1, [&](ushort i) { accumulated_1[i] = 0.0f; });
  q4_store_input_sums<Rows0, Simdgroups>(input, input_size, 0, sums_0, 0,
                                         simd_lane, simd_group);
  q4_store_input_sums<Rows1, Simdgroups>(input_1, input_size, 0, sums_1, 0,
                                         simd_lane, simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
    auto a0_slice = a_0.template slice<64, Rows0>(quant_group * 64, 0);
    auto a1_slice = a_1.template slice<64, Rows1>(quant_group * 64, 0);
    tensor<device int8_t, dextents<int, 2>, tensor_inline> b(
        tile_weights + (ulong(quant_group) * StorageN + tile_offset) * 64,
        dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b_slice = b.slice<64, TileN>(0, 0);
    if constexpr (SequentialDots) {
    {
    auto partial_0 = operation_0.template get_destination_cooperative_tensor<
        decltype(a0_slice), decltype(b_slice), float>();
    operation_0.run(a0_slice, b_slice, partial_0);
    q4_visit(accumulated_0, traversal_0,
             [&](ushort i) __attribute__((always_inline)) {
      auto index = accumulated_0.get_multidimensional_index(i);
      ulong parameter = (ulong(tile) * quant_groups + quant_group) * StorageN +
                        tile_offset + index[0];
      const uint sum_offset =
          ((quant_group >> 2) & 1) * (4 * Rows0) + (quant_group & 3) * Rows0;
      accumulated_0[i] = q8a16_accumulate(
          accumulated_0[i], partial_0[i], sums_0[sum_offset + index[1]],
          q8_scale(scales, parameter), q8_folded_bias(scales, biases, parameter));
    });
    }
    {
    auto partial_1 = operation_1.template get_destination_cooperative_tensor<
        decltype(a1_slice), decltype(b_slice), float>();
    operation_1.run(a1_slice, b_slice, partial_1);
    q4_visit(accumulated_1, traversal_1,
             [&](ushort i) __attribute__((always_inline)) {
      auto index = accumulated_1.get_multidimensional_index(i);
      ulong parameter = (ulong(tile) * quant_groups + quant_group) * StorageN +
                        tile_offset + index[0];
      const uint sum_offset =
          ((quant_group >> 2) & 1) * (4 * Rows1) + (quant_group & 3) * Rows1;
      accumulated_1[i] = q8a16_accumulate(
          accumulated_1[i], partial_1[i], sums_1[sum_offset + index[1]],
          q8_scale(scales, parameter), q8_folded_bias(scales, biases, parameter));
    });
    }
    } else {
    auto partial_0 = operation_0.template get_destination_cooperative_tensor<
        decltype(a0_slice), decltype(b_slice), float>();
    auto partial_1 = operation_1.template get_destination_cooperative_tensor<
        decltype(a1_slice), decltype(b_slice), float>();
    operation_0.run(a0_slice, b_slice, partial_0);
    operation_1.run(a1_slice, b_slice, partial_1);
    q4_visit(accumulated_0, traversal_0,
             [&](ushort i) __attribute__((always_inline)) {
      auto index = accumulated_0.get_multidimensional_index(i);
      ulong parameter = (ulong(tile) * quant_groups + quant_group) * StorageN +
                        tile_offset + index[0];
      const uint sum_offset =
          ((quant_group >> 2) & 1) * (4 * Rows0) + (quant_group & 3) * Rows0;
      accumulated_0[i] = q8a16_accumulate(
          accumulated_0[i], partial_0[i], sums_0[sum_offset + index[1]],
          q8_scale(scales, parameter), q8_folded_bias(scales, biases, parameter));
    });
    q4_visit(accumulated_1, traversal_1,
             [&](ushort i) __attribute__((always_inline)) {
      auto index = accumulated_1.get_multidimensional_index(i);
      ulong parameter = (ulong(tile) * quant_groups + quant_group) * StorageN +
                        tile_offset + index[0];
      const uint sum_offset =
          ((quant_group >> 2) & 1) * (4 * Rows1) + (quant_group & 3) * Rows1;
      accumulated_1[i] = q8a16_accumulate(
          accumulated_1[i], partial_1[i], sums_1[sum_offset + index[1]],
          q8_scale(scales, parameter), q8_folded_bias(scales, biases, parameter));
    });
    }
    if ((quant_group & 3) == 3 && quant_group + 1 < quant_groups) {
      uint next_group = (quant_group + 1) >> 2;
      q4_store_input_sums<Rows0, Simdgroups>(
          input, input_size, quant_group * 64 + 64, sums_0,
          (next_group & 1) * (4 * Rows0), simd_lane, simd_group);
      q4_store_input_sums<Rows1, Simdgroups>(
          input_1, input_size, quant_group * 64 + 64, sums_1,
          (next_group & 1) * (4 * Rows1), simd_lane, simd_group);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  q4_visit(accumulated_0, traversal_0, [&](ushort i) {
    auto index = accumulated_0.get_multidimensional_index(i);
    uint output_index = index[1] * output_size + output_origin + index[0];
    float value = float(bfloat(accumulated_0[i]));
    if constexpr (MultiplySiluGate) {
      float g = float(gate[output_index]);
      value = g / (1.0f + fast::exp2(-1.44269504089f * g)) * value;
    }
    output[output_index] = bfloat(value);
  });
  device bfloat *output_1 = output + ulong(Rows0) * output_size;
  device bfloat *gate_1 = gate + ulong(Rows0) * output_size;
  q4_visit(accumulated_1, traversal_1, [&](ushort i) {
    auto index = accumulated_1.get_multidimensional_index(i);
    uint output_index = index[1] * output_size + output_origin + index[0];
    float value = float(bfloat(accumulated_1[i]));
    if constexpr (MultiplySiluGate) {
      float g = float(gate_1[output_index]);
      value = g / (1.0f + fast::exp2(-1.44269504089f * g)) * value;
    }
    output_1[output_index] = bfloat(value);
  });
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Prefill tiles: 32-row tiles, precomputed per-(row, group) terms staged to
// threadgroup memory in TermBatch-group batches on eight simdgroups, or read
// from device memory directly on four. fp32 terms batch 256 groups; the A8
// float2 terms batch 128, both within the 32 KiB budget.
template <Q8Activation A, ushort TileM, ushort TileN, ushort Simdgroups,
          bool AddResidual, bool MultiplySiluGate, typename P = float,
          bool Pipelined = false, bool StageParameters = false,
          bool PackedParameters = false>
inline void q8_mpp_prefill_tile(
    device typename Q8Operand<A>::Input *input, device int8_t *weights,
    device P *scales, device P *biases, device bfloat *output,
    device bfloat *auxiliary, uint output_size, uint input_size,
    device const typename Q8Operand<A>::Term *precomputed_terms,
    uint output_origin, uint simd_lane, uint simd_group,
    threadgroup typename Q8Operand<A>::Term *input_terms = nullptr,
    threadgroup float2 *staged_parameters = nullptr) {
  using Operands = Q8Operand<A>;
  using Partial = typename Operands::Partial;
  constexpr bool StagedTerms = Simdgroups == 8;
  constexpr ushort TermBatch = Operands::TermBatch;
  auto a = tensor(input, dextents<int, 2>{int(input_size), TileM},
                  array<int, 2>{1, int(input_size)});
  auto c = tensor(output, dextents<int, 2>{int(output_size), TileM},
                  array<int, 2>{1, int(output_size)});
  constexpr auto descriptor =
      matmul2d_descriptor(TileM, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  auto a0 = a.template slice<64, TileM>(0, 0);
  uint quant_groups = input_size / 64;
  constexpr ushort WeightTileN = 256; // weights are stored in 256-column tiles
  uint tile = output_origin / WeightTileN;
  uint tile_column = output_origin % WeightTileN;
  device int8_t *tile_weights =
      weights + (ulong(tile) * quant_groups * WeightTileN + tile_column) * 64;
  tensor<device int8_t, dextents<int, 2>, tensor_inline> first_b(
      tile_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
  auto b0 = first_b.slice<64, TileN>(0, 0);
  auto accumulated = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i)
    accumulated[i] = 0.0f;

  auto load_terms = [&](uint start) {
    uint count = min(uint(TermBatch), quant_groups - start);
    uint thread_index = simd_group * 32 + simd_lane;
    for (uint index = thread_index; index < count * TileM;
         index += Simdgroups * 32) {
      uint quant_group = start + index / TileM;
      uint row = index % TileM;
      input_terms[index] = precomputed_terms[quant_group * TileM + row];
    }
  };
  if constexpr (StagedTerms) {
    load_terms(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if constexpr (Pipelined) {
    static_assert(A == Q8Activation::A8 && TileM == 32 && TileN == 128 &&
                      Simdgroups == 4 && is_same_v<P, bfloat>);
    using PartialTensor = decltype(operation.template
        get_destination_cooperative_tensor<decltype(a0), decltype(b0), Partial>());
    auto run_group = [&](uint group, thread PartialTensor &partial) {
      auto a_slice = a.template slice<64, TileM>(group * 64, 0);
      tensor<device int8_t, dextents<int, 2>, tensor_inline> b(
          tile_weights + ulong(group) * WeightTileN * 64,
          dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
      auto b_slice = b.slice<64, TileN>(0, 0);
      operation.run(a_slice, b_slice, partial);
    };
    auto finish_group = [&](uint group, thread PartialTensor &partial) {
#pragma unroll
      for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        auto index = accumulated.get_multidimensional_index(i);
        uint row = index[1];
        ulong parameter = (ulong(tile) * quant_groups + group) * WeightTileN +
                          tile_column + index[0];
        const float2 ax = precomputed_terms[group * TileM + row];
        accumulated[i] = q8a8_accumulate(
            accumulated[i], partial[i], ax, q8_scale(scales, parameter),
            q8_folded_bias(scales, biases, parameter));
      }
    };
    uint group = 0;
    for (; group + 1 < quant_groups; group += 2) {
      PartialTensor first, second;
      run_group(group, first);
      run_group(group + 1, second);
      finish_group(group, first);
      finish_group(group + 1, second);
    }
    if (group < quant_groups) {
      PartialTensor partial;
      run_group(group, partial);
      finish_group(group, partial);
    }
  } else
  for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
    if constexpr (StageParameters) {
      uint thread_index = simd_group * 32 + simd_lane;
      for (uint column = thread_index; column < TileN;
           column += Simdgroups * 32) {
        ulong parameter = (ulong(tile) * quant_groups + quant_group) *
                              WeightTileN + tile_column + column;
        staged_parameters[column] =
            float2(q8_scale(scales, parameter),
                   q8_folded_bias(scales, biases, parameter));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint input_origin = quant_group * 64;
    auto a_slice = a.template slice<64, TileM>(input_origin, 0);
    device int8_t *group_weights =
        tile_weights + ulong(quant_group) * WeightTileN * 64;
    tensor<device int8_t, dextents<int, 2>, tensor_inline> b(
        group_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b_slice = b.slice<64, TileN>(0, 0);
    auto partial = operation.template get_destination_cooperative_tensor<
        decltype(a_slice), decltype(b_slice), Partial>();
    operation.run(a_slice, b_slice, partial);

#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
      auto index = accumulated.get_multidimensional_index(i);
      uint row = index[1];
      ulong parameter =
          (ulong(tile) * quant_groups + quant_group) * WeightTileN +
          tile_column + index[0];
      if constexpr (A == Q8Activation::A16) {
        const float sum =
            StagedTerms
                ? input_terms[(quant_group % TermBatch) * TileM + row]
                : precomputed_terms[quant_group * TileM + row];
        accumulated[i] = q8a16_accumulate(
            accumulated[i], partial[i], sum, q8_scale(scales, parameter),
            q8_folded_bias(scales, biases, parameter));
      } else {
        // Prefill A8 reads either parameter form through the same helpers
        // as A16; for fp32 they return the stored s and b' unchanged.
        const float2 ax =
            StagedTerms
                ? input_terms[(quant_group % TermBatch) * TileM + row]
                : precomputed_terms[quant_group * TileM + row];
        float scale, bias;
        if constexpr (PackedParameters) {
          static_assert(is_same_v<P, float> && !StageParameters,
                        "packed A8 parameters require FP32 interleaved view");
          const float2 pair =
              reinterpret_cast<device const float2 *>(scales)[parameter];
          scale = pair.x;
          bias = pair.y;
        } else {
          scale = StageParameters ? staged_parameters[index[0]].x
                                  : q8_scale(scales, parameter);
          bias = StageParameters ? staged_parameters[index[0]].y
                                 : q8_folded_bias(scales, biases, parameter);
        }
        accumulated[i] = q8a8_accumulate(accumulated[i], partial[i], ax,
                                         scale, bias);
      }
    }
    if constexpr (StageParameters)
      threadgroup_barrier(mem_flags::mem_threadgroup);
    if (StagedTerms && quant_group % TermBatch == TermBatch - 1 &&
        quant_group + 1 < quant_groups) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      load_terms(quant_group + 1);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  auto converted = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), bfloat>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
    float value = float(bfloat(accumulated[i]));
    if constexpr (MultiplySiluGate) {
      auto index = accumulated.get_multidimensional_index(i);
      float gate =
          float(auxiliary[index[1] * output_size + output_origin + index[0]]);
      value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * value;
    }
    if constexpr (AddResidual) {
      auto index = accumulated.get_multidimensional_index(i);
      value +=
          float(auxiliary[index[1] * output_size + output_origin + index[0]]);
    }
    converted[i] = bfloat(value);
  }
  converted.store(c.template slice<TileN, TileM>(output_origin, 0));
}

// A16 fused up projections emit the down projection's fp32 group sums from
// the rounded bf16 output, matching q4_prefill_write_output_sums.
template <ushort TileM, ushort TileN, ushort Simdgroups>
inline void q8_prefill_write_output_sums(device const bfloat *output,
                                         device float *output_sums,
                                         uint output_size, uint output_origin,
                                         uint simd_lane, uint simd_group) {
  constexpr uint QuantGroups = TileN / 64;
  threadgroup_barrier(mem_flags::mem_device);
  for (uint task = simd_group; task < TileM * QuantGroups; task += Simdgroups) {
    uint row = task / QuantGroups;
    uint local_group = task % QuantGroups;
    uint origin =
        row * output_size + output_origin + local_group * 64 + simd_lane;
    float sum = simd_sum(float(output[origin]) + float(output[origin + 32]));
    if (simd_lane == 0) {
      uint quant_group = output_origin / 64 + local_group;
      output_sums[quant_group * TileM + row] = sum;
    }
  }
}

// The A8 counterpart writes the down projection's quantized input and its
// {sx, sx * qxsum} terms. ax carries the tiled [group][row] layout the next
// projection's epilogue reads.
template <ushort TileM, ushort TileN, ushort Simdgroups>
inline void q8_prefill_quantize_output(device const bfloat *output,
                                       device int8_t *qx, device float2 *ax,
                                       uint output_size, uint output_origin,
                                       uint simd_lane, uint simd_group) {
  constexpr uint QuantGroups = TileN / 64;
  threadgroup_barrier(mem_flags::mem_device);
  for (uint task = simd_group; task < TileM * QuantGroups; task += Simdgroups) {
    uint row = task / QuantGroups;
    uint local_group = task % QuantGroups;
    uint origin =
        row * output_size + output_origin + local_group * 64 + simd_lane;
    int qa, qb;
    const float2 term =
        q8_quantize_pair(float(output[origin]), float(output[origin + 32]), qa,
                         qb);
    qx[origin] = int8_t(qa);
    qx[origin + 32] = int8_t(qb);
    if (simd_lane == 0) {
      uint quant_group = output_origin / 64 + local_group;
      ax[quant_group * TileM + row] = term;
    }
  }
}
