#pragma once

#include "metal/abi/KernelABI.h"

// Q4 (4-bit, group 64, StorageN=256) tiles shared by dense and MoE projections.
// A threadgroup computes Rows x TileN outputs with fp32 accumulation and a
// per-quant-group scale/bias epilogue.

// Resolve cooperative fragment traversal once, outside the quant-group loop.
// The compact path is used only after every validity bit matches; a different
// compiler/device layout retains the intrinsic-checked traversal.
enum class Q4Traversal : ushort {
  All, FourOfEight, PrefixAndFourOfEight, HalfPrefix, Guarded
};

template <class Tensor>
__attribute__((always_inline)) inline Q4Traversal
q4_traversal(const thread Tensor &values) {
  const ushort capacity = values.get_capacity();
  bool all = true;
  // Apple9 M24/N128 fragments can have only a contiguous first half valid.
  bool halfPrefix = capacity != 0 && (capacity % 2) == 0;
  bool striped = capacity != 0 && (capacity % 8) == 0;
  bool prefixed = capacity != 0 && (capacity % 16) == 0;
#pragma unroll
  for (ushort i = 0; i < capacity; ++i) {
    const bool valid = values.is_valid_element(i);
    all &= valid;
    halfPrefix &= valid == (i < capacity / 2);
    striped &= valid == ((i & 7) < 4);
    prefixed &= valid == (i < capacity / 2 || ((i & 7) < 4));
  }
  return all ? Q4Traversal::All : striped ? Q4Traversal::FourOfEight
       : prefixed ? Q4Traversal::PrefixAndFourOfEight
       : halfPrefix ? Q4Traversal::HalfPrefix : Q4Traversal::Guarded;
}

template <class Tensor, class Body>
__attribute__((always_inline)) inline void
q4_visit(const thread Tensor &values, Q4Traversal traversal,
         const thread Body &body) {
  if (traversal == Q4Traversal::All) {
#pragma unroll
    for (ushort i = 0; i < values.get_capacity(); ++i) body(i);
  } else if (traversal == Q4Traversal::FourOfEight) {
#pragma unroll
    for (ushort i = 0; i < values.get_capacity() / 2; ++i)
      body(ushort((i / 4) * 8 + i % 4));
  } else if (traversal == Q4Traversal::PrefixAndFourOfEight) {
#pragma unroll
    for (ushort i = 0; i < values.get_capacity() / 2; ++i) body(i);
#pragma unroll
    for (ushort i = 0; i < values.get_capacity() / 4; ++i)
      body(ushort(values.get_capacity() / 2 + (i / 4) * 8 + i % 4));
  } else if (traversal == Q4Traversal::HalfPrefix) {
#pragma unroll
    for (ushort i = 0; i < values.get_capacity() / 2; ++i) body(i);
  } else {
#pragma unroll
    for (ushort i = 0; i < values.get_capacity(); ++i)
      if (values.is_valid_element(i)) body(i);
  }
}

template <ushort Rows = 8, ushort Simdgroups = 8>
inline void q4_store_input_sums(device const bfloat *input, uint input_size,
                                uint input_origin, threadgroup float *sums,
                                uint sum_origin, uint simd_lane,
                                uint simd_group) {
  for (uint row = simd_group; row < Rows; row += Simdgroups) {
    uint origin = row * input_size + input_origin + simd_lane;
    float first = simd_sum(float(input[origin]) + float(input[origin + 32]));
    float second =
        simd_sum(float(input[origin + 64]) + float(input[origin + 96]));
    float third =
        simd_sum(float(input[origin + 128]) + float(input[origin + 160]));
    float fourth =
        simd_sum(float(input[origin + 192]) + float(input[origin + 224]));
    if (simd_lane == 0) {
      sums[sum_origin + row] = first;
      sums[sum_origin + Rows + row] = second;
      sums[sum_origin + 2 * Rows + row] = third;
      sums[sum_origin + 3 * Rows + row] = fourth;
    }
  }
}

// Pipelined issues two quant groups' matmuls before either epilogue. Narrow
// projections run few threadgroups and are bound by the latency of one group's
// weight load and matmul, so overlapping two hides most of it; wide
// projections lose to the extra live registers. The epilogues still apply in
// group order, so the results are bit-identical to the sequential form.
// Simdgroups is the cooperative scope of the matmul: four simdgroups halve a
// threadgroup to 128 threads so four of them fit a core at the 512-thread
// occupancy knee; the per-element arithmetic is unchanged, so every
// Simdgroups instance of one tile is bit-identical to the others.
template <ushort TileN, bool GateUp, bool AddResidual,
          ushort StorageN = TileN, bool Pipelined = false, ushort Simdgroups = 8>
inline void q4_mpp_tile(device bfloat *input, device uchar *weights_0,
                        device bfloat *scales_0, device bfloat *biases_0,
                        device bfloat *output_0, device uchar *weights_1,
                        device bfloat *scales_1, device bfloat *biases_1,
                        device bfloat *residual, uint output_size,
                        uint input_size, threadgroup float *input_sums,
                        uint output_origin, uint simd_lane, uint simd_group) {
  auto a = tensor(input, dextents<int, 2>{int(input_size), 8},
                  array<int, 2>{1, int(input_size)});
  constexpr auto descriptor =
      matmul2d_descriptor(8, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  auto a0 = a.slice<64, 8>(0, 0);
  uint quant_groups = input_size / 64;
  uint tile = output_origin / StorageN;
  uint tile_offset = output_origin % StorageN;
  device uchar *tile_weights_0 =
      weights_0 + ulong(tile) * quant_groups * StorageN * 64 / 2;
  device uchar *tile_weights_1 =
      weights_1 + ulong(tile) * quant_groups * StorageN * 64 / 2;
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_b0(
      tile_weights_0 + tile_offset * 32, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_b1(
      tile_weights_1 + tile_offset * 32, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  auto b00 = first_b0.slice<64, TileN>(0, 0);
  auto b10 = first_b1.slice<64, TileN>(0, 0);
  auto accumulated_0 = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b00), float>();
  auto accumulated_1 = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b10), float>();
  // MSL 2.22.3 partitions logical elements and makes capacity uniform across
  // participating threads. The descriptor's 8 x TileN destination
  // has no padding when total capacity equals that logical element count.
  const bool fullyOccupied =
      uint(accumulated_0.get_capacity()) * (uint(Simdgroups) * 32u) == 8u * TileN;
  const auto traversal = fullyOccupied ? Q4Traversal::All
                                       : q4_traversal(accumulated_0);
  q4_visit(accumulated_0, traversal, [&](ushort i) {
    accumulated_0[i] = 0.0f;
    if constexpr (GateUp)
      accumulated_1[i] = 0.0f;
  });

  q4_store_input_sums<8, Simdgroups>(input, input_size, 0, input_sums, 0,
                                     simd_lane, simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto run_group = [&](uint quant_group,
                       thread decltype(accumulated_0) &partial_0,
                       thread decltype(accumulated_1) &partial_1) {
    uint input_origin = quant_group * 64;
    auto a_slice = a.slice<64, 8>(input_origin, 0);
    device uchar *group_weights_0 =
        tile_weights_0 + (ulong(quant_group) * StorageN + tile_offset) * 64 / 2;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b0(
        group_weights_0, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b0_slice = b0.slice<64, TileN>(0, 0);
    operation.run(a_slice, b0_slice, partial_0);
    device uchar *group_weights_1 =
        tile_weights_1 + (ulong(quant_group) * StorageN + tile_offset) * 64 / 2;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b1(
        group_weights_1, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b1_slice = b1.slice<64, TileN>(0, 0);
    if constexpr (GateUp)
      operation.run(a_slice, b1_slice, partial_1);
  };
  auto finish_group = [&](uint quant_group,
                          thread decltype(accumulated_0) &partial_0,
                          thread decltype(accumulated_1) &partial_1) {
    q4_visit(accumulated_0, traversal,
             [&](ushort i) __attribute__((always_inline)) {
      auto index = accumulated_0.get_multidimensional_index(i);
      uint row = index[1];
      ulong parameter = (ulong(tile) * quant_groups + quant_group) * StorageN +
                        tile_offset + index[0];
      uint sum_offset = ((quant_group >> 2) & 1) * 32 + (quant_group & 3) * 8;
      accumulated_0[i] +=
          partial_0[i] * float(scales_0[parameter]) +
          input_sums[sum_offset + row] * float(biases_0[parameter]);
      if constexpr (GateUp) {
        accumulated_1[i] +=
            partial_1[i] * float(scales_1[parameter]) +
            input_sums[sum_offset + row] * float(biases_1[parameter]);
      }
    });
    if ((quant_group & 3) == 3 && quant_group + 1 < quant_groups) {
      uint next_group = (quant_group + 1) >> 2;
      q4_store_input_sums<8, Simdgroups>(input, input_size,
                                         quant_group * 64 + 64, input_sums,
                                         (next_group & 1) * 32, simd_lane,
                                         simd_group);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  };
  if constexpr (Pipelined) {
    uint quant_group = 0;
    for (; quant_group + 1 < quant_groups; quant_group += 2) {
      decltype(accumulated_0) first_0, second_0;
      decltype(accumulated_1) first_1, second_1;
      run_group(quant_group, first_0, first_1);
      run_group(quant_group + 1, second_0, second_1);
      finish_group(quant_group, first_0, first_1);
      finish_group(quant_group + 1, second_0, second_1);
    }
    if (quant_group < quant_groups) {
      decltype(accumulated_0) partial_0;
      decltype(accumulated_1) partial_1;
      run_group(quant_group, partial_0, partial_1);
      finish_group(quant_group, partial_0, partial_1);
    }
  } else {
    for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
      decltype(accumulated_0) partial_0;
      decltype(accumulated_1) partial_1;
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
  // Persistent callers run the next tile on the same scratch straight away,
  // and its prologue rewrites input-sum region 0 while a straggling simdgroup
  // may still read that region in the last quant-group block's epilogue (the
  // last block is even when K % 512 == 256). Every simdgroup finishes first.
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <ushort Rows, ushort TileN, bool GateUp, bool AddResidual,
          ushort StorageN = TileN, bool MultiplySiluGate = false,
          ushort Simdgroups = 8>
inline void q4_mpp_tile_batched(
    device bfloat *input, device uchar *weights_0, device bfloat *scales_0,
    device bfloat *biases_0, device bfloat *output_0, device uchar *weights_1,
    device bfloat *scales_1, device bfloat *biases_1, device bfloat *residual,
    uint output_size, uint input_size, threadgroup float *input_sums,
    uint output_origin, uint simd_lane, uint simd_group) {
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows},
                  array<int, 2>{1, int(input_size)});
  constexpr auto descriptor =
      matmul2d_descriptor(Rows, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  uint quant_groups = input_size / 64;
  uint tile = output_origin / StorageN;
  uint tile_offset = output_origin % StorageN;
  device uchar *tile_weights_0 =
      weights_0 + ulong(tile) * quant_groups * StorageN * 64 / 2;
  device uchar *tile_weights_1 =
      weights_1 + ulong(tile) * quant_groups * StorageN * 64 / 2;
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_b0(
      tile_weights_0 + tile_offset * 32, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_b1(
      tile_weights_1 + tile_offset * 32, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  auto a0 = a.slice<64, Rows>(0, 0);
  auto b00 = first_b0.slice<64, TileN>(0, 0);
  auto b10 = first_b1.slice<64, TileN>(0, 0);
  auto accumulated_0 = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b00), float>();
  auto accumulated_1 = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b10), float>();
  // Full Rows x TileN destination and uniform partition capacity, as above.
  const bool fullyOccupied =
      uint(accumulated_0.get_capacity()) * (uint(Simdgroups) * 32u) ==
      uint(Rows) * TileN;
  const auto traversal = fullyOccupied ? Q4Traversal::All
                                       : q4_traversal(accumulated_0);
  q4_visit(accumulated_0, traversal, [&](ushort i) {
    accumulated_0[i] = 0.0f;
    if constexpr (GateUp)
      accumulated_1[i] = 0.0f;
  });
  q4_store_input_sums<Rows, Simdgroups>(input, input_size, 0, input_sums, 0, simd_lane,
                            simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
    uint input_origin = quant_group * 64;
    auto a_slice = a.slice<64, Rows>(input_origin, 0);
    device uchar *group_weights_0 =
        tile_weights_0 + (ulong(quant_group) * StorageN + tile_offset) * 64 / 2;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b0(
        group_weights_0, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b0_slice = b0.slice<64, TileN>(0, 0);
    auto partial_0 = operation.template get_destination_cooperative_tensor<
        decltype(a_slice), decltype(b0_slice), float>();
    operation.run(a_slice, b0_slice, partial_0);
    device uchar *group_weights_1 =
        tile_weights_1 + (ulong(quant_group) * StorageN + tile_offset) * 64 / 2;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b1(
        group_weights_1, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b1_slice = b1.slice<64, TileN>(0, 0);
    auto partial_1 = operation.template get_destination_cooperative_tensor<
        decltype(a_slice), decltype(b1_slice), float>();
    if constexpr (GateUp)
      operation.run(a_slice, b1_slice, partial_1);
    q4_visit(accumulated_0, traversal,
             [&](ushort i) __attribute__((always_inline)) {
      auto index = accumulated_0.get_multidimensional_index(i);
      uint row = index[1];
      ulong parameter = (ulong(tile) * quant_groups + quant_group) * StorageN +
                        tile_offset + index[0];
      uint sum_offset =
          ((quant_group >> 2) & 1) * (4 * Rows) + (quant_group & 3) * Rows;
      accumulated_0[i] +=
          partial_0[i] * float(scales_0[parameter]) +
          input_sums[sum_offset + row] * float(biases_0[parameter]);
      if constexpr (GateUp) {
        accumulated_1[i] +=
            partial_1[i] * float(scales_1[parameter]) +
            input_sums[sum_offset + row] * float(biases_1[parameter]);
      }
    });
    if ((quant_group & 3) == 3 && quant_group + 1 < quant_groups) {
      uint next_group = (quant_group + 1) >> 2;
      q4_store_input_sums<Rows, Simdgroups>(input, input_size, input_origin + 64,
                                input_sums, (next_group & 1) * (4 * Rows),
                                simd_lane, simd_group);
      threadgroup_barrier(mem_flags::mem_threadgroup);
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
  // Same next-tile hazard on input-sum region 0 as q4_mpp_tile.
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Split-K form of the Rows=8 q4_mpp_tile: SplitK partitions of Simdgroups
// simdgroups each, all in one threadgroup, stream equal quant-group ranges of
// the same 8 x TileN tile and leave fp32 partial sums in threadgroup memory,
// [partition][row][column]; the caller reduces them and applies the epilogue.
// One tile thus occupies SplitK times the simdgroups of the sequential form,
// which is what a narrow projection needs to reach the occupancy knee of 16
// resident simdgroups per core when it has fewer tiles than cores. Every
// partition runs the same loop count, so the input-sum barriers inside stay
// aligned across the threadgroup; input_size % (256 * SplitK) == 0 keeps each
// range a whole number of four-group input-sum blocks.
//
// Numerics: each partition accumulates its own quant-group range in the
// sequential kernel's group order with the same per-group terms, so the
// only difference from the sequential form is the association of the fp32
// sum: four range sums added at the end instead of one running sum. After
// the single bf16 rounding, cancellation can make the difference exceed one
// output ulp; qualification needs an operand-magnitude error bound. Instances
// with the same SplitK retain the same per-element accumulation order.
template <ushort TileN, bool GateUp, ushort StorageN = TileN,
          bool Pipelined = true, ushort Simdgroups = 8, ushort SplitK = 4>
inline void q4_mpp_tile_split(device bfloat *input, device uchar *weights_0,
                              device bfloat *scales_0, device bfloat *biases_0,
                              threadgroup float *partials,
                              device uchar *weights_1, device bfloat *scales_1,
                              device bfloat *biases_1, uint input_size,
                              threadgroup float *input_sums,
                              uint output_origin, uint simd_lane,
                              uint simd_group, uint partition) {
  auto a = tensor(input, dextents<int, 2>{int(input_size), 8},
                  array<int, 2>{1, int(input_size)});
  constexpr auto descriptor =
      matmul2d_descriptor(8, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  auto a0 = a.slice<64, 8>(0, 0);
  uint total_quant_groups = input_size / 64;
  uint quant_groups = total_quant_groups / SplitK;
  uint first_group = partition * quant_groups;
  uint tile = output_origin / StorageN;
  uint tile_offset = output_origin % StorageN;
  device uchar *tile_weights_0 =
      weights_0 +
      (ulong(tile) * total_quant_groups + first_group) * StorageN * 64 / 2;
  device uchar *tile_weights_1 =
      weights_1 +
      (ulong(tile) * total_quant_groups + first_group) * StorageN * 64 / 2;
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_b0(
      tile_weights_0 + tile_offset * 32, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_b1(
      tile_weights_1 + tile_offset * 32, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  auto b00 = first_b0.slice<64, TileN>(0, 0);
  auto b10 = first_b1.slice<64, TileN>(0, 0);
  auto accumulated_0 = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b00), float>();
  auto accumulated_1 = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b10), float>();
  // Full 8 x TileN destination and uniform partition capacity, as above.
  const bool fullyOccupied =
      uint(accumulated_0.get_capacity()) * (uint(Simdgroups) * 32u) ==
      8u * TileN;
  const auto traversal = fullyOccupied ? Q4Traversal::All
                                       : q4_traversal(accumulated_0);
  q4_visit(accumulated_0, traversal, [&](ushort i) {
    accumulated_0[i] = 0.0f;
    if constexpr (GateUp)
      accumulated_1[i] = 0.0f;
  });

  q4_store_input_sums<8, Simdgroups>(input, input_size, first_group * 64,
                                     input_sums, 0, simd_lane, simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto run_group = [&](uint quant_group,
                       thread decltype(accumulated_0) &partial_0,
                       thread decltype(accumulated_1) &partial_1) {
    uint input_origin = (first_group + quant_group) * 64;
    auto a_slice = a.slice<64, 8>(input_origin, 0);
    device uchar *group_weights_0 =
        tile_weights_0 + (ulong(quant_group) * StorageN + tile_offset) * 64 / 2;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b0(
        group_weights_0, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b0_slice = b0.slice<64, TileN>(0, 0);
    operation.run(a_slice, b0_slice, partial_0);
    device uchar *group_weights_1 =
        tile_weights_1 + (ulong(quant_group) * StorageN + tile_offset) * 64 / 2;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b1(
        group_weights_1, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b1_slice = b1.slice<64, TileN>(0, 0);
    if constexpr (GateUp)
      operation.run(a_slice, b1_slice, partial_1);
  };
  auto finish_group = [&](uint quant_group,
                          thread decltype(accumulated_0) &partial_0,
                          thread decltype(accumulated_1) &partial_1) {
    q4_visit(accumulated_0, traversal,
             [&](ushort i) __attribute__((always_inline)) {
      auto index = accumulated_0.get_multidimensional_index(i);
      uint row = index[1];
      ulong parameter =
          (ulong(tile) * total_quant_groups + first_group + quant_group) *
              StorageN + tile_offset + index[0];
      uint sum_offset = ((quant_group >> 2) & 1) * 32 + (quant_group & 3) * 8;
      accumulated_0[i] +=
          partial_0[i] * float(scales_0[parameter]) +
          input_sums[sum_offset + row] * float(biases_0[parameter]);
      if constexpr (GateUp) {
        accumulated_1[i] +=
            partial_1[i] * float(scales_1[parameter]) +
            input_sums[sum_offset + row] * float(biases_1[parameter]);
      }
    });
    if ((quant_group & 3) == 3 && quant_group + 1 < quant_groups) {
      uint next_group = (quant_group + 1) >> 2;
      q4_store_input_sums<8, Simdgroups>(
          input, input_size, (first_group + quant_group) * 64 + 64, input_sums,
          (next_group & 1) * 32, simd_lane, simd_group);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  };
  if constexpr (Pipelined) {
    uint quant_group = 0;
    for (; quant_group + 1 < quant_groups; quant_group += 2) {
      decltype(accumulated_0) first_0, second_0;
      decltype(accumulated_1) first_1, second_1;
      run_group(quant_group, first_0, first_1);
      run_group(quant_group + 1, second_0, second_1);
      finish_group(quant_group, first_0, first_1);
      finish_group(quant_group + 1, second_0, second_1);
    }
    if (quant_group < quant_groups) {
      decltype(accumulated_0) partial_0;
      decltype(accumulated_1) partial_1;
      run_group(quant_group, partial_0, partial_1);
      finish_group(quant_group, partial_0, partial_1);
    }
  } else {
    for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
      decltype(accumulated_0) partial_0;
      decltype(accumulated_1) partial_1;
      run_group(quant_group, partial_0, partial_1);
      finish_group(quant_group, partial_0, partial_1);
    }
  }

  // fp32 partials, [partition][row][column]; the gate/up second stream
  // follows all SplitK first-stream partitions.
  q4_visit(accumulated_0, traversal, [&](ushort i) {
    auto index = accumulated_0.get_multidimensional_index(i);
    uint slot = index[1] * TileN + index[0];
    partials[partition * 8 * TileN + slot] = accumulated_0[i];
    if constexpr (GateUp)
      partials[(SplitK + partition) * 8 * TileN + slot] = accumulated_1[i];
  });
  // The caller's reduction reads every partition's partials, and the next
  // tile's prologue rewrites input-sum region 0; every simdgroup finishes.
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
