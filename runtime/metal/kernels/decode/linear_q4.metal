#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/q4_mpp_tiles.h"

// Decode projections: persistent threadgroups stride over TileN-wide output
// tiles, TileCall names the q4_mpp_tiles.h instantiation and Sums holds eight
// input sums per row. The auxiliary buffer is the residual the epilogue adds
// or the gate it applies SiLU to.
#define Q4_DECODE_AFFINE(Name, TileCall, Sums, TileN)                          \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *output [[buffer(4)]],                        \
                   constant Q4Params &params [[buffer(5)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      TileCall(input, weights, scales, biases, output, weights, scales,        \
               biases, output, params.output_size, params.input_size,          \
               input_sums, tile * TileN, simd_lane, simd_group);               \
    }                                                                          \
  }

#define Q4_DECODE_AUXILIARY(Name, Auxiliary, TileCall, Sums, TileN)            \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *Auxiliary [[buffer(4)]],                     \
                   device bfloat *output [[buffer(5)]],                        \
                   constant Q4Params &params [[buffer(6)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      TileCall(input, weights, scales, biases, output, weights, scales,        \
               biases, Auxiliary, params.output_size, params.input_size,       \
               input_sums, tile * TileN, simd_lane, simd_group);               \
    }                                                                          \
  }

#define Q4_DECODE_GATE_UP(Name, TileCall, Sums, TileN)                         \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights_0 [[buffer(1)]],                      \
                   device bfloat *scales_0 [[buffer(2)]],                      \
                   device bfloat *biases_0 [[buffer(3)]],                      \
                   device bfloat *output [[buffer(4)]],                        \
                   device uchar *weights_1 [[buffer(5)]],                      \
                   device bfloat *scales_1 [[buffer(6)]],                      \
                   device bfloat *biases_1 [[buffer(7)]],                      \
                   constant Q4Params &params [[buffer(8)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      TileCall(input, weights_0, scales_0, biases_0, output, weights_1,        \
               scales_1, biases_1, output, params.output_size,                 \
               params.input_size, input_sums, tile * TileN, simd_lane,         \
               simd_group);                                                    \
    }                                                                          \
  }

Q4_DECODE_AFFINE(decode_linear_q4_n128, (q4_mpp_tile<128, false, false, 256>), 64,
                 128)
Q4_DECODE_AFFINE(decode_linear_q4_n128_m16,
                 (q4_mpp_tile_batched<16, 128, false, false, 256>), 128, 128)
Q4_DECODE_AFFINE(decode_linear_q4_n128_m24,
                 (q4_mpp_tile_batched<24, 128, false, false, 256>), 192, 128)
Q4_DECODE_AFFINE(decode_linear_q4_n128_m24_sg4,
                 (q4_mpp_tile_batched<24, 128, false, false, 256, false, 4>), 192, 128)
Q4_DECODE_AFFINE(decode_linear_q4_n256_m16,
                 (q4_mpp_tile_batched<16, 256, false, false>), 128, 256)
Q4_DECODE_AFFINE(decode_linear_q4_n256_m24,
                 (q4_mpp_tile_batched<24, 256, false, false>), 192, 256)
Q4_DECODE_AFFINE(decode_linear_q4_n256, (q4_mpp_tile<256, false, false>), 64, 256)
Q4_DECODE_AFFINE(decode_linear_q4_n128_paired,
                 (q4_mpp_tile<128, false, false, 256, true>), 64, 128)
// 128 threads: four 8 x 256 tiles per core reach the occupancy knee for very
// wide one-lane projections, with half the input re-reads of N128 tiles.
Q4_DECODE_AFFINE(decode_linear_q4_n256_paired_sg4,
                 (q4_mpp_tile<256, false, false, 256, true, 4>), 64, 256)
Q4_DECODE_AUXILIARY(decode_linear_q4_n128_residual_paired, residual,
                    (q4_mpp_tile<128, false, true, 256, true>), 64, 128)
Q4_DECODE_AUXILIARY(decode_linear_q4_n128_residual, residual,
                    (q4_mpp_tile<128, false, true, 256>), 64, 128)
Q4_DECODE_GATE_UP(decode_linear_q4_n256_gate_up, (q4_mpp_tile<256, true, false>), 64, 256)
Q4_DECODE_AUXILIARY(decode_linear_q4_n128_residual_m16, residual,
                    (q4_mpp_tile_batched<16, 128, false, true, 256>), 128, 128)
Q4_DECODE_AUXILIARY(decode_linear_q4_n128_residual_m24, residual,
                    (q4_mpp_tile_batched<24, 128, false, true, 256>), 192, 128)
Q4_DECODE_AUXILIARY(decode_linear_q4_n128_residual_m24_sg4, residual,
                    (q4_mpp_tile_batched<24, 128, false, true, 256, false, 4>), 192, 128)
Q4_DECODE_GATE_UP(decode_linear_q4_n256_gate_up_m16,
                  (q4_mpp_tile_batched<16, 256, true, false>), 128, 256)
Q4_DECODE_AFFINE(decode_linear_q4_n128_m32,
                 (q4_mpp_tile_batched<32, 128, false, false, 256>), 256, 128)
Q4_DECODE_AFFINE(decode_linear_q4_n256_m32,
                 (q4_mpp_tile_batched<32, 256, false, false>), 256, 256)
Q4_DECODE_AUXILIARY(decode_linear_q4_n128_residual_m32, residual,
                    (q4_mpp_tile_batched<32, 128, false, true, 256>), 256, 128)
Q4_DECODE_AUXILIARY(decode_linear_q4_n256_up_silu_m32, gate,
                    (q4_mpp_tile_batched<32, 256, false, false, 256, true>),
                    256, 256)
Q4_DECODE_AUXILIARY(decode_linear_q4_n256_up_silu_m24, gate,
                    (q4_mpp_tile_batched<24, 256, false, false, 256, true>),
                    192, 256)
// Narrow plain projections at 24 and 32 rows: N64 tiles double the N128 grid
// with each column's group chain unchanged (the same bytes, checked against
// N128 by linear-plan on every sequential candidate).
Q4_DECODE_AFFINE(decode_linear_q4_n64_m24,
                 (q4_mpp_tile_batched<24, 64, false, false, 256>), 192, 64)
Q4_DECODE_AFFINE(decode_linear_q4_n64_m32,
                 (q4_mpp_tile_batched<32, 64, false, false, 256>), 256, 64)
// M64 (native B8): eight lanes of verify rows in one cooperative tile.
Q4_DECODE_AFFINE(decode_linear_q4_n64_m64,
                 (q4_mpp_tile_batched<64, 64, false, false, 256>), 512, 64)
Q4_DECODE_AFFINE(decode_linear_q4_n128_m64,
                 (q4_mpp_tile_batched<64, 128, false, false, 256>), 512, 128)
Q4_DECODE_AFFINE(decode_linear_q4_n256_m64,
                 (q4_mpp_tile_batched<64, 256, false, false>), 512, 256)
Q4_DECODE_AUXILIARY(decode_linear_q4_n128_residual_m64, residual,
                    (q4_mpp_tile_batched<64, 128, false, true, 256>), 512, 128)
Q4_DECODE_AUXILIARY(decode_linear_q4_n256_up_silu_m64, gate,
                    (q4_mpp_tile_batched<64, 256, false, false, 256, true>),
                    512, 256)

// B5 N256 screen winner: M48 padding keeps the existing uint4b arithmetic.
Q4_DECODE_AFFINE(decode_linear_q4_n256_m48,
                 (q4_mpp_tile_batched<48, 256, false, false>), 384, 256)
Q4_DECODE_AUXILIARY(decode_linear_q4_n256_up_silu_m48, gate,
                    (q4_mpp_tile_batched<48, 256, false, false, 256, true>),
                    384, 256)

// Wide-decode row slabs: one dispatch carries two independent 32-row
// cooperative tiles selected by group.z. Each slab runs the identical M32
// tile math on a 32-row window of the shared activation/output/auxiliary
// buffers, so active rows keep the m32 byte stream while one command covers
// the whole 64-row logical width (the draft model's B5-B8 decode batch).
#define Q4_DECODE_SLAB_AFFINE(Name, TileCall, Sums, TileN)                     \
  kernel void Name(device bfloat *input [[buffer(0)]],                        \
                   device uchar *weights [[buffer(1)]],                       \
                   device bfloat *scales [[buffer(2)]],                       \
                   device bfloat *biases [[buffer(3)]],                       \
                   device bfloat *output [[buffer(4)]],                       \
                   constant Q4Params &params [[buffer(5)]],                   \
                   uint3 group [[threadgroup_position_in_grid]],              \
                   uint simd_lane [[thread_index_in_simdgroup]],              \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {      \
    threadgroup float input_sums[Sums];                                       \
    input += ulong(group.z) * 32 * params.input_size;                         \
    output += ulong(group.z) * 32 * params.output_size;                       \
    uint tiles = params.output_size / TileN;                                  \
    for (uint tile = group.x; tile < tiles; tile += params.persistent_groups) {\
      TileCall(input, weights, scales, biases, output, weights, scales,       \
               biases, output, params.output_size, params.input_size,         \
               input_sums, tile * TileN, simd_lane, simd_group);              \
    }                                                                         \
  }

#define Q4_DECODE_SLAB_AUXILIARY(Name, Auxiliary, TileCall, Sums, TileN)       \
  kernel void Name(device bfloat *input [[buffer(0)]],                        \
                   device uchar *weights [[buffer(1)]],                       \
                   device bfloat *scales [[buffer(2)]],                       \
                   device bfloat *biases [[buffer(3)]],                       \
                   device bfloat *Auxiliary [[buffer(4)]],                    \
                   device bfloat *output [[buffer(5)]],                       \
                   constant Q4Params &params [[buffer(6)]],                   \
                   uint3 group [[threadgroup_position_in_grid]],              \
                   uint simd_lane [[thread_index_in_simdgroup]],              \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {      \
    threadgroup float input_sums[Sums];                                       \
    input += ulong(group.z) * 32 * params.input_size;                         \
    Auxiliary += ulong(group.z) * 32 * params.output_size;                    \
    output += ulong(group.z) * 32 * params.output_size;                       \
    uint tiles = params.output_size / TileN;                                  \
    for (uint tile = group.x; tile < tiles; tile += params.persistent_groups) {\
      TileCall(input, weights, scales, biases, output, weights, scales,       \
               biases, Auxiliary, params.output_size, params.input_size,      \
               input_sums, tile * TileN, simd_lane, simd_group);              \
    }                                                                         \
  }

Q4_DECODE_SLAB_AFFINE(decode_linear_q4_n64_m64_slab,
                      (q4_mpp_tile_batched<32, 64, false, false, 256>), 256, 64)
Q4_DECODE_SLAB_AFFINE(decode_linear_q4_n128_m64_slab,
                      (q4_mpp_tile_batched<32, 128, false, false, 256>), 256, 128)
Q4_DECODE_SLAB_AFFINE(decode_linear_q4_n256_m64_slab,
                      (q4_mpp_tile_batched<32, 256, false, false>), 256, 256)
Q4_DECODE_SLAB_AUXILIARY(decode_linear_q4_n128_residual_m64_slab, residual,
                         (q4_mpp_tile_batched<32, 128, false, true, 256>),
                         256, 128)
Q4_DECODE_SLAB_AUXILIARY(decode_linear_q4_n256_up_silu_m64_slab, gate,
                         (q4_mpp_tile_batched<32, 256, false, false, 256, true>),
                         256, 256)
#undef Q4_DECODE_SLAB_AFFINE
#undef Q4_DECODE_SLAB_AUXILIARY
#undef Q4_DECODE_AFFINE
#undef Q4_DECODE_AUXILIARY
#undef Q4_DECODE_GATE_UP
