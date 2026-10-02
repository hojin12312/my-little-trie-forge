#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct Q4Params {
  uint32_t output_size;
  uint32_t input_size;
  uint32_t persistent_groups;
};

static_assert(sizeof(Q4Params) == 12,
              "Q4 decode projection parameters are 12 bytes on both sides");

// Separate from ops::LinearMatrix so host-only fields cannot change the ABI.
struct Q4PrefillParams {
  uint32_t output_size;
  uint32_t input_size;
};

static_assert(sizeof(Q4PrefillParams) == 8,
              "Q4 prefill projection parameters are 8 bytes on both sides");

// Dense Q8 projections keep signed int8 weights in StorageN=256 order with
// fp32 group scales and shift-folded fp32 biases. A16 kernels consume bf16
// activations with fp32 group sums; A8 kernels consume symmetric int8
// activations with float2 {scale, scale * quantized sum} terms per group.
struct Q8DecodeParams {
  uint32_t output_size;
  uint32_t input_size;
  uint32_t persistent_groups;
};

static_assert(sizeof(Q8DecodeParams) == 12,
              "Q8 decode projection parameters are 12 bytes on both sides");

struct SplitQ8Params {
  uint32_t output_size;
  uint32_t input_size;
  uint32_t splits;
  uint32_t rows;
};

static_assert(sizeof(SplitQ8Params) == 16);

// Active reduction uses the unchanged maximum scratch allocation and pitch.
struct ActiveSplitQ8ReduceParams {
  uint32_t output_size;
  uint32_t active_rows;
  uint32_t partial_pitch_rows;
};
static_assert(sizeof(ActiveSplitQ8ReduceParams) == 12);

struct Q8PrefillParams {
  uint32_t output_size;
  uint32_t input_size;
};

static_assert(sizeof(Q8PrefillParams) == 8,
              "Q8 prefill projection parameters are 8 bytes on both sides");

// Standalone activation quantization and the fused RMS-norm variants share
// this layout. rows is the physical row count: prefill uses it to zero the
// padded tail of each 32-row tile; decode uses it to index the [group][row]
// term layout.
struct Q8QuantizeParams {
  uint32_t input_size;
  uint32_t rows;
};

static_assert(sizeof(Q8QuantizeParams) == 8,
              "Q8 quantization parameters are 8 bytes on both sides");
