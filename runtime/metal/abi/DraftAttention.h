#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct DraftConvBatchParams {
  uint32_t groups;
  uint32_t finish;
  uint32_t lanes;
};

static_assert(sizeof(DraftConvBatchParams) == 12,
              "Draft convolution parameters are 12 bytes on both sides");

struct DraftQkvBatchParams {
  uint32_t groups;
  uint32_t lanes;
};

static_assert(sizeof(DraftQkvBatchParams) == 8,
              "Draft QKV parameters are 8 bytes on both sides");

struct DraftAttentionBatchParams {
  uint32_t cache_stride;
  uint32_t splits;
  uint32_t lanes;
  uint32_t cache_length[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(DraftAttentionBatchParams) ==
                  12 + 4 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Draft attention parameters are 28 bytes at width four on both sides");

struct DraftContextParams {
  uint32_t tokens;
  uint32_t cache_stride;
  uint32_t start_position;
};

static_assert(sizeof(DraftContextParams) == 12,
              "Draft context prefill parameters are 12 bytes on both sides");

struct DraftContextBatchParams {
  uint32_t cache_stride;
  uint32_t lanes;
  uint32_t start_position[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(DraftContextBatchParams) ==
                  8 + 4 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Draft context commit parameters are 24 bytes at width four on both sides");

struct CaptureParams {
  uint32_t rows;
  uint32_t slot;
  uint32_t source_start;
  uint32_t destination_start;
  uint32_t hidden_width;
  uint32_t target_width;
};

static_assert(sizeof(CaptureParams) == 24,
              "Target hidden capture parameters are 24 bytes on both sides");

struct LastHiddenRowsParams {
  uint32_t rows;
  uint32_t width;
};

static_assert(sizeof(LastHiddenRowsParams) == 8,
              "Last hidden row parameters are 8 bytes on both sides");
