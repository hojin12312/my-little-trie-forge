#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct FullPrefillParams {
  uint32_t tokens;
  uint32_t cache_stride;
  uint32_t row_stride;
};

static_assert(sizeof(FullPrefillParams) == 12,
              "Full attention prefill parameters are 12 bytes on both sides");

struct FullDecodeBatchParams {
  uint32_t tokens;
  uint32_t cache_stride;
  uint32_t row_stride;
  uint32_t lanes;
};

static_assert(sizeof(FullDecodeBatchParams) == 16,
              "Full attention verify parameters are 16 bytes on both sides");

// One full-attention layer's paged KV. Every current row is written directly
// into its final Q8 page slot before attention. Prefill and verify both read
// all visible history from the same paged representation. Decode accepts rows
// only by advancing committed_tokens; the next command overwrites rejected
// slots.
struct SplashChunkedPrefillParams {
  uint32_t committed_tokens;
  uint32_t chunk_tokens;
  uint32_t chunk_stride;
  uint32_t page_table_entries;
  uint32_t physical_page_count;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
};

static_assert(sizeof(SplashChunkedPrefillParams) == 32,
              "Q8 chunked store parameters are 32 bytes on both sides");

// Prefill divides each query tile's visible Page32 history into balanced
// splits. The same count and partition rule are used by split and reduce.
struct SplashQ8PrefillAttentionParams {
  uint32_t committed_tokens;
  uint32_t rows;
  uint32_t chunk_stride;
  uint32_t page_table_entries;
  uint32_t physical_page_count;
  uint32_t split_count;
  uint32_t reserved0;
  uint32_t reserved1;
};

static_assert(sizeof(SplashQ8PrefillAttentionParams) == 32,
              "Q8 prefill attention parameters are 32 bytes on both sides");

struct SplashQ8VerifyAttentionParams {
  uint32_t committed_tokens;
  uint32_t active_rows;
  uint32_t chunk_stride;
  uint32_t page_table_entries;
  uint32_t physical_page_count;
  // Filled from the plan: this lane's history-scaled split count and the
  // plan-wide slot stride that every lane's partials use.
  uint32_t split_count;
  uint32_t slot_splits;
  uint32_t reserved2;
};

static_assert(sizeof(SplashQ8VerifyAttentionParams) == 32,
              "Q8 verify attention parameters are 32 bytes on both sides");
