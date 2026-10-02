#pragma once

#include "metal/kernels/common/paged_attention_tile.h"

// P2 diagnostic: one threadgroup owns the current Page32 K/V codes for the
// entire QK/softmax/PV step. Only INT8 is instantiated: two 8 KiB operands
// fit beside the existing score/probability/statistics tile.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile>
inline void splash_paged_attention_stagepage_tile(
    device bfloat *tile_queries, device int8_t *cache_keys,
    device const float *key_scales_buffer, device int8_t *cache_values,
    device const float *value_scales_buffer, device const uint *page_table,
    uint kv_head, uint committed_tokens, uint active_rows, uint splits,
    uint split, device float *partials, device float *statistics, ulong slot,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    threadgroup int8_t *staged_keys, threadgroup int8_t *staged_values,
    uint thread_index) {
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr ushort N = SplashQ8PageTokens;
  constexpr ushort D = SplashQ8HeadDimension;
  constexpr uint PageElements = uint(N) * D;
  const uint visible_tokens = committed_tokens + active_rows;
  const uint pages = splash_attention_pages(visible_tokens);
  const uint per_split = splash_attention_pages_per_split(pages, splits);
  const uint page_begin = split * per_split;
  if (page_begin >= pages) return;
  const uint page_end = min(pages, page_begin + per_split);
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto qt = tensor(tile_queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto st = tensor(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto kt = tensor(staged_keys, dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto vt = tensor(staged_values, dextents<int, 2>{N, D}, array<int, 2>{1, N});
  auto q0 = qt.slice<D, M>(0, 0);
  auto p0 = pt.slice<N, M>(0, 0);
  auto k0 = kt.slice<D, N>(0, 0);
  auto v0 = vt.slice<N, D>(0, 0);
  constexpr auto qk_descriptor =
      matmul2d_descriptor(M, N, D, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, N, false, true, true,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto running = pv.template get_destination_cooperative_tensor<
      decltype(p0), decltype(v0), float>();
  const bool running_full =
      uint(running.get_capacity()) * 256u == uint(M) * D;
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index)
    if (running_full || running.is_valid_element(index))
      running[index] = 0.0f;

  for (uint page = page_begin; page < page_end; ++page) {
    const uint physical = page_table[page];
    const uint token_start = page * N;
    device int8_t *key_source =
        cache_keys + splash_q8_key_index<KVHeads>(physical, kv_head, 0, 0);
    device int8_t *value_source =
        cache_values + splash_q8_value_index<KVHeads>(physical, kv_head, 0, 0);
    for (uint index = thread_index; index < PageElements; index += 256) {
      staged_keys[index] = key_source[index];
      staged_values[index] = value_source[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong scale_index =
        splash_q8_scale_index<KVHeads>(physical, kv_head, 0);
    device const float *key_scales = key_scales_buffer + scale_index;
    device const float *value_scales = value_scales_buffer + scale_index;

    auto page_scores = qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(k0), float>();
    qk.run(q0, k0, page_scores);
    page_scores.store(st.slice<N, M>(0, 0));
    if (thread_index == 0)
      atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    splash_attention_page_softmax<QueryHeadsPerKVHead, RowsPerTile,
                                   true, true>(
        scores, probabilities, row_max, row_sum, previous_scale, rescale,
        reinterpret_cast<device const float4 *>(key_scales),
        reinterpret_cast<device const float4 *>(value_scales), token_start,
        visible_tokens, committed_tokens, active_rows, thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(rescale, memory_order_relaxed)) {
#pragma unroll
      for (ushort index = 0; index < running.get_capacity(); ++index) {
        if (!running_full && !running.is_valid_element(index)) continue;
        auto coordinates = running.get_multidimensional_index(index);
        running[index] *= previous_scale[coordinates[1]];
      }
    }
    pv.run(p0, v0, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  auto target = tensor(partials + slot * M * D,
                       dextents<int, 2>{D, M}, array<int, 2>{1, D});
  running.store(target.slice<D, M>(0, 0));
  if (thread_index < M) {
    statistics[(slot * M + thread_index) * 2] = row_max[thread_index];
    statistics[(slot * M + thread_index) * 2 + 1] = row_sum[thread_index];
  }
}
