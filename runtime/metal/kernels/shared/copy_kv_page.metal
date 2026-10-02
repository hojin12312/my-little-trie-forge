#include <metal_stdlib>
using namespace metal;

// One plane of one sealed KV Page32. Host views restrict both source and
// destination to the exact page slice; all layer/scale planes use this copy.
kernel void copy_kv_page_words(device const uint *source [[buffer(0)]],
                               device uint *destination [[buffer(1)]],
                               constant uint &word_count [[buffer(2)]],
                               uint index [[thread_position_in_grid]],
                               uint grid_size [[threads_per_grid]]) {
  for (uint word = index; word < word_count; word += grid_size)
    destination[word] = source[word];
}
