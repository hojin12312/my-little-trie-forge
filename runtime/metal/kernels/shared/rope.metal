#include "metal/abi/KernelABI.h"

kernel void rope_build_tables(
    device const uint *target_positions [[buffer(0)]],
    device const uint *draft_positions [[buffer(1)]],
    device const float *target_inverse_frequencies [[buffer(2)]],
    device const float *draft_inverse_frequencies [[buffer(3)]],
    device float *target_cosine [[buffer(4)]],
    device float *target_sine [[buffer(5)]],
    device float *draft_cosine [[buffer(6)]],
    device float *draft_sine [[buffer(7)]],
    constant RopeTableParams &params [[buffer(8)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  // Target rows carry (t, h, w) positions; Qwen3.5's interleaved M-RoPE
  // assigns frequency i to axis i % 3. Text rows repeat one position.
  const uint target_elements = params.target_rows * 32;
  for (uint element = index; element < target_elements; element += grid_size) {
    const uint row = element / 32;
    const uint dim = element % 32;
    const float angle = float(target_positions[row * 3 + dim % 3]) *
                        target_inverse_frequencies[dim];
    target_cosine[element] = cos(angle);
    target_sine[element] = sin(angle);
  }
  const uint draft_elements = params.draft_rows * 64;
  for (uint element = index; element < draft_elements; element += grid_size) {
    const uint row = element / 64;
    const uint dim = element % 64;
    const float angle =
        float(draft_positions[row]) * draft_inverse_frequencies[dim];
    draft_cosine[element] = cos(angle);
    draft_sine[element] = sin(angle);
  }
}
