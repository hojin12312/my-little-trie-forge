#include "metal/abi/KernelABI.h"

kernel void capture_target_hidden(device const bfloat *input [[buffer(0)]],
                                  device bfloat *captured [[buffer(1)]],
                                  constant CaptureParams &params [[buffer(2)]],
                                  uint index [[thread_position_in_grid]],
                                  uint grid_size [[threads_per_grid]]) {
  for (uint element = index; element < params.rows * params.hidden_width;
       element += grid_size) {
    uint row = element / params.hidden_width;
    uint dim = element % params.hidden_width;
    captured[(params.destination_start + row) * params.target_width +
             params.slot * params.hidden_width + dim] =
        input[(params.source_start + row) * params.hidden_width + dim];
  }
}
