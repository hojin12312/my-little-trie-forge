#pragma once

#include "metal/abi/ExecutionGeometry.h"
#include "metal/MetalBackend.hpp"

#include <span>
#include <stdexcept>
#include <vector>

namespace splash::ops {

// Batched kernels bind one buffer per lane for each per-lane state tensor and
// index them by lane, so every slot must be bound even when fewer lanes run.
inline void appendLaneBindings(std::vector<metal::MetalBuffer> &bindings,
                               std::span<const metal::MetalBuffer> first,
                               std::span<const metal::MetalBuffer> second) {
  constexpr size_t lanes = SPLASH_MAXIMUM_BATCH_WIDTH;
  if (first.size() != lanes || second.size() != lanes)
    throw std::invalid_argument("lane bindings must cover every lane");
  bindings.insert(bindings.end(), first.begin(), first.end());
  bindings.insert(bindings.end(), second.begin(), second.end());
}

} // namespace splash::ops
