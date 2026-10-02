#pragma once

#include "metal/CommandGraph.hpp"

#include <cstdint>

namespace splash::ops {

struct RoPETableShape final {
  uint32_t targetRows = 0;
  uint32_t draftRows = 0;
};

class RoPE final {
public:
  static void addTables(
      metal::CommandGraph &graph, metal::MetalBuffer targetPositions,
      metal::MetalBuffer draftPositions,
      metal::MetalBuffer targetInverseFrequencies,
      metal::MetalBuffer draftInverseFrequencies,
      metal::MetalBuffer targetCosine, metal::MetalBuffer targetSine,
      metal::MetalBuffer draftCosine, metal::MetalBuffer draftSine,
      RoPETableShape shape, uint32_t maximumRows);
};

} // namespace splash::ops
