#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"
#include "ops/LinearQ8.hpp"

#include <cstdint>

namespace splash::ops {

// Q8 token embedding tables are row-major signed int8 codes with fp32 scales
// and fp32 folded biases, one parameter per 64-input group, so the gather
// kernel can address token rows directly. Compact tables hold bf16 s and z
// instead (see Q8DenseProjection); the kernel rebuilds b' = fma(128, s, z).
struct Q8EmbeddingProjection final {
  metal::MetalBuffer weights;
  metal::MetalBuffer scales;
  metal::MetalBuffer biases;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  Q8Parameters parameters = Q8Parameters::Float32Folded;
};

// A token embedding table in either package packing. The package format
// decides which side is populated.
struct EmbeddingTable final {
  Q4Projection q4;
  Q8EmbeddingProjection q8;
  bool quantized8 = false;
};

class Embedding final {
public:
  static void add(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                  const EmbeddingTable &table, metal::MetalBuffer output,
                  uint32_t rows);
};

} // namespace splash::ops
