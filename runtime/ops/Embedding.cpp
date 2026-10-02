#include "ops/Embedding.hpp"

#include "metal/abi/Embedding.h"

#include <stdexcept>
#include <utility>

namespace splash::ops {
namespace {

// Mirrors the package's dense quantization group; ops must not depend on the
// model weight loader.
constexpr uint32_t kQuantGroupElements = 64;

const char *embeddingPipeline(uint32_t hiddenSize, bool quantized8,
                              bool compact = false) {
  switch (hiddenSize) {
  case 5120:
    if (quantized8)
      return compact ? "embedding_q8_h5120_bf16p" : "embedding_q8_h5120";
    return "embedding_q4_h5120";
  case 2048:
    if (quantized8)
      throw std::invalid_argument("no compiled Q8 embedding shape for h2048");
    return "embedding_q4_h2048";
  default:
    throw std::invalid_argument("unsupported compiled embedding shape");
  }
}

} // namespace

void Embedding::add(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                    const EmbeddingTable &table, metal::MetalBuffer output,
                    uint32_t rows) {
  if (table.quantized8) {
    const Q8EmbeddingProjection &projection = table.q8;
    if (!rows || !projection.outputSize || !projection.inputSize)
      throw std::invalid_argument("invalid Q8 embedding shape");
    if (projection.inputSize % kQuantGroupElements)
      throw std::invalid_argument("Q8 embedding input is not group aligned");
    if (projection.parameters != Q8Parameters::Float32Folded &&
        projection.parameters != Q8Parameters::BFloat16)
      throw std::invalid_argument("invalid Q8 embedding parameter format");
    const bool compact = projection.parameters == Q8Parameters::BFloat16;
    const uint32_t hiddenGroups = projection.inputSize / kQuantGroupElements;
    const uint64_t parameterBytes = uint64_t{projection.outputSize} *
        hiddenGroups * (compact ? 2 : 4);
    if (projection.scales.sizeBytes() < parameterBytes ||
        projection.biases.sizeBytes() < parameterBytes)
      throw std::invalid_argument("Q8 embedding parameters are undersized");
    const Q4EmbeddingParams params{rows, projection.outputSize};
    graph.add(embeddingPipeline(projection.inputSize, true, compact),
              {std::move(tokens), projection.weights, projection.scales,
               projection.biases, std::move(output)},
              params, {hiddenGroups, 1, 1});
    return;
  }
  const Q4Projection &projection = table.q4;
  if (!rows || !projection.outputSize || !projection.inputSize)
    throw std::invalid_argument("invalid Q4 embedding shape");
  const uint32_t hiddenGroups = (projection.inputSize + 127) / 128;
  const Q4EmbeddingParams params{rows, projection.outputSize};
  graph.add(embeddingPipeline(projection.inputSize, false),
            {std::move(tokens), projection.weights, projection.scales,
             projection.biases, std::move(output)},
            params, {hiddenGroups, 1, 1});
}

} // namespace splash::ops
