#include "model/Qwen3_8.hpp"

#include <string_view>

namespace splash::model {
namespace {

constexpr std::string_view kHeadMagic = "MDFL0002";

void validateLayout(const Qwen3_8Layout &layout) {
  if (!layout.maximumContextTokens || !layout.layers || !layout.hiddenSize ||
      !layout.vocabularySize || !layout.packedGdnWidth ||
      !layout.packedFullWidth || !layout.convolutionDimension ||
      !layout.gdnKeyHeads || !layout.gdnValueHeads ||
      !layout.gdnHeadDimension || !layout.attentionWidth ||
      !layout.intermediateSize || !layout.attentionQueryHeads ||
      !layout.attentionKvHeads || !layout.attentionHeadDimension ||
      !layout.rotaryPairs || !(layout.rotaryTheta > 0.0F) ||
      !layout.fullAttentionPeriod) {
    throw WeightStoreError("Qwen3.8 layout contains a zero dimension");
  }
  validateQ4Layout(layout.packedGdnWidth, layout.hiddenSize);
  validateQ4Layout(layout.packedFullWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth);
  validateQ4Layout(layout.intermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.intermediateSize);
  validateQ4Layout(layout.vocabularySize, layout.hiddenSize);
  if (layout.attentionQueryHeads * layout.attentionHeadDimension !=
      layout.attentionWidth) {
    throw WeightStoreError("Qwen3.8 attention layout is inconsistent");
  }
}

} // namespace

Qwen3_8Weights loadQwen3_8Weights(metal::MetalBackend &backend,
                                  const std::filesystem::path &directory,
                                  Qwen3_8Layout layout,
                                  bool denseQuantized8,
                                  ops::Q8Parameters q8Parameters) {
  validateLayout(layout);
  return loadQwenTargetWeights<Qwen3_8Weights>(
      backend, directory, layout, kHeadMagic, denseQuantized8,
      [&](WeightFile &file, bool quantized8, Qwen3_8LayerWeights &layer) {
        layer.gateProjection = readDenseProjection(
            file, backend, layout.intermediateSize, layout.hiddenSize,
            "mlp-gate", quantized8, q8Parameters);
        layer.upProjection = readDenseProjection(
            file, backend, layout.intermediateSize, layout.hiddenSize,
            "mlp-up", quantized8, q8Parameters);
        layer.downProjection = readDenseProjection(
            file, backend, layout.hiddenSize, layout.intermediateSize,
            "mlp-down", quantized8, q8Parameters);
      },
      q8Parameters);
}

} // namespace splash::model
