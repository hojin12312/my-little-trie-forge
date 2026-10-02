#include "model/QwenVision.hpp"

#include <string>
#include <utility>

namespace splash::model {
namespace {

void validateLayout(const ops::VisionLayout &layout) {
  if (!layout.depth || !layout.hiddenSize || !layout.patchDimension ||
      !layout.intermediateSize || !layout.paddedIntermediateSize ||
      !layout.mergedHiddenSize || !layout.outputHiddenSize || !layout.heads ||
      !layout.headDimension || !layout.positionGridSide || !layout.patchSize ||
      !layout.spatialMerge ||
      layout.heads * layout.headDimension != layout.hiddenSize ||
      layout.paddedIntermediateSize < layout.intermediateSize ||
      layout.mergedHiddenSize !=
          layout.hiddenSize * layout.spatialMerge * layout.spatialMerge ||
      layout.patchDimension != 3 * 2 * layout.patchSize * layout.patchSize) {
    throw WeightStoreError("Qwen vision layout is inconsistent");
  }
}

ops::VisionAffine readAffine(WeightFile &file, uint32_t outputSize,
                             uint32_t inputSize, std::string_view label) {
  return {
      file.section(checkedWeightMultiply(
                       checkedWeightMultiply(outputSize, inputSize,
                                             "vision weight elements"),
                       kBFloat16Bytes, "vision weight bytes"),
                   label),
      file.section(checkedWeightMultiply(outputSize, kBFloat16Bytes,
                                         "vision bias bytes"),
                   label),
  };
}

ops::VisionNorm readNorm(WeightFile &file, uint32_t width,
                         std::string_view label) {
  const uint64_t bytes =
      checkedWeightMultiply(width, kBFloat16Bytes, "vision norm bytes");
  return {file.section(bytes, label), file.section(bytes, label)};
}

} // namespace

QwenVisionWeights loadQwenVisionWeights(metal::MetalBackend &backend,
                                        const std::filesystem::path &directory,
                                        ops::VisionLayout layout) {
  validateLayout(layout);
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  QwenVisionWeights result;
  result.tensors.layout = layout;
  result.tensors.blocks.reserve(layout.depth);

  WeightFile file(backend, directory / "model.bin", "vision/model.bin",
                  kVisionMagic, layout.depth, 0);
  result.tensors.patchEmbedding =
      readAffine(file, layout.hiddenSize, layout.patchDimension, "patch-embed");
  result.tensors.positionTable = file.section(
      checkedWeightMultiply(
          checkedWeightMultiply(
              checkedWeightMultiply(layout.positionGridSide,
                                    layout.positionGridSide,
                                    "vision position count"),
              layout.hiddenSize, "vision position elements"),
          kBFloat16Bytes, "vision position bytes"),
      "position-table");
  for (uint32_t blockIndex = 0; blockIndex < layout.depth; ++blockIndex) {
    ops::VisionBlock block;
    block.norm1 = readNorm(file, layout.hiddenSize, "norm1");
    block.qkv = readAffine(file, 3 * layout.hiddenSize, layout.hiddenSize, "qkv");
    block.projection =
        readAffine(file, layout.hiddenSize, layout.hiddenSize, "proj");
    block.norm2 = readNorm(file, layout.hiddenSize, "norm2");
    block.upProjection = readAffine(file, layout.paddedIntermediateSize,
                                    layout.hiddenSize, "fc1");
    block.downProjection = readAffine(file, layout.hiddenSize,
                                      layout.paddedIntermediateSize, "fc2");
    result.tensors.blocks.push_back(std::move(block));
  }
  result.tensors.mergerNorm = readNorm(file, layout.hiddenSize, "merger-norm");
  result.tensors.mergerUpProjection = readAffine(
      file, layout.mergedHiddenSize, layout.mergedHiddenSize, "merger-fc1");
  result.tensors.mergerDownProjection = readAffine(
      file, layout.outputHiddenSize, layout.mergedHiddenSize, "merger-fc2");
  file.finish();
  result.files.push_back(file.record());

  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace splash::model
