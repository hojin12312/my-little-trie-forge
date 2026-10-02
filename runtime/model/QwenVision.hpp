#pragma once

#include "WeightStore.hpp"
#include "ops/Vision.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {

// The current Qwen targets share one vision-tower architecture. Its language
// projection width belongs to VisionLayout, so the loader is independent of a
// particular text model and validates the package-selected width.
struct QwenVisionWeights final {
  ops::VisionWeights tensors;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
  std::string manifestFingerprintSha256;
};

inline constexpr std::string_view kVisionMagic = "MDFV0001";

[[nodiscard]] QwenVisionWeights
loadQwenVisionWeights(metal::MetalBackend &backend,
                      const std::filesystem::path &directory,
                      ops::VisionLayout layout = {});

} // namespace splash::model
