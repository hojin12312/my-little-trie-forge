#pragma once

#include "DFlashDraft.hpp"
#include "Model.hpp"
#include "Qwen3_6Moe.hpp"
#include "Qwen3_8.hpp"
#include "ops/Vision.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>

namespace splash::model {

using TargetLayout = std::variant<Qwen3_8Layout, Qwen3_6MoeLayout>;

// Package metadata validated before weight buffers are loaded. The engine
// consumes capabilities; model loading consumes the concrete layouts.
struct ModelDescriptor final {
  std::string name;
  TargetLayout target;
  DFlashDraftLayout draft;
  ops::VisionLayout vision;
  ModelCapabilities capabilities;
  kv::Layout targetKvLayout;
  CompositeStateLayout stateLayout;
  // Schema-5 packages store every dense target projection, the logits head
  // and the token embedding as dense Q8 tensors instead of Q4, with fp32
  // (s, b') parameters (splash-packed-q8) or compact bf16 (s, z) parameters
  // (splash-packed-q8c).
  bool denseQuantized8 = false;
  ops::Q8Parameters denseQ8Parameters = ops::Q8Parameters::Float32Folded;
  // Exact bytes parsed during package inspection, including artifact digests.
  // Synthetic descriptors retain zero; this is separate from layout identity.
  std::array<uint8_t, 32> packageManifestSha256{};

  [[nodiscard]] bool valid() const noexcept;
};

// Derives the capabilities and cache layouts the engine consumes from the
// concrete target and draft layouts.
[[nodiscard]] ModelDescriptor makeModelDescriptor(std::string name,
                                                  TargetLayout target,
                                                  DFlashDraftLayout draft,
                                                  ops::VisionLayout vision);
[[nodiscard]] ModelDescriptor
inspectModelPackage(const std::filesystem::path &root);

} // namespace splash::model
