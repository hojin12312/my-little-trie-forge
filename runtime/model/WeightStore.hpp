#pragma once

#include "metal/MetalBackend.hpp"
#include "ops/Embedding.hpp"
#include "ops/Linear.hpp"
#include "ops/LinearQ8.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::model {

inline constexpr uint32_t kQ4GroupElements = 64;
inline constexpr uint64_t kBFloat16Bytes = 2;

inline constexpr uint64_t kWeightFileAlignment = 16 * 1024;
inline constexpr uint32_t kQ4StorageN = 256;

class WeightStoreError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

struct WeightFileRecord final {
  std::string relativePath;
  std::string magic;
  uint32_t layer = 0;
  uint32_t type = 0;
  uint64_t declaredBytes = 0;
};

// A read-only mmap with one no-copy Metal base buffer.  Sections are checked,
// aligned views that retain the mapping; no model loader owns raw mmap state.
class WeightFile final {
public:
  WeightFile(metal::MetalBackend &backend, std::filesystem::path path,
             std::string relativePath, std::string_view expectedMagic,
             uint32_t expectedLayer, uint32_t expectedType);
  ~WeightFile();

  WeightFile(const WeightFile &) = delete;
  WeightFile &operator=(const WeightFile &) = delete;

  [[nodiscard]] metal::MetalBuffer section(uint64_t bytes,
                                            std::string_view label = {});
  void finish();
  [[nodiscard]] const WeightFileRecord &record() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] uint64_t checkedWeightMultiply(uint64_t left, uint64_t right,
                                             std::string_view description);
[[nodiscard]] uint64_t q4PackedBytes(uint32_t outputSize,
                                     uint32_t inputSize);
void validateQ4Layout(uint32_t outputSize, uint32_t inputSize);

[[nodiscard]] ops::Q4Projection
readQ4Projection(WeightFile &file, metal::MetalBackend &backend,
                 uint32_t outputSize, uint32_t inputSize,
                 std::string_view label);

// Embedding weights, scales and biases are independently aligned sections
// so token gather can bind each table directly.
[[nodiscard]] ops::Q4Projection
readQ4ProjectionComponents(WeightFile &file, uint32_t outputSize,
                           uint32_t inputSize, std::string_view label);

[[nodiscard]] ops::Q8Projection
readQ8Projection(WeightFile &file, metal::MetalBackend &backend,
                 uint32_t outputSize, uint32_t inputSize,
                 std::string_view label);

// Dense Q8 sections store signed int8 codes with fp32 scales and fp32
// folded biases (b' = z + 128*s), unlike the MoE router's bf16 parameters.
// Compact sections (splash-packed-q8c) store the checkpoint's bf16 scales and
// zero points z at the same index instead, half the parameter bytes.
[[nodiscard]] uint64_t q8DensePackedBytes(
    uint32_t outputSize, uint32_t inputSize,
    ops::Q8Parameters parameters = ops::Q8Parameters::Float32Folded);
[[nodiscard]] ops::Q8DenseProjection
readQ8DenseProjection(WeightFile &file, metal::MetalBackend &backend,
                      uint32_t outputSize, uint32_t inputSize,
                      std::string_view label,
                      ops::Q8Parameters parameters =
                          ops::Q8Parameters::Float32Folded);

// Q8 token embeddings are row-major with independently aligned weights,
// scales and biases sections, like the Q4 component reader.
[[nodiscard]] ops::Q8EmbeddingProjection
readQ8DenseProjectionComponents(WeightFile &file, uint32_t outputSize,
                                uint32_t inputSize, std::string_view label,
                                ops::Q8Parameters parameters =
                                    ops::Q8Parameters::Float32Folded);

[[nodiscard]] ops::ExpertQ4Projection
readExpertQ4Projection(WeightFile &file, uint32_t experts,
                       uint32_t outputSize, uint32_t inputSize,
                       std::string_view label);

[[nodiscard]] std::string
weightManifestFingerprint(std::span<const WeightFileRecord> records);

} // namespace splash::model
