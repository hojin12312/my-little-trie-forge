#pragma once

#include "metal/DeviceCapabilities.hpp"
#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <string_view>
#include <vector>

namespace splash::ops {

// Dense Q8 projections store signed int8 weights in the StorageN=256 packed
// order with fp32 scales and fp32 biases. The package converter shifts the
// affine uint8 codes by -128 and folds the offset into the bias
// (b' = b + 128 * s), so kernels only see signed operands — MPP has no
// mixed-sign i8 x ui8 matmul. The field layout matches Q8Projection, but the
// parameter tensors here are fp32 while the MoE router's are bf16.
//
// Compact parameters keep the checkpoint's bf16 scale s and zero point z at
// the same index instead (biases then holds z, not b'). The A16 kernels
// rebuild b' = fma(128, s, z) in fp32, which is bit-identical to the folded
// value, so a compact projection produces the same output bytes. The A8
// prefill kernels read both forms the same way; A8 decode plans reject
// compact parameters.
enum class Q8Parameters : uint8_t { Float32Folded, BFloat16 };

struct Q8DenseProjection final {
  metal::MetalBuffer weights;
  metal::MetalBuffer scales;
  metal::MetalBuffer biases;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  Q8Parameters parameters = Q8Parameters::Float32Folded;
};

// A dense projection in either package packing. The package format decides
// which side is populated; a model never mixes packings within one package.
struct DenseProjection final {
  Q4Projection q4;
  Q8DenseProjection q8;
  bool quantized8 = false;
};

// A16 multiplies bf16 activations by int8 weights and consumes the fp32
// group sums the Q4 input pipelines produce (prefill_linear_q4_sums32 or the
// fused norms). A8 quantizes activations into row-major int8 codes plus
// float2 {sx, sx * qxsum} terms per 64-input group and runs exact int32
// dots with an fp32 epilogue.
enum class Q8Activation : uint8_t { A16, A8 };

// Compute tiles over the StorageN=256 packing. Paired tiles pipeline two
// quant groups; Paired256 is the four-simdgroup N256 paired tile. Paired
// tiles serve one decode lane, except the A16 decode residual: Paired128 also
// at 16 rows and Paired64 only at 32 rows. N64 exists only for the A16 decode
// residual at 24 and 32 rows, where N128 leaves one threadgroup per core on
// narrow outputs. Split and simdgroup tiles are Apple9-only and do not exist
// for Q8.
enum class Q8Tile : uint8_t { N128, N256, Paired128, Paired256, N64, Paired64 };

struct Q8Config final {
  Q8Tile tile = Q8Tile::N128;
  // Decode persistent threadgroup count. Prefill uses its matrix grid and
  // requires zero here.
  uint32_t groups = 0;
  // Cooperative scope of one tile: the decode m24 and paired N256 kernels
  // run four simdgroups; everything else runs eight.
  LinearSimdgroups simdgroups = LinearSimdgroups::Eight;
  Q8Activation activation = Q8Activation::A16;
  bool operator==(const Q8Config &) const = default;
};

struct Q8Choice final {
  LinearWorkload workload;
  Q8Config configuration;
};

// A8 input scratch: the quantized int8 activations and the float2 terms.
// Prefill tiles the terms as [row tile][group][row in tile]; decode keeps a
// flat [group][row] table. Reused serially within one command stream.
struct Q8Scratch final {
  metal::MetalBuffer input;
  metal::MetalBuffer terms;
};
struct Q8ScratchSize final {
  uint64_t input = 0, terms = 0;
  [[nodiscard]] uint64_t bytes() const noexcept { return input + terms; }
};

class Q8Plan final {
public:
  [[nodiscard]] LinearWorkload workload() const noexcept { return workload_; }
  [[nodiscard]] Q8Config configuration() const noexcept { return config_; }
  [[nodiscard]] Q8Activation activation() const noexcept {
    return config_.activation;
  }
  [[nodiscard]] uint32_t storageRows() const noexcept;
  [[nodiscard]] uint32_t tileColumns() const noexcept;
  [[nodiscard]] uint32_t threadsPerThreadgroup() const noexcept;
  // A16 input side channel: fp32 group sums for prefill, nothing for decode.
  [[nodiscard]] uint64_t sumsBytes() const noexcept;
  // A8 input side channel sized for storageRows rows of the input matrix.
  [[nodiscard]] Q8ScratchSize scratchSize() const noexcept;
  [[nodiscard]] uint64_t gateScratchBytes() const noexcept;
  // The up projection's down-stream side channel: fp32 sums for A16, the
  // quantized int8 input plus float2 terms over the output width for A8.
  [[nodiscard]] uint64_t downSumsBytes() const noexcept;
  [[nodiscard]] Q8ScratchSize downScratchSize() const noexcept;
  [[nodiscard]] std::string_view pipeline() const noexcept { return pipeline_; }
  [[nodiscard]] std::string_view secondPipeline() const noexcept {
    return secondPipeline_;
  }

private:
  friend class Q8Linear;
  Q8Plan(LinearWorkload workload, Q8Config config);
  LinearWorkload workload_;
  Q8Config config_;
  std::string_view pipeline_;
  std::string_view secondPipeline_;
};

// The plan defines which fields are used and how much scratch they require.
// For A8, input/terms hold the quantized activation produced by an explicit
// quantize dispatch or a fused norm, and inputPrepared marks that the side
// channel is already valid.
struct Q8Buffers final {
  metal::MetalBuffer input;
  metal::MetalBuffer output;
  metal::MetalBuffer sums;
  metal::MetalBuffer residual;
  metal::MetalBuffer gateScratch;
  metal::MetalBuffer terms;
  metal::MetalBuffer downInput;
  metal::MetalBuffer downTerms;
  metal::MetalBuffer downSums;
  Q8Scratch scratch{};
  bool inputPrepared = false;
};

// Residual decode/verify policy for compact Q8 packages. Split2Rounded — the
// split-K partial kernels plus one rounded reduction — is the shipping
// default; SPLASH_Q8_RESIDUAL_DECODE=direct selects the former fused epilogue
// for comparison and as a fallback. The policy applies only where the
// split-K kernels cover the shape (see Q8Linear::add); every other residual
// keeps the direct epilogue regardless.
enum class Q8ResidualDecode : uint8_t { Split2Rounded, Direct };
[[nodiscard]] inline bool q8PrefillStageParameters() noexcept {
  const char *value = std::getenv("SPLASH_DEV_PREFILL_STAGEP");
  return value && std::string_view(value) == "1";
}
[[nodiscard]] inline bool q8Split2M32SafeM16() noexcept {
  const char *value = std::getenv("SPLASH_DEV_SPLIT2_M32_SAFEM16");
  return value && std::string_view(value) == "1";
}
[[nodiscard]] inline bool q8Split2M32Sg16Profile(
    const DeviceCapabilities &device) noexcept {
  return device.deviceName == "Apple M5 Max" &&
         device.appleGpuFamily == 10 && device.gpuCoreCount == 40;
}
[[nodiscard]] inline bool q8Split2M32Sg16(
    bool profiledDefault = false) noexcept {
  const char *value = std::getenv("SPLASH_DEV_SPLIT2_M32_SG16");
  if (value && std::string_view(value) == "0") return false;
  if (value && std::string_view(value) == "1") return true;
  return profiledDefault;
}
[[nodiscard]] inline Q8ResidualDecode q8ResidualDecodePolicy() noexcept {
  const char *value = std::getenv("SPLASH_Q8_RESIDUAL_DECODE");
  if (value && std::string_view(value) == "direct")
    return Q8ResidualDecode::Direct;
  return Q8ResidualDecode::Split2Rounded;
}
[[nodiscard]] inline std::string_view q8ResidualDecodeName(
    Q8ResidualDecode policy) noexcept {
  switch (policy) {
  case Q8ResidualDecode::Split2Rounded: return "split2-rounded";
  case Q8ResidualDecode::Direct: return "direct";
  }
  return "unknown";
}

// True where Q8Linear::add dispatches the split-K residual kernels instead of
// the fused epilogue: A16 decode on compact parameters of the two production
// residual shapes, with the policy resolved to Split2Rounded. The caller must
// then provide scratch.input of q8ResidualSplit2ScratchBytes — the runtime
// routes it through LinearScratch::partials (see DecodeArena).
[[nodiscard]] inline bool q8ResidualDecodeSplit2Applies(
    const LinearWorkload &w, const Q8DenseProjection &p,
    Q8Activation activation) noexcept {
  const auto [n, k] = w.matrix;
  return w.phase == LinearPhase::Decode && activation == Q8Activation::A16 &&
         w.epilogue == LinearEpilogue::Residual &&
         p.parameters == Q8Parameters::BFloat16 &&
         q8ResidualDecodePolicy() == Q8ResidualDecode::Split2Rounded &&
         n == 5120 && (k == 6144 || k == 17408);
}
[[nodiscard]] inline uint64_t q8ResidualSplit2ScratchBytes(
    const LinearWorkload &w) noexcept {
  return uint64_t{2} * w.rows * w.matrix.outputSize * sizeof(float);
}

// Wide decode (64-row) geometry for the B5-B8 logical widths. The padded M64
// kernels collapse on the 40-core M5 Max (about 2.3x worse per row than M32);
// the measured winner runs the same persistent groups as two M32 row slabs
// selected by group.z in one dispatch (see dev/results/production-campaign-v1/
// opt-wide-decode-b1-b8). The split2 residual keeps its native M64 partials —
// every alternative measured slower on those shapes. Only reachable on builds
// with SPLASH_MAXIMUM_BATCH_WIDTH=8; SPLASH_DEV_WIDE_DECODE=native restores
// the padded M64 tiles for comparison.
[[nodiscard]] inline bool q8WideDecodeSlab() noexcept {
  const char *value = std::getenv("SPLASH_DEV_WIDE_DECODE");
  return !(value && std::string_view(value) == "native");
}

// Shared-weight campaign (dev/results/production-campaign-v1/
// opt-wide-decode-shared-weight): for lanes 5-6 (active rows 40/48) the
// single-threadgroup dual-fragment tile beats the slab by ~20-30% on the
// n256 shapes because each 64-wide weight slice is fetched once and the tail
// fragment's run hits the cache. At lanes 7-8 the dual tile's register
// pressure makes it lose to the slab, so auto selects per active width.
// SPLASH_DEV_WIDE_DECODE=slab forces the slab everywhere, =dual forces the
// dual tile whenever one exists, =native keeps the padded M64 control.
enum class Q8WideDecode : ushort { Auto, Slab, Dual, Native };
[[nodiscard]] inline Q8WideDecode q8WideDecodeMode() noexcept {
  const char *value = std::getenv("SPLASH_DEV_WIDE_DECODE");
  if (!value) return Q8WideDecode::Auto;
  const std::string_view mode(value);
  if (mode == "native") return Q8WideDecode::Native;
  if (mode == "slab") return Q8WideDecode::Slab;
  if (mode == "dual") return Q8WideDecode::Dual;
  return Q8WideDecode::Auto;
}
// Active-row values the shared-weight dual kernels are instantiated for.
[[nodiscard]] inline bool q8WideDecodeDualRows(uint32_t activeRows) noexcept {
  return activeRows == 40 || activeRows == 48 || activeRows == 56 ||
         activeRows == 64;
}

// Owns Q8 pipeline selection and dispatch on Apple10. Decode grid sizing
// reuses the Q4 policies; the tile set is the Q8 kernel family.
class Q8Linear final {
public:
  explicit Q8Linear(const DeviceCapabilities &device) noexcept;

  // Per workload: two activation modes times the tile set. Prefill tries
  // N128/N256 on eight simdgroups and the four-simdgroup N128 tile; decode
  // tries the lane forms plus the paired one-lane tiles. The widest decode
  // case is baseline + N128{4 groups x 2 scopes} + N256{4} + paired{2} = 13
  // per activation, doubled to 26.
  static constexpr std::size_t kMaximumCandidates = 26;

  [[nodiscard]] Q8Plan plan(LinearWorkload workload) const;
  [[nodiscard]] static Q8Plan plan(LinearWorkload workload, Q8Config config);
  [[nodiscard]] std::vector<Q8Plan> candidates(LinearWorkload workload) const;
  // The dense Q8 model path prepares one prefill input side channel per
  // buffer and none for decode, so the prefill policy is a single activation
  // mode and decode is A16. Installed choices are held to both.
  [[nodiscard]] bool prefillActivation8() const noexcept { return prefillA8_; }
  // Installed only at startup; encoding does a read-only lookup, never tuning.
  void setChoices(std::span<const Q8Choice> choices);
  void add(metal::CommandGraph &graph, Q8Buffers buffers,
           const Q8DenseProjection &projection, const Q8Plan &plan,
           const Q8DenseProjection *gate = nullptr,
           Q4DispatchStats *stats = nullptr, uint32_t activeRows = 0) const;

  // Input preparation. A16 reuses the Q4 fp32 sums pass; A8 quantizes into
  // scratch. The fused norms write both the bf16 output and the side channel.
  void addPrefillSums(metal::CommandGraph &graph, metal::MetalBuffer input,
                      metal::MetalBuffer sums, LinearMatrix matrix,
                      uint32_t rows) const;
  void addPrefillQuantize(metal::CommandGraph &graph, metal::MetalBuffer input,
                          Q8Scratch scratch, LinearMatrix matrix,
                          uint32_t rows) const;
  void addDecodeQuantize(metal::CommandGraph &graph, metal::MetalBuffer input,
                         Q8Scratch scratch, LinearMatrix matrix,
                         uint32_t rows) const;
  void addPrefillNormQuantize(metal::CommandGraph &graph,
                              metal::MetalBuffer input,
                              metal::MetalBuffer weight,
                              metal::MetalBuffer output, Q8Scratch scratch,
                              LinearMatrix matrix, uint32_t rows) const;
  void addDecodeNormQuantize(metal::CommandGraph &graph,
                             metal::MetalBuffer input,
                             metal::MetalBuffer weight,
                             metal::MetalBuffer output, Q8Scratch scratch,
                             LinearMatrix matrix, uint32_t rows) const;

  void addPrefill(metal::CommandGraph &graph, metal::MetalBuffer input,
                  const Q8DenseProjection &projection,
                  metal::MetalBuffer output, metal::MetalBuffer sums,
                  Q8Scratch scratch, LinearMatrix matrix, uint32_t rows) const;
  void addPrefillUpWithGate(metal::CommandGraph &graph,
                            metal::MetalBuffer input,
                            const Q8DenseProjection &up,
                            metal::MetalBuffer gateScratch,
                            metal::MetalBuffer output, metal::MetalBuffer sums,
                            metal::MetalBuffer downSums, Q8Scratch scratch,
                            Q8Scratch downScratch, LinearMatrix matrix,
                            uint32_t rows) const;
  void addPrefillResidual(metal::CommandGraph &graph,
                          metal::MetalBuffer input,
                          const Q8DenseProjection &projection,
                          metal::MetalBuffer residual,
                          metal::MetalBuffer output, metal::MetalBuffer sums,
                          Q8Scratch scratch, LinearMatrix matrix,
                          uint32_t rows) const;

  void addDecode(metal::CommandGraph &graph, metal::MetalBuffer input,
                 const Q8DenseProjection &projection,
                 metal::MetalBuffer output, LinearMatrix matrix,
                 Q8Scratch scratch = {}) const;
  void addDecodeBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
                      const Q8DenseProjection &projection,
                      metal::MetalBuffer output, LinearMatrix matrix,
                      uint32_t lanes, Q4DispatchStats &stats,
                      Q8Scratch scratch = {}, bool inputPrepared = false) const;
  void addGateUpBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
                      const Q8DenseProjection &gate,
                      const Q8DenseProjection &up,
                      metal::MetalBuffer gateScratch,
                      metal::MetalBuffer output, LinearMatrix matrix,
                      uint32_t lanes, Q4DispatchStats &stats,
                      Q8Scratch scratch = {}, bool inputPrepared = false) const;
  void addResidualBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
                        const Q8DenseProjection &projection,
                        metal::MetalBuffer residual,
                        metal::MetalBuffer output, LinearMatrix matrix,
                        uint32_t lanes, Q4DispatchStats &stats,
                        Q8Scratch scratch = {}, bool inputPrepared = false) const;
  [[nodiscard]] Q8ScratchSize decodeScratchSize(LinearWorkload workload) const;

private:
  [[nodiscard]] Q8Config baseline(LinearWorkload workload) const;
  uint32_t gpuCores_ = 0;
  bool prefillA8_ = false;
  bool m32Sg16Profile_ = false;
  std::vector<Q8Choice> choices_;
};

} // namespace splash::ops
