#include "ops/LinearQ8.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Linear.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {
namespace {

constexpr uint32_t kPrefillRows = 32;
constexpr uint32_t kQuantGroup = 64;
static_assert(SPLASH_TARGET_VERIFY_ROWS == 8,
              "Q8 decode lanes carry eight verify rows each");
static_assert(sizeof(LinearMatrix) == 8);

bool pairedTile(Q8Tile tile) noexcept {
  return tile == Q8Tile::Paired128 || tile == Q8Tile::Paired256 ||
         tile == Q8Tile::Paired64;
}

// The four-simdgroup kernels: every prefill N128 tile, the decode M24 N128
// plain and residual projections, and the paired N256 tile.
bool supportsFourSimdgroups(LinearWorkload w, Q8Tile tile) noexcept {
  if (tile == Q8Tile::Paired256)
    return w.phase == LinearPhase::Decode &&
        w.rows == SPLASH_TARGET_VERIFY_ROWS &&
        w.epilogue == LinearEpilogue::None;
  if (tile != Q8Tile::N128) return false;
  return w.phase == LinearPhase::Prefill ||
      (w.rows == 24 && (w.epilogue == LinearEpilogue::None ||
                        w.epilogue == LinearEpilogue::Residual));
}

void validate(LinearWorkload w) {
  if (!w.matrix.outputSize || w.matrix.outputSize % 256 ||
      !w.matrix.inputSize || w.matrix.inputSize % kQuantGroup)
    throw std::invalid_argument("invalid Q8 linear matrix");
  if (w.phase == LinearPhase::Prefill) {
    if (!w.rows || w.rows > SPLASH_PREFILL_TOKEN_BUDGET ||
        w.epilogue == LinearEpilogue::GateUp)
      throw std::invalid_argument("invalid Q8 prefill workload");
  } else if (w.phase == LinearPhase::Decode) {
    if (w.matrix.inputSize % 256 || !w.rows ||
        w.rows % SPLASH_TARGET_VERIFY_ROWS ||
        w.rows > SPLASH_TARGET_VERIFY_ROWS * SPLASH_MAXIMUM_BATCH_WIDTH ||
        w.epilogue == LinearEpilogue::UpWithGate)
      throw std::invalid_argument("invalid Q8 decode workload");
  } else {
    throw std::invalid_argument("invalid Q8 linear phase");
  }
  if (w.epilogue != LinearEpilogue::None &&
      w.epilogue != LinearEpilogue::Residual &&
      w.epilogue != LinearEpilogue::GateUp &&
      w.epilogue != LinearEpilogue::UpWithGate)
    throw std::invalid_argument("invalid Q8 linear epilogue");
}

void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes) {
  if (bytes && (!buffer || buffer.sizeBytes() < bytes))
    throw std::invalid_argument(
        "Q8 buffer is below plan requirement (" +
        std::to_string(buffer ? buffer.sizeBytes() : 0) + " < " +
        std::to_string(bytes) + ")");
}

void requireProjection(const Q8DenseProjection &p, LinearMatrix matrix) {
  if (p.outputSize != matrix.outputSize || p.inputSize != matrix.inputSize)
    throw std::invalid_argument("Q8 projection does not match plan");
  if (p.parameters != Q8Parameters::Float32Folded &&
      p.parameters != Q8Parameters::BFloat16)
    throw std::invalid_argument("invalid Q8 parameter format");
  requireBytes(p.weights, uint64_t{matrix.outputSize} * matrix.inputSize);
  const uint64_t bytes = uint64_t{matrix.outputSize} *
      (matrix.inputSize / kQuantGroup) *
      (p.parameters == Q8Parameters::BFloat16 ? 2 : 4);
  requireBytes(p.scales, bytes);
  requireBytes(p.biases, bytes);
}

LinearWorkload decode(LinearMatrix matrix, uint32_t lanes,
                      LinearEpilogue epilogue) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid Q8 decode batch width");
  return {matrix, decodeRowsForLanes(lanes), LinearPhase::Decode, epilogue};
}

} // namespace

uint32_t Q8Plan::storageRows() const noexcept {
  return workload_.phase == LinearPhase::Prefill
      ? ((workload_.rows + kPrefillRows - 1) / kPrefillRows) * kPrefillRows
      : workload_.rows;
}
uint32_t Q8Plan::tileColumns() const noexcept {
  switch (config_.tile) {
  case Q8Tile::N256:
  case Q8Tile::Paired256: return 256;
  case Q8Tile::N128:
  case Q8Tile::Paired128: return 128;
  case Q8Tile::N64:
  case Q8Tile::Paired64: return 64;
  }
  return 0;
}
uint32_t Q8Plan::threadsPerThreadgroup() const noexcept {
  return static_cast<uint32_t>(config_.simdgroups) * 32;
}
uint64_t Q8Plan::sumsBytes() const noexcept {
  return config_.activation == Q8Activation::A16 &&
          workload_.phase == LinearPhase::Prefill
      ? uint64_t{storageRows()} * (workload_.matrix.inputSize / kQuantGroup) * 4
      : 0;
}
Q8ScratchSize Q8Plan::scratchSize() const noexcept {
  if (config_.activation != Q8Activation::A8) return {};
  return {uint64_t{storageRows()} * workload_.matrix.inputSize,
          uint64_t{storageRows()} * (workload_.matrix.inputSize / kQuantGroup) *
              sizeof(float) * 2};
}
uint64_t Q8Plan::gateScratchBytes() const noexcept {
  const bool needed = workload_.epilogue == LinearEpilogue::UpWithGate ||
      (workload_.epilogue == LinearEpilogue::GateUp && !secondPipeline_.empty());
  return needed ? uint64_t{storageRows()} * workload_.matrix.outputSize * 2 : 0;
}
uint64_t Q8Plan::downSumsBytes() const noexcept {
  return workload_.epilogue == LinearEpilogue::UpWithGate &&
          config_.activation == Q8Activation::A16
      ? uint64_t{storageRows()} * (workload_.matrix.outputSize / kQuantGroup) * 4
      : 0;
}
Q8ScratchSize Q8Plan::downScratchSize() const noexcept {
  if (workload_.epilogue != LinearEpilogue::UpWithGate ||
      config_.activation != Q8Activation::A8)
    return {};
  return {uint64_t{storageRows()} * workload_.matrix.outputSize,
          uint64_t{storageRows()} * (workload_.matrix.outputSize / kQuantGroup) *
              sizeof(float) * 2};
}

Q8Plan::Q8Plan(LinearWorkload w, Q8Config config) : workload_(w), config_(config) {
  validate(w);
  if (config.activation != Q8Activation::A16 &&
      config.activation != Q8Activation::A8)
    throw std::invalid_argument("invalid Q8 activation mode");
  if (config.simdgroups != LinearSimdgroups::Four &&
      config.simdgroups != LinearSimdgroups::Eight)
    throw std::invalid_argument("invalid Q8 cooperative scope");
  const bool four = config.simdgroups == LinearSimdgroups::Four;
  if (four && !supportsFourSimdgroups(w, config.tile))
    throw std::invalid_argument("Q8 tile requires its kernel's simdgroup count");
  if (w.matrix.outputSize % tileColumns())
    throw std::invalid_argument("Q8 matrix is not divisible by tile columns");
  const bool residual = w.epilogue == LinearEpilogue::Residual;
  const bool a8 = config.activation == Q8Activation::A8;
  const uint32_t lane =
      w.phase == LinearPhase::Decode ? w.rows / SPLASH_TARGET_VERIFY_ROWS : 0;
  if (w.phase == LinearPhase::Decode && lane > 4 && lane != 8)
    throw std::invalid_argument("Q8 decode rows must be 8-32 by eight or 64");
  if (lane == 8 && a8)
    throw std::invalid_argument("A8 decode has no 64-row (B8) kernels");
  // Kernel name tables indexed by lane - 1; the A8 forms carry the q8a8
  // infix. Prefill N256 sg4 tiles and decode N256 residuals are not
  // instantiated, matching the Q4 family.
  constexpr std::array<std::string_view, 8> a16N128{
      "decode_linear_q8_n128", "decode_linear_q8_n128_m16",
      "decode_linear_q8_n128_m24", "decode_linear_q8_n128_m32", "", "", "",
      "decode_linear_q8_n128_m64"};
  constexpr std::array<std::string_view, 8> a8N128{
      "decode_linear_q8a8_n128", "decode_linear_q8a8_n128_m16",
      "decode_linear_q8a8_n128_m24", "decode_linear_q8a8_n128_m32", "", "", "",
      ""};
  constexpr std::array<std::string_view, 8> a16N256{
      "decode_linear_q8_n256", "decode_linear_q8_n256_m16",
      "decode_linear_q8_n256_m24", "decode_linear_q8_n256_m32", "", "", "",
      "decode_linear_q8_n256_m64"};
  constexpr std::array<std::string_view, 8> a8N256{
      "decode_linear_q8a8_n256", "decode_linear_q8a8_n256_m16",
      "decode_linear_q8a8_n256_m24", "decode_linear_q8a8_n256_m32", "", "", "",
      ""};
  constexpr std::array<std::string_view, 8> a16Residual{
      "decode_linear_q8_n128_residual", "decode_linear_q8_n128_residual_m16",
      "decode_linear_q8_n128_residual_m24",
      "decode_linear_q8_n128_residual_m32", "", "", "",
      "decode_linear_q8_n128_residual_m64"};
  constexpr std::array<std::string_view, 8> a8Residual{
      "decode_linear_q8a8_n128_residual",
      "decode_linear_q8a8_n128_residual_m16",
      "decode_linear_q8a8_n128_residual_m24",
      "decode_linear_q8a8_n128_residual_m32", "", "", "", ""};
  constexpr std::array<std::string_view, 8> a16UpSilu{
      "", "", "decode_linear_q8_n256_up_silu_m24",
      "decode_linear_q8_n256_up_silu_m32", "", "", "",
      "decode_linear_q8_n256_up_silu_m64"};
  constexpr std::array<std::string_view, 8> a8UpSilu{
      "", "", "decode_linear_q8a8_n256_up_silu_m24",
      "decode_linear_q8a8_n256_up_silu_m32", "", "", "", ""};

  if (w.phase == LinearPhase::Prefill) {
    if (config.groups || pairedTile(config.tile) || config.tile == Q8Tile::N64)
      throw std::invalid_argument("invalid Q8 prefill configuration");
    if (w.epilogue == LinearEpilogue::UpWithGate) {
      // Only the N256 eight-simdgroup and N128 four-simdgroup fused kernels
      // are instantiated, matching the Q4 forms.
      if (config.tile == Q8Tile::N256 && !four)
        pipeline_ = a8 ? "prefill_linear_q8a8_n256_up_silu_quantize"
                       : "prefill_linear_q8_n256_up_silu_sums";
      else if (config.tile == Q8Tile::N128 && four)
        pipeline_ = a8 ? "prefill_linear_q8a8_n128_up_silu_quantize_sg4"
                       : "prefill_linear_q8_n128_up_silu_sums_sg4";
      else
        throw std::invalid_argument(
            "Q8 fused prefill up requires N256 or N128 on four simdgroups");
    } else if (residual) {
      if (config.tile != Q8Tile::N128 && config.tile != Q8Tile::N256)
        throw std::invalid_argument("invalid Q8 residual tile");
      if (four && config.tile != Q8Tile::N128)
        throw std::invalid_argument("Q8 residual sg4 requires N128");
      if (four)
        pipeline_ = a8 ? "prefill_linear_q8a8_n128_residual_sg4"
                       : "prefill_linear_q8_n128_residual_sg4";
      else if (config.tile == Q8Tile::N128)
        pipeline_ = a8 ? "prefill_linear_q8a8_n128_residual"
                       : "prefill_linear_q8_n128_residual";
      else
        pipeline_ = a8 ? "prefill_linear_q8a8_n256_residual"
                       : "prefill_linear_q8_n256_residual";
    } else {
      if (four && config.tile != Q8Tile::N128)
        throw std::invalid_argument("Q8 sg4 prefill requires N128");
      if (config.tile != Q8Tile::N128 && config.tile != Q8Tile::N256)
        throw std::invalid_argument("invalid Q8 prefill tile");
      if (four)
        pipeline_ = a8 ? "prefill_linear_q8a8_n128_sg4"
                       : "prefill_linear_q8_n128_sg4";
      else if (config.tile == Q8Tile::N128)
        pipeline_ = a8 ? "prefill_linear_q8a8_n128" : "prefill_linear_q8_n128";
      else
        pipeline_ = a8 ? "prefill_linear_q8a8_n256" : "prefill_linear_q8_n256";
    }
    if (a8 && four && !residual &&
        w.epilogue == LinearEpilogue::None && q8PrefillStageParameters())
      pipeline_ = "prefill_linear_q8a8_n128_stagep";
    return;
  }

  // Decode: persistent threadgroups stride over tileColumns-wide tiles.
  if (!config.groups || config.groups > w.matrix.outputSize / tileColumns())
    throw std::invalid_argument("invalid Q8 decode group count");
  // The pipelined batched residuals: Paired128 at 16 rows, Paired64 at 32.
  const bool pairedBatchedResidual =
      residual && !a8 && !four &&
      ((config.tile == Q8Tile::Paired128 && lane == 2) ||
       (config.tile == Q8Tile::Paired64 && lane == 4));
  if (pairedTile(config.tile) && !pairedBatchedResidual &&
      (lane != 1 || config.tile == Q8Tile::Paired64 ||
       w.matrix.outputSize % 256))
    throw std::invalid_argument(
        "paired Q8 tile requires one lane and paired columns");
  if (config.tile == Q8Tile::Paired256 && !four)
    throw std::invalid_argument("paired256 requires four simdgroups");
  if (w.epilogue == LinearEpilogue::GateUp && config.tile != Q8Tile::N256)
    throw std::invalid_argument("Q8 gate/up requires N256");
  if (config.tile == Q8Tile::N64 &&
      (!residual || a8 || four || (lane != 3 && lane != 4 && lane != 8)))
    throw std::invalid_argument(
        "Q8 N64 is the A16 decode residual at 24, 32 or 64 rows only");
  if (residual && config.tile != Q8Tile::N128 &&
      config.tile != Q8Tile::Paired128 && config.tile != Q8Tile::N64 &&
      config.tile != Q8Tile::Paired64)
    throw std::invalid_argument("Q8 decode residual requires N128 or N64");

  if (w.epilogue == LinearEpilogue::GateUp) {
    if (lane == 1)
      pipeline_ = a8 ? "decode_linear_q8a8_n256_gate_up"
                     : "decode_linear_q8_n256_gate_up";
    else if (lane == 2)
      pipeline_ = a8 ? "decode_linear_q8a8_n256_gate_up_m16"
                     : "decode_linear_q8_n256_gate_up_m16";
    else {
      // The m24/m32 forms dispatch the gate projection first, then the up
      // projection fused with silu(gate) read back from the scratch.
      pipeline_ = (a8 ? a8N256 : a16N256)[lane - 1];
      secondPipeline_ = (a8 ? a8UpSilu : a16UpSilu)[lane - 1];
    }
  } else if (residual) {
    if (config.tile == Q8Tile::N64)
      pipeline_ = lane == 3 ? "decode_linear_q8_n64_residual_m24"
                  : lane == 4 ? "decode_linear_q8_n64_residual_m32"
                              : "decode_linear_q8_n64_residual_m64";
    else if (config.tile == Q8Tile::Paired64)
      pipeline_ = "decode_linear_q8_n64_residual_m32_paired";
    else if (config.tile == Q8Tile::Paired128 && lane == 2)
      pipeline_ = "decode_linear_q8_n128_residual_m16_paired";
    else if (config.tile == Q8Tile::Paired128)
      pipeline_ = a8 ? "decode_linear_q8a8_n128_residual_paired"
                     : "decode_linear_q8_n128_residual_paired";
    else if (four)
      pipeline_ = a8 ? "decode_linear_q8a8_n128_residual_m24_sg4"
                     : "decode_linear_q8_n128_residual_m24_sg4";
    else
      pipeline_ = (a8 ? a8Residual : a16Residual)[lane - 1];
  } else if (config.tile == Q8Tile::Paired128) {
    pipeline_ = a8 ? "decode_linear_q8a8_n128_paired"
                   : "decode_linear_q8_n128_paired";
  } else if (config.tile == Q8Tile::Paired256) {
    pipeline_ = a8 ? "decode_linear_q8a8_n256_paired_sg4"
                   : "decode_linear_q8_n256_paired_sg4";
  } else {
    if (four) {
      if (lane != 3 || config.tile != Q8Tile::N128)
        throw std::invalid_argument(
            "Q8 sg4 decode requires the m24 N128 tile");
      pipeline_ = a8 ? "decode_linear_q8a8_n128_m24_sg4"
                     : "decode_linear_q8_n128_m24_sg4";
    } else if (config.tile == Q8Tile::N256) {
      pipeline_ = (a8 ? a8N256 : a16N256)[lane - 1];
    } else {
      pipeline_ = (a8 ? a8N128 : a16N128)[lane - 1];
    }
  }
}

namespace {

// Decode groups stream output tiles. Same policies as the Q4 path, measured
// on 16/20-core Apple10 GPUs; Q8 retunes them in the tuning phase.
struct DecodeGroupPolicy final {
  uint32_t fullGridGroupsPerCore;
  uint32_t waveGroupsPerCore;
  uint32_t manyWaveTilesPerCore;
};
constexpr DecodeGroupPolicy kN128Groups{4, 4, 12}, kN128M16Groups{5, 4, 12},
    kGateUpGroups{3, 3, 8}, kFourSimdgroupGroups{8, 8, 24};

uint32_t maxCoreTiles(uint32_t tiles, uint32_t groups, uint32_t cores) noexcept {
  uint32_t worst = 0;
  for (uint32_t core = 0; core < cores; ++core) {
    uint32_t load = 0;
    for (uint32_t group = core; group < groups; group += cores)
      load += (tiles - group + groups - 1) / groups;
    worst = std::max(worst, load);
  }
  return worst;
}

uint32_t decodeGroups(uint32_t tiles, uint32_t cores,
                      DecodeGroupPolicy policy) noexcept {
  const uint32_t wave = policy.waveGroupsPerCore * cores;
  if (tiles <= policy.fullGridGroupsPerCore * cores ||
      tiles >= policy.manyWaveTilesPerCore * cores)
    return tiles;
  const uint32_t twoTile = (tiles + 1) / 2;
  if (twoTile > wave) return wave;
  const uint32_t balanced = (tiles + cores - 1) / cores;
  uint32_t groups =
      std::max(twoTile, policy.fullGridGroupsPerCore * cores * 3 / 4);
  while (maxCoreTiles(tiles, groups, cores) != balanced) ++groups;
  return groups;
}

constexpr uint32_t kPaired256TilesPerCore = 8;
constexpr uint32_t kPaired256WaveGroupsPerCore = 4;
constexpr uint32_t kAssumedGpuCores = 32;

// The dense Q8 model path prepares one prefill input side channel per buffer
// (fused RMS norm plus one explicit quantize pass) and no decode side channel,
// so prefill runs one activation mode and decode stays A16. Measured on the M5
// Max at the production N128/sg4 tiles: A8 is 1.12x faster on prefill (608.6ms
// -> 542.0ms for the packed 512-row prefill, measured before the epilogue fix
// below). Its lane divergence was the A8 epilogue rounding identical rows
// differently at different tile rows (fixed by q8a8_accumulate in
// q8_mpp_tiles.h); it stays off until the quality gate is re-run on the fixed
// kernels (the earlier 21/22 golden result predates the fix). Decode
// keeps A16 whatever this says: its A8 forms measured no gain over the
// bandwidth-bound A16 kernels and would need extra preparation passes.
// SPLASH_Q8_PREFILL_A8 (make Q8_PREFILL_ACTIVATION=a8) builds the A8
// experiment; the default build is A16.
#if defined(SPLASH_Q8_PREFILL_A8) && SPLASH_Q8_PREFILL_A8
constexpr Q8Activation kPrefillActivation = Q8Activation::A8;
#else
constexpr Q8Activation kPrefillActivation = Q8Activation::A16;
#endif
constexpr Q8Activation kDecodeActivation = Q8Activation::A16;

} // namespace

Q8Linear::Q8Linear(const DeviceCapabilities &device) noexcept
    : gpuCores_(device.gpuCoreCount ? device.gpuCoreCount : kAssumedGpuCores),
      prefillA8_(kPrefillActivation == Q8Activation::A8),
      m32Sg16Profile_(q8Split2M32Sg16Profile(device)) {}

// Apple10 baseline mirrors the measured Q4 policy; the activation is the one
// mode the model prepares input for.
Q8Config Q8Linear::baseline(LinearWorkload w) const {
  validate(w);
  const uint32_t tiles128 = w.matrix.outputSize / 128;
  const uint32_t tiles256 = w.matrix.outputSize / 256;
  if (w.phase == LinearPhase::Prefill)
    return {Q8Tile::N128, 0, LinearSimdgroups::Four, kPrefillActivation};
  const uint32_t lanes = w.rows / SPLASH_TARGET_VERIFY_ROWS;
  const auto groups = [&](uint32_t tiles, DecodeGroupPolicy policy) {
    return decodeGroups(tiles, gpuCores_, policy);
  };
  if (w.epilogue == LinearEpilogue::GateUp)
    return {Q8Tile::N256, groups(tiles256, kGateUpGroups)};
  if (lanes == 1) {
    if (w.epilogue == LinearEpilogue::None &&
        tiles256 >= kPaired256TilesPerCore * gpuCores_)
      return {Q8Tile::Paired256,
              std::min(tiles256, kPaired256WaveGroupsPerCore * gpuCores_),
              LinearSimdgroups::Four};
    return {Q8Tile::Paired128, groups(tiles128, kN128Groups)};
  }
  // Q8 weights double the Q4 bytes per tile, so wide plain projections keep
  // every N256 tile resident at once instead of looping persistent groups:
  // measured on the 40-core M5 Max (tune-kernels, dev/results/q8-tuning) at
  // +8 to +30% on the GDN/attention input projections for M16-M32 and
  // neutral to +11% on the lm head, B2-B4 whole-graph decode +3.5 to +5%.
  // Every A16 decode candidate is bit-identical to the one-lane bytes, so
  // this changes speed only.
  if (w.epilogue == LinearEpilogue::None && tiles256 >= gpuCores_)
    return {Q8Tile::N256, tiles256};
  // Residuals whose N128 grid leaves at most one threadgroup per core (the
  // model's mixer output and FFN down, N = 5120, on 40 cores) run N64 tiles
  // at M24 and M32: twice the threadgroups, each column's group chain
  // unchanged, so the bytes equal the N128 kernels'. Measured alone on the
  // 40-core M5 Max with rotating weights (dev/results/q8-decode-residual):
  // M24 -26% (mixer output) and -28% (FFN down), M32 -15% and -14%; at M8 and
  // M16 N128 stays faster.
  //
  // The same residuals pipeline two quant groups (the one-lane paired
  // cadence, group order unchanged, same bytes) where that measured faster
  // alone (dev/results/q8-decode-residual/pipelined): M16 on the N128 grid
  // -11% (mixer output) and -10% (FFN down), M32 on the N64 grid -4% and -8%;
  // at M24 pipelining gained nothing.
  if (w.epilogue == LinearEpilogue::Residual && lanes == 2 &&
      tiles128 <= gpuCores_)
    return {Q8Tile::Paired128, tiles128};
  if (w.epilogue == LinearEpilogue::Residual && lanes == 4 &&
      tiles128 <= gpuCores_)
    return {Q8Tile::Paired64, w.matrix.outputSize / 64};
  if (w.epilogue == LinearEpilogue::Residual && lanes >= 3 &&
      tiles128 <= gpuCores_)
    return {Q8Tile::N64, w.matrix.outputSize / 64};
  // One resident N128 tile per group on four simdgroups: the model's mixer
  // output and FFN down residuals at M24, whole-graph B3 decode +3.8% over
  // eight simdgroups on the same grid (tune-kernels, dev/results/q8-tuning).
  if (lanes == 3 && tiles128 <= gpuCores_ && w.matrix.inputSize >= 4096)
    return {Q8Tile::N128, tiles128, LinearSimdgroups::Four};
  if (lanes == 3)
    return {Q8Tile::N128, groups(tiles128, kFourSimdgroupGroups),
            LinearSimdgroups::Four};
  return {Q8Tile::N128,
          groups(tiles128, lanes == 2 ? kN128M16Groups : kN128Groups)};
}

Q8Plan Q8Linear::plan(LinearWorkload workload) const {
  const auto found = std::lower_bound(
      choices_.begin(), choices_.end(), workload,
      [](const Q8Choice &choice, LinearWorkload key) {
        return choice.workload < key;
      });
  return Q8Plan(workload, found != choices_.end() && found->workload == workload
                              ? found->configuration
                              : baseline(workload));
}
Q8Plan Q8Linear::plan(LinearWorkload workload, Q8Config config) {
  return Q8Plan(workload, config);
}
void Q8Linear::setChoices(std::span<const Q8Choice> choices) {
  std::vector<Q8Choice> pending(choices.begin(), choices.end());
  for (const auto &choice : pending)
    (void)plan(choice.workload, choice.configuration);
  // Reject a policy the model cannot prepare input for instead of encoding a
  // kernel whose side channel was never written.
  for (const auto &choice : pending) {
    const bool prefill = choice.workload.phase == LinearPhase::Prefill;
    if (choice.configuration.activation != (prefill ? kPrefillActivation
                                                    : kDecodeActivation))
      throw std::invalid_argument(
          prefill ? "Q8 prefill choices must match the model's input side channel"
                  : "Q8 decode choices must use A16");
  }
  std::sort(pending.begin(), pending.end(), [](const auto &a, const auto &b) {
    return a.workload < b.workload;
  });
  for (size_t i = 1; i < pending.size(); ++i)
    if (pending[i - 1].workload == pending[i].workload)
      throw std::invalid_argument("duplicate Q8 linear choice");
  choices_ = std::move(pending);
}

std::vector<Q8Plan> Q8Linear::candidates(LinearWorkload w) const {
  std::vector<Q8Plan> result;
  result.reserve(kMaximumCandidates);
  for (const Q8Activation activation : {Q8Activation::A16, Q8Activation::A8}) {
    // Decode is A16 in every build; the 64-row (B8) tile has no A8 kernels.
    if (activation == Q8Activation::A8 && w.phase == LinearPhase::Decode &&
        w.rows > 4 * SPLASH_TARGET_VERIFY_ROWS)
      continue;
    result.push_back(Q8Plan(w, [&] {
      Q8Config config = baseline(w);
      config.activation = activation;
      // N64 and the pipelined batched residuals have no A8 kernel; A8 decode
      // candidates keep the N128 grid.
      if (activation == Q8Activation::A8 && w.phase == LinearPhase::Decode &&
          (config.tile == Q8Tile::N64 || config.tile == Q8Tile::Paired64 ||
           (pairedTile(config.tile) && w.rows != SPLASH_TARGET_VERIFY_ROWS)))
        config = {Q8Tile::N128, w.matrix.outputSize / 128,
                  LinearSimdgroups::Eight, activation};
      return config;
    }()));
    const auto append = [&](Q8Config config) {
      for (const auto &existing : result)
        if (existing.configuration() == config) return;
      result.push_back(Q8Plan(w, config));
    };
    for (const Q8Tile tile : {Q8Tile::N128, Q8Tile::N256}) {
      const uint32_t columns = tile == Q8Tile::N256 ? 256 : 128;
      if (w.matrix.outputSize % columns ||
          (w.epilogue == LinearEpilogue::GateUp && tile != Q8Tile::N256) ||
          (w.phase == LinearPhase::Decode &&
           w.epilogue == LinearEpilogue::Residual && tile != Q8Tile::N128))
        continue;
      if (w.phase == LinearPhase::Prefill) {
        if (tile == Q8Tile::N256 || w.epilogue != LinearEpilogue::UpWithGate)
          append({tile, 0, LinearSimdgroups::Eight, activation});
        if (supportsFourSimdgroups(w, tile))
          append({tile, 0, LinearSimdgroups::Four, activation});
      } else {
        const uint32_t tiles = w.matrix.outputSize / columns;
        for (const uint32_t groups :
             {2 * gpuCores_, 3 * gpuCores_, 4 * gpuCores_, tiles}) {
          append({tile, std::min(groups, tiles), LinearSimdgroups::Eight,
                  activation});
          if (supportsFourSimdgroups(w, tile))
            append({tile, std::min(groups, tiles), LinearSimdgroups::Four,
                    activation});
        }
      }
    }
    // A16 decode residuals at 24 and 32 rows also run the N64 tile, at its
    // full grid and at two groups per core.
    if (activation == Q8Activation::A16 && w.phase == LinearPhase::Decode &&
        w.epilogue == LinearEpilogue::Residual &&
        (w.rows == 3 * SPLASH_TARGET_VERIFY_ROWS ||
         w.rows == 4 * SPLASH_TARGET_VERIFY_ROWS) &&
        w.matrix.outputSize % 64 == 0) {
      const uint32_t tiles = w.matrix.outputSize / 64;
      append({Q8Tile::N64, tiles, LinearSimdgroups::Eight, activation});
      append({Q8Tile::N64, std::min(tiles, 2 * gpuCores_),
              LinearSimdgroups::Eight, activation});
    }
    // The pipelined batched residuals: N128 at 16 rows, N64 at 32 rows, each
    // at its full grid and at one group per core.
    if (activation == Q8Activation::A16 && w.phase == LinearPhase::Decode &&
        w.epilogue == LinearEpilogue::Residual &&
        (w.rows == 2 * SPLASH_TARGET_VERIFY_ROWS ||
         w.rows == 4 * SPLASH_TARGET_VERIFY_ROWS)) {
      const bool m16 = w.rows == 2 * SPLASH_TARGET_VERIFY_ROWS;
      const uint32_t columns = m16 ? 128 : 64;
      if (w.matrix.outputSize % columns == 0) {
        const uint32_t tiles = w.matrix.outputSize / columns;
        const Q8Tile tile = m16 ? Q8Tile::Paired128 : Q8Tile::Paired64;
        append({tile, tiles, LinearSimdgroups::Eight, activation});
        append({tile, std::min(tiles, gpuCores_), LinearSimdgroups::Eight,
                activation});
      }
    }
    // One-lane paired tiles at one resident wave and at the full grid. The
    // paired residual kernel exists on N128 only; gate/up has no paired form.
    if (w.phase == LinearPhase::Decode &&
        w.rows == SPLASH_TARGET_VERIFY_ROWS &&
        w.epilogue != LinearEpilogue::GateUp) {
      const uint32_t n = w.matrix.outputSize;
      append({Q8Tile::Paired128, n / 128, LinearSimdgroups::Eight, activation});
      append({Q8Tile::Paired128,
              std::min(n / 128, kPaired256WaveGroupsPerCore * gpuCores_),
              LinearSimdgroups::Eight, activation});
      if (w.epilogue == LinearEpilogue::None)
        for (const uint32_t groups :
             {kPaired256WaveGroupsPerCore * gpuCores_, n / 256})
          append({Q8Tile::Paired256, std::min(groups, n / 256),
                  LinearSimdgroups::Four, activation});
    }
  }
  return result;
}

Q8ScratchSize Q8Linear::decodeScratchSize(LinearWorkload w) const {
  auto size = Q8Plan(w, baseline(w)).scratchSize();
  const auto selected = plan(w).scratchSize();
  size.input = std::max(size.input, selected.input);
  size.terms = std::max(size.terms, selected.terms);
  return size;
}

void Q8Linear::addPrefillSums(metal::CommandGraph &graph,
                              metal::MetalBuffer input,
                              metal::MetalBuffer sums, LinearMatrix matrix,
                              uint32_t rows) const {
  validate({matrix, rows, LinearPhase::Prefill, LinearEpilogue::None});
  const uint32_t tiles = (rows + kPrefillRows - 1) / kPrefillRows;
  const uint64_t storageRows = uint64_t{tiles} * kPrefillRows;
  requireBytes(input, storageRows * matrix.inputSize * 2);
  requireBytes(sums, storageRows * (matrix.inputSize / kQuantGroup) * 4);
  graph.add("prefill_linear_q4_sums32", {input, sums},
            Q4PrefillParams{matrix.outputSize, matrix.inputSize}, {tiles, 1, 1});
}

void Q8Linear::addPrefillQuantize(metal::CommandGraph &graph,
                                  metal::MetalBuffer input, Q8Scratch scratch,
                                  LinearMatrix matrix, uint32_t rows) const {
  validate({matrix, rows, LinearPhase::Prefill, LinearEpilogue::None});
  const uint32_t tiles = (rows + kPrefillRows - 1) / kPrefillRows;
  const uint64_t storageRows = uint64_t{tiles} * kPrefillRows;
  requireBytes(input, storageRows * matrix.inputSize * 2);
  requireBytes(scratch.input, storageRows * matrix.inputSize);
  requireBytes(scratch.terms,
               storageRows * (matrix.inputSize / kQuantGroup) * 8);
  graph.add("prefill_linear_q8a8_quantize",
            {input, scratch.input, scratch.terms},
            Q8QuantizeParams{matrix.inputSize, rows}, {tiles, 1, 1});
}

void Q8Linear::addDecodeQuantize(metal::CommandGraph &graph,
                                 metal::MetalBuffer input, Q8Scratch scratch,
                                 LinearMatrix matrix, uint32_t rows) const {
  validate({matrix, rows, LinearPhase::Decode, LinearEpilogue::None});
  requireBytes(input, uint64_t{rows} * matrix.inputSize * 2);
  requireBytes(scratch.input, uint64_t{rows} * matrix.inputSize);
  requireBytes(scratch.terms,
               uint64_t{rows} * (matrix.inputSize / kQuantGroup) * 8);
  graph.add("decode_linear_q8a8_quantize",
            {input, scratch.input, scratch.terms},
            Q8QuantizeParams{matrix.inputSize, rows}, {rows, 1, 1});
}

void Q8Linear::addPrefillNormQuantize(
    metal::CommandGraph &graph, metal::MetalBuffer input,
    metal::MetalBuffer weight, metal::MetalBuffer output, Q8Scratch scratch,
    LinearMatrix matrix, uint32_t rows) const {
  validate({matrix, rows, LinearPhase::Prefill, LinearEpilogue::None});
  const uint32_t tiles = (rows + kPrefillRows - 1) / kPrefillRows;
  const uint64_t storageRows = uint64_t{tiles} * kPrefillRows;
  requireBytes(input, storageRows * matrix.inputSize * 2);
  requireBytes(weight, uint64_t{matrix.inputSize} * 2);
  requireBytes(output, storageRows * matrix.inputSize * 2);
  requireBytes(scratch.input, storageRows * matrix.inputSize);
  requireBytes(scratch.terms,
               storageRows * (matrix.inputSize / kQuantGroup) * 8);
  graph.add("prefill_norm_rms_q8a8",
            {input, weight, output, scratch.input, scratch.terms},
            Q8QuantizeParams{matrix.inputSize, rows},
            {uint32_t(storageRows), 1, 1});
}

void Q8Linear::addDecodeNormQuantize(
    metal::CommandGraph &graph, metal::MetalBuffer input,
    metal::MetalBuffer weight, metal::MetalBuffer output, Q8Scratch scratch,
    LinearMatrix matrix, uint32_t rows) const {
  validate({matrix, rows, LinearPhase::Decode, LinearEpilogue::None});
  requireBytes(input, uint64_t{rows} * matrix.inputSize * 2);
  requireBytes(weight, uint64_t{matrix.inputSize} * 2);
  requireBytes(output, uint64_t{rows} * matrix.inputSize * 2);
  requireBytes(scratch.input, uint64_t{rows} * matrix.inputSize);
  requireBytes(scratch.terms,
               uint64_t{rows} * (matrix.inputSize / kQuantGroup) * 8);
  graph.add("norm_rms_q8a8_decode",
            {input, weight, output, scratch.input, scratch.terms},
            Q8QuantizeParams{matrix.inputSize, rows}, {rows, 1, 1});
}

void Q8Linear::add(metal::CommandGraph &graph, Q8Buffers b,
    const Q8DenseProjection &p, const Q8Plan &selected,
    const Q8DenseProjection *gate, Q4DispatchStats *stats,
    uint32_t activeRows) const {
  const LinearWorkload w = selected.workload();
  const auto [n, k] = w.matrix;
  const bool a8 = selected.activation() == Q8Activation::A8;
  requireProjection(p, w.matrix);
  // The A16 kernels and the A8 prefill kernels read compact parameters under
  // the _bf16p suffix. A8 decode has no compact form (production decode is
  // A16 in every build).
  const bool compact = p.parameters == Q8Parameters::BFloat16;
  if (compact && a8 && w.phase == LinearPhase::Decode)
    throw std::invalid_argument(
        "compact Q8 parameters require an A16 decode plan");
  requireBytes(b.output, uint64_t{selected.storageRows()} * n * 2);
  if (a8) {
    // The quantized input is either the prepared scratch or the caller's
    // buffers with the float2 terms.
    requireBytes(b.input, uint64_t{selected.storageRows()} * k);
    requireBytes(b.terms, selected.scratchSize().terms);
  } else {
    requireBytes(b.input, uint64_t{selected.storageRows()} * k * 2);
    requireBytes(b.sums, selected.sumsBytes());
  }
  requireBytes(b.gateScratch, selected.gateScratchBytes());
  requireBytes(b.downSums, selected.downSumsBytes());
  if (w.epilogue == LinearEpilogue::UpWithGate && a8) {
    const auto down = selected.downScratchSize();
    requireBytes(b.downInput, down.input);
    requireBytes(b.downTerms, down.terms);
  }
  if (w.epilogue == LinearEpilogue::Residual)
    requireBytes(b.residual, uint64_t{selected.storageRows()} * n * 2);
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (!gate) throw std::invalid_argument("Q8 gate projection is missing");
    requireProjection(*gate, w.matrix);
    if (gate->parameters != p.parameters)
      throw std::invalid_argument("Q8 gate and up parameter formats differ");
  } else if (gate)
    throw std::invalid_argument("unexpected Q8 gate projection");

  // 64-row decode plans dispatch the same persistent groups as two M32 row
  // slabs (group.z selects the half) instead of the padded M64 tile — the
  // measured wide-decode winner. The split2 residual branch below bypasses
  // this lambda and keeps its native M64 partials.
  const uint32_t active = activeRows ? activeRows : w.rows;
  const Q8WideDecode wide =
      w.phase == LinearPhase::Decode && w.rows == 64 && !a8
          ? q8WideDecodeMode()
          : Q8WideDecode::Native;
  const auto dispatch = [&](std::string_view pipeline,
                            std::initializer_list<metal::MetalBuffer> bindings) {
    std::string name(pipeline);
    // Shared-weight dual tile: n256 decode kernels ending in _m64 get a
    // per-width _d<rows> sibling; everything else keeps the two-slab suffix.
    bool dual = false;
    if (wide != Q8WideDecode::Native && w.epilogue != LinearEpilogue::Residual &&
        (name == "decode_linear_q8_n256_m64" ||
         name == "decode_linear_q8_n256_up_silu_m64") &&
        q8WideDecodeDualRows(active) &&
        (wide == Q8WideDecode::Dual ||
         (wide == Q8WideDecode::Auto && (active == 40 || active == 48)))) {
      name.replace(name.size() - 3, 3, "d" + std::to_string(active));
      dual = true;
    }
    if (dual && compact && m32Sg16Profile_ && (active == 40 || active == 48) &&
        k == 5120 && (n == 14336 || n == 16640 || n == 17408)) {
      const char *order = std::getenv("SPLASH_DEV_DUAL_DOT_ORDER");
      if (order && std::string_view(order) == "sequential") name += "_seq";
    }
    const bool slab = wide != Q8WideDecode::Native && !dual;
    if (slab) name += "_slab";
    if (compact) name += "_bf16p";
    if (w.phase == LinearPhase::Prefill)
      graph.add(std::move(name), bindings,
                Q8PrefillParams{w.matrix.outputSize, w.matrix.inputSize},
                {selected.storageRows() / kPrefillRows,
                 n / selected.tileColumns(), 1},
                {selected.threadsPerThreadgroup(), 1, 1});
    else {
      const uint32_t groups = selected.configuration().groups;
      graph.add(std::move(name), bindings,
                Q8DecodeParams{n, k, groups},
                {groups, 1, slab ? 2 : 1U},
                {slab || dual ? 256 : selected.threadsPerThreadgroup(), 1, 1});
    }
  };

  // A8 binds the float2 terms in the slot before the parameters; A16 binds
  // the fp32 sums on prefill and nothing extra on decode.
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (selected.secondPipeline().empty()) {
      if (a8)
        dispatch(selected.pipeline(),
                 {b.input, gate->weights, gate->scales, gate->biases, b.output,
                  p.weights, p.scales, p.biases, b.terms});
      else
        dispatch(selected.pipeline(),
                 {b.input, gate->weights, gate->scales, gate->biases, b.output,
                  p.weights, p.scales, p.biases});
    } else {
      // First the gate projection into the scratch, then up x silu(gate).
      if (a8) {
        dispatch(selected.pipeline(), {b.input, gate->weights, gate->scales,
                                       gate->biases, b.gateScratch, b.terms});
        dispatch(selected.secondPipeline(),
                 {b.input, p.weights, p.scales, p.biases, b.gateScratch,
                  b.output, b.terms});
      } else {
        dispatch(selected.pipeline(), {b.input, gate->weights, gate->scales,
                                       gate->biases, b.gateScratch});
        dispatch(selected.secondPipeline(),
                 {b.input, p.weights, p.scales, p.biases, b.gateScratch,
                  b.output});
      }
    }
  } else if (w.epilogue == LinearEpilogue::UpWithGate) {
    if (a8)
      dispatch(selected.pipeline(),
               {b.input, p.weights, p.scales, p.biases, b.gateScratch,
                b.output, b.terms, b.downInput, b.downTerms});
    else
      dispatch(selected.pipeline(),
               {b.input, p.weights, p.scales, p.biases, b.gateScratch,
                b.output, b.sums, b.downSums});
  } else if (w.epilogue == LinearEpilogue::Residual) {
    if (q8ResidualDecodeSplit2Applies(w, p, selected.activation())) {
      requireBytes(b.scratch.input, q8ResidualSplit2ScratchBytes(w));
      const SplitQ8Params params{n, k, 2, w.rows};
      const bool safeM32 = w.rows == 32 && q8Split2M32SafeM16();
      const bool sg16M32 = !safeM32 && w.rows == 32 &&
          q8Split2M32Sg16(m32Sg16Profile_);
      const char *b5Residual = w.rows == 64 && active == 40 && m32Sg16Profile_
          ? std::getenv("SPLASH_DEV_B5_RESIDUAL") : nullptr;
      const bool activeB5 = b5Residual && std::string_view(b5Residual) == "active-sg16";
      const std::string pipeline = activeB5
          ? "decode_q8_split2_partial_m40_sg16" : sg16M32
          ? "decode_q8_split2_partial_m32_sg16"
          : safeM32 ? "decode_q8_split2_partial_m32_safem16"
          : "decode_q8_split2_partial_m" + std::to_string(w.rows);
      graph.add(pipeline,
                {b.input, p.weights, p.scales, p.biases, b.scratch.input},
                params, {n / 128, 2, safeM32 ? 2U : 1U},
                {(sg16M32 || activeB5) ? 512U : 256U, 1, 1});
      if (activeB5)
        graph.add("decode_q8_split2_reduce_rounded_active",
                  {b.scratch.input, b.residual, b.output},
                  ActiveSplitQ8ReduceParams{n, active, w.rows},
                  {(uint64_t{active} * n + 255) / 256, 1, 1}, {256, 1, 1});
      else graph.add("decode_q8_split2_reduce_rounded",
                {b.scratch.input, b.residual, b.output}, params,
                {(uint64_t{w.rows} * n + 255) / 256, 1, 1}, {256, 1, 1});
    } else if (a8)
      dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases,
                                     b.residual, b.output, b.terms});
    else if (w.phase == LinearPhase::Prefill)
      dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases,
                                     b.residual, b.output, b.sums});
    else
      dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases,
                                     b.residual, b.output});
  } else if (a8) {
    dispatch(selected.pipeline(),
             {b.input, p.weights, p.scales, p.biases, b.output, b.terms});
  } else if (w.phase == LinearPhase::Prefill) {
    dispatch(selected.pipeline(),
             {b.input, p.weights, p.scales, p.biases, b.output, b.sums});
  } else {
    dispatch(selected.pipeline(),
             {b.input, p.weights, p.scales, p.biases, b.output});
  }
  if (stats && w.phase == LinearPhase::Decode) {
    const uint32_t lanes = w.rows / SPLASH_TARGET_VERIFY_ROWS;
    if (lanes > 1) {
      stats->fusedSourceOperations += uint64_t{lanes} *
          (selected.secondPipeline().empty() ? 1 : 2);
      if (lanes == 2) stats->m16Dispatches += selected.secondPipeline().empty() ? 1 : 2;
      else if (lanes == 3) stats->m24Dispatches += selected.secondPipeline().empty() ? 1 : 2;
      else if (lanes <= 4)
        stats->m32Dispatches += selected.secondPipeline().empty() ? 1 : 2;
      else stats->m64Dispatches += selected.secondPipeline().empty() ? 1 : 2;
    }
  }
}

void Q8Linear::addPrefill(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q8DenseProjection &p, metal::MetalBuffer output,
    metal::MetalBuffer sums, Q8Scratch scratch, LinearMatrix matrix,
    uint32_t rows) const {
  add(graph, {input, output, sums, {}, {}, scratch.terms, {}, {}, {}, scratch},
      p, plan({matrix, rows, LinearPhase::Prefill, LinearEpilogue::None}));
}
void Q8Linear::addPrefillResidual(metal::CommandGraph &graph,
    metal::MetalBuffer input, const Q8DenseProjection &p,
    metal::MetalBuffer residual, metal::MetalBuffer output,
    metal::MetalBuffer sums, Q8Scratch scratch, LinearMatrix matrix,
    uint32_t rows) const {
  add(graph,
      {input, output, sums, residual, {}, scratch.terms, {}, {}, {}, scratch},
      p, plan({matrix, rows, LinearPhase::Prefill, LinearEpilogue::Residual}));
}
void Q8Linear::addPrefillUpWithGate(metal::CommandGraph &graph,
    metal::MetalBuffer input, const Q8DenseProjection &up,
    metal::MetalBuffer gateScratch, metal::MetalBuffer output,
    metal::MetalBuffer sums, metal::MetalBuffer downSums, Q8Scratch scratch,
    Q8Scratch downScratch, LinearMatrix matrix, uint32_t rows) const {
  add(graph,
      {input, output, sums, {}, gateScratch, scratch.terms, downScratch.input,
       downScratch.terms, downSums, scratch},
      up, plan({matrix, rows, LinearPhase::Prefill, LinearEpilogue::UpWithGate}));
}
void Q8Linear::addDecode(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q8DenseProjection &p, metal::MetalBuffer output, LinearMatrix matrix,
    Q8Scratch scratch) const {
  add(graph, {input, output, {}, {}, {}, scratch.terms, {}, {}, {}, scratch}, p,
      plan(decode(matrix, 1, LinearEpilogue::None)));
}
void Q8Linear::addDecodeBatch(metal::CommandGraph &graph,
    metal::MetalBuffer input, const Q8DenseProjection &p,
    metal::MetalBuffer output, LinearMatrix matrix, uint32_t lanes,
    Q4DispatchStats &stats, Q8Scratch scratch, bool inputPrepared) const {
  add(graph,
      {input, output, {}, {}, {}, scratch.terms, {}, {}, {}, scratch,
       inputPrepared},
      p, plan(decode(matrix, lanes, LinearEpilogue::None)), nullptr, &stats,
      lanes * SPLASH_TARGET_VERIFY_ROWS);
}
void Q8Linear::addResidualBatch(metal::CommandGraph &graph,
    metal::MetalBuffer input, const Q8DenseProjection &p,
    metal::MetalBuffer residual, metal::MetalBuffer output,
    LinearMatrix matrix, uint32_t lanes, Q4DispatchStats &stats,
    Q8Scratch scratch, bool inputPrepared) const {
  // Padded-M32 residual at three lanes: the arena is widened to the M32
  // contract (projectionLanes pads lanes==3 to four), the split2 partial
  // workspace already covers 64 rows, and rows 24-31 are an idle lane slot.
  const LinearWorkload workload{
      matrix, lanes == 3 ? 32 : decodeRowsForLanes(lanes),
      LinearPhase::Decode, LinearEpilogue::Residual};
  add(graph,
      {input, output, {}, residual, {}, scratch.terms, {}, {}, {}, scratch,
       inputPrepared},
      p, plan(workload), nullptr, &stats, lanes * SPLASH_TARGET_VERIFY_ROWS);
}
void Q8Linear::addGateUpBatch(metal::CommandGraph &graph,
    metal::MetalBuffer input, const Q8DenseProjection &gate,
    const Q8DenseProjection &up, metal::MetalBuffer gateScratch,
    metal::MetalBuffer output, LinearMatrix matrix, uint32_t lanes,
    Q4DispatchStats &stats, Q8Scratch scratch, bool inputPrepared) const {
  add(graph,
      {input, output, {}, {}, gateScratch, scratch.terms, {}, {}, {}, scratch,
       inputPrepared},
      up, plan(decode(matrix, lanes, LinearEpilogue::GateUp)), &gate, &stats,
      lanes * SPLASH_TARGET_VERIFY_ROWS);
}

} // namespace splash::ops
