#include "engine/RuntimeResources.hpp"
#include "engine/RestoreBundle.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "model/QwenState.hpp"
#include "model/QwenStateStore.hpp"
#include "ops/KvPageCodec.hpp"
#include "ops/LinearQ8.hpp"

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <future>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <utility>
#include <unordered_map>
#include <unordered_set>

namespace splash::engine {
namespace {

template <typename... Parts>
void logKernelStartup(const Parts &...parts) noexcept {
  try {
    std::ostringstream text;
    (text << ... << parts);
    const std::string message = text.str();
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    char timestamp[9] = "--:--:--";
    if (localtime_r(&now, &local))
      std::strftime(timestamp, sizeof(timestamp), "%H:%M:%S", &local);
    std::ostringstream line;
    line << timestamp << ' ';
    // Native stderr is inherited by serve. Keep each optional startup notice
    // bounded and on one line, including messages from caught exceptions.
    for (unsigned char character : std::string_view(message).substr(0, 768))
      line << (character < 32 || character == 127 ? ' ' : char(character));
    if (message.size() > 768) line << "...";
    std::cerr << line.str() << '\n';
  } catch (...) {
    // Optional diagnostics must not affect startup or serving.
  }
}

static_assert(model::ExecutionLimits::maximumBatchWidth ==
              SPLASH_MAXIMUM_BATCH_WIDTH);
static_assert(model::ExecutionLimits::prefillTokenBudget ==
              SPLASH_PREFILL_TOKEN_BUDGET);
static_assert(model::ExecutionLimits::draftQueryRows ==
              SPLASH_DRAFT_QUERY_ROWS);
static_assert(model::ExecutionLimits::draftProposalTokens ==
              SPLASH_DRAFT_PROPOSAL_TOKENS);
static_assert(model::ExecutionLimits::targetVerifyRows ==
              SPLASH_TARGET_VERIFY_ROWS);
static_assert(model::ExecutionLimits::draftContextTokens ==
              SPLASH_DRAFT_SLIDING_WINDOW);
static_assert(model::ExecutionLimits::speculativeScratchTokens ==
              SPLASH_SPECULATIVE_SCRATCH_TOKENS);

std::string errorText(RuntimeResourceStage stage, std::string_view message,
                      std::string_view budgetDescription) {
  std::ostringstream out;
  out << "runtime resource assembly failed [" << runtimeResourceStageName(stage)
      << "]: " << message;
  if (!budgetDescription.empty()) {
    out << '\n' << budgetDescription;
  }
  return out.str();
}

uint8_t hexNibble(char value) {
  if (value >= '0' && value <= '9') {
    return static_cast<uint8_t>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<uint8_t>(value - 'a' + 10);
  }
  if (value >= 'A' && value <= 'F') {
    return static_cast<uint8_t>(value - 'A' + 10);
  }
  throw std::invalid_argument("manifest SHA-256 is not hexadecimal");
}

uint64_t checkedAdd(uint64_t left, uint64_t right, std::string_view label) {
  if (right > std::numeric_limits<uint64_t>::max() - left) {
    throw std::overflow_error(std::string(label) + " byte count overflows");
  }
  return left + right;
}

uint64_t packedModelFileBytes(const std::filesystem::path &root) {
  uint64_t bytes = 0;
  for (std::string_view directory : {"target", "draft", "vision"}) {
    const std::filesystem::path package = root / directory;
    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(package)) {
      if (!entry.is_regular_file())
        continue;
      bytes = checkedAdd(bytes, entry.file_size(), "model package");
    }
  }
  if (!bytes) {
    throw std::invalid_argument("model package contains no regular files");
  }
  return bytes;
}

uint64_t mebibytes(uint64_t bytes) noexcept { return bytes / kMiB; }

// The startup admission rule. Deliberately independent of the model size:
// weights are mapped, not copied, so the package never has to fit in
// reclaimable memory at once. Residency is what must fit, and it is checked
// here again at every Metal operation as loading and warmup build it up.
void requireStartupHeadroom(
    const MemoryGovernor::HostAvailableMemoryProvider &hostAvailableMemory,
    uint64_t reserveBytes, MemoryPressure pressure) {
  const std::optional<uint64_t> available = hostAvailableMemory();
  if (!available || *available <= reserveBytes ||
      pressure == MemoryPressure::Critical) {
    std::ostringstream message;
    message << "not enough free memory to start: ";
    if (!available)
      message << "reclaimable host memory cannot be measured";
    else
      message << mebibytes(*available) << " MiB reclaimable, "
              << mebibytes(reserveBytes) << " MiB protected for macOS, system "
              << "pressure " << memoryPressureName(pressure);
    message << "; close memory-heavy applications and retry";
    throw metal::MetalAllocationError(message.str(),
                                      metal::AllocationFailure::HostPressure);
  }
}

std::array<uint8_t, 32> parseSha256(std::string_view value) {
  if (value.size() != 64) {
    throw std::invalid_argument(
        "manifest SHA-256 must contain exactly 64 hex characters");
  }
  std::array<uint8_t, 32> result{};
  for (size_t index = 0; index < result.size(); ++index) {
    result[index] = static_cast<uint8_t>((hexNibble(value[index * 2]) << 4) |
                                         hexNibble(value[index * 2 + 1]));
  }
  return result;
}

std::string sha256(std::string_view value) {
  if (value.size() > std::numeric_limits<CC_LONG>::max()) {
    throw std::overflow_error("runtime cache identity is too large to hash");
  }
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  if (!CC_SHA256(value.data(), static_cast<CC_LONG>(value.size()),
                 digest.data())) {
    throw std::runtime_error("runtime cache identity SHA-256 failed");
  }
  return digestHex(digest);
}

std::string
canonicalRuntimeCacheNamespace(const RuntimeCacheIdentity &identity) {
  // Length-prefix the unconstrained strings; every other field has a fixed
  // name and decimal representation. This is the one semantic cache tuple,
  // never a hash of compiler padding or native struct bytes.
  std::ostringstream canonical;
  canonical << "splash.runtime-cache-identity\n"
            << "loaded_model_layout_sha256=" << identity.modelLayoutSha256
            << '\n'
            << "build_id_bytes=" << identity.buildId.size() << '\n'
            << "build_id=" << identity.buildId << '\n'
            << "dtype=" << kv::storageFormatName(identity.kvLayout.format()) << '\n'
            << "page_tokens=" << identity.kvLayout.pageTokens << '\n'
            << "elements_per_scale=" << identity.kvLayout.elementsPerScale
            << '\n'
            << "target_model_sha256="
            << digestHex(identity.kvLayout.modelArtifactSha256) << '\n'
            << "q8_quantization=" << identity.kvLayout.quantization << '\n'
            << "q8_scale_type=" << identity.kvLayout.scaleType << '\n'
            << "q8_key_layout=" << identity.kvLayout.keyLayout << '\n'
            << "q8_value_layout=" << identity.kvLayout.valueLayout << '\n'
            << "q8_attention_layers=" << identity.kvLayout.attentionLayers
            << '\n'
            << "q8_kv_heads=" << identity.kvLayout.kvHeads << '\n'
            << "q8_head_dimension=" << identity.kvLayout.headDimension << '\n'
            << "q8_quantized_minimum=" << identity.kvLayout.quantizedMinimum
            << '\n'
            << "q8_quantized_maximum=" << identity.kvLayout.quantizedMaximum
            << '\n'
            << "q8_bytes_per_layer_page=" << identity.kvLayout.bytesPerLayerPage
            << '\n'
            << "q8_bytes_per_model_page=" << identity.kvLayout.bytesPerModelPage
            << '\n';
  if (identity.prefillStageParameters)
    canonical << "dev_prefill_stage_parameters=1\n";
  if (identity.prefillDirectSingleSplit)
    canonical << "dev_p3a_direct_single_split=1\n";
  if (identity.split2M32SafeM16)
    canonical << "dev_split2_m32_safem16=1\n";
  if (identity.split2M32Sg16)
    canonical << "dev_split2_m32_sg16=1\n";
  if (!identity.pairedLongPrefill)
    canonical << "dev_paired_long_prefill=0\n";
  if (identity.canonicalLongPrefill)
    canonical << "dev_canonical_long_prefill_minimum="
              << identity.canonicalMinimumTokens << '\n';
  if (identity.canonicalDecodeBurst)
    canonical << "dev_canonical_decode_burst=1\n";
  if (identity.decodeBoundaryRetention)
    canonical << "dev_decode_boundary_retention=1\n";
  if (identity.decodeBoundaryReadyOnly)
    canonical << "dev_decode_boundary_ready_only=1\n";
  return sha256(canonical.str());
}

void requireLoadedModel(const model::ModelPackage &package) {
  if (!package.targetActualAllocatedBytes() ||
      !package.draft.actualAllocatedBytes ||
      !package.vision.actualAllocatedBytes ||
      package.manifestFingerprintSha256.empty() ||
      package.targetManifestFingerprint().empty()) {
    throw std::invalid_argument(
        "loaded model package has incomplete allocation accounting");
  }
}

} // namespace

std::string_view runtimeResourceStageName(RuntimeResourceStage stage) {
  switch (stage) {
  case RuntimeResourceStage::Configuration:
    return "configuration";
  case RuntimeResourceStage::BackendCreation:
    return "backend_creation";
  case RuntimeResourceStage::CapabilityValidation:
    return "capability_validation";
  case RuntimeResourceStage::ModelLoading:
    return "model_loading";
  case RuntimeResourceStage::MemoryPlanning:
    return "memory_planning";
  case RuntimeResourceStage::StorageAllocation:
    return "storage_allocation";
  }
  return "unknown";
}

RuntimeCacheIdentity
makeRuntimeCacheIdentity(std::string_view combinedManifestSha256,
                         std::string_view targetManifestSha256,
                         std::string_view buildId,
                         kv::Layout targetKvLayout,
                         bool m32Sg16Profile) {
  if (buildId.empty()) {
    throw std::invalid_argument("runtime build id is required");
  }
  if (!targetKvLayout.valid()) {
    throw std::invalid_argument("runtime target KV layout is invalid");
  }
  // Parse both digests even though only the target digest belongs in the
  // physical-page ABI. This rejects malformed combined manifests early.
  std::array<uint8_t, 32> combinedDigest = parseSha256(combinedManifestSha256);
  std::array<uint8_t, 32> targetDigest = parseSha256(targetManifestSha256);
  RuntimeCacheIdentity result;
  result.modelLayoutSha256 = digestHex(combinedDigest);
  result.buildId = buildId;
  result.prefillStageParameters = ops::q8PrefillStageParameters();
  result.prefillDirectSingleSplit = ops::q8PrefillDirectSingleSplit();
  result.split2M32SafeM16 = ops::q8Split2M32SafeM16();
  result.split2M32Sg16 = ops::q8Split2M32Sg16(m32Sg16Profile);
  const auto enabled = [](const char *name) {
    const char *value = std::getenv(name);
    return value && value[0] == '1' && value[1] == '\0';
  };
  const char *paired = std::getenv("SPLASH_DEV_MAX_CONTEXT_PAIRED_PREFILL");
  result.pairedLongPrefill =
      !(paired && paired[0] == '0' && paired[1] == '\0');
  result.canonicalLongPrefill =
      enabled("SPLASH_DEV_CANONICAL_LONG_PREFILL");
  result.canonicalDecodeBurst =
      enabled("SPLASH_DEV_CANONICAL_DECODE_BURST");
  result.decodeBoundaryRetention =
      enabled("SPLASH_DEV_DECODE_BOUNDARY_RETENTION");
  result.decodeBoundaryReadyOnly =
      enabled("SPLASH_DEV_DECODE_BOUNDARY_READY_ONLY");
  if (result.canonicalLongPrefill) {
    if (const char *minimum = std::getenv("SPLASH_DEV_CANONICAL_MIN_TOKENS")) {
      uint32_t parsed = 0;
      const auto [end, error] =
          std::from_chars(minimum, minimum + std::strlen(minimum), parsed);
      if (error != std::errc{} || *end || parsed < 4096 || parsed > 131072)
        throw std::invalid_argument("invalid canonical prefill minimum");
      result.canonicalMinimumTokens = parsed;
    }
  }
  result.kvLayout = kv::makeLayoutGuard(targetKvLayout, targetDigest);
  result.namespaceSha256 = canonicalRuntimeCacheNamespace(result);
  result.cacheNamespace.digest = parseSha256(result.namespaceSha256);
  return result;
}

namespace {

// Opt-in owner-thread restore and sparse publication. One admitted restore
// owns one destination KV chain; one optional writer pins one immutable
// source chain. Each poll handles at most one completed object/GPU page.
class QwenRestoreCoordinator final : public StorageCoordinator {
  using Clock = std::chrono::steady_clock;
  static uint64_t elapsedUs(Clock::time_point began) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        Clock::now() - began).count();
  }
public:
  QwenRestoreCoordinator(RuntimeResources &resources,
                         const RuntimeResourcesConfig &config)
      : resources_(resources),
        states_(dynamic_cast<model::QwenStateStorage &>(
            resources.stateStorage())),
        store_(*config.ssdCacheRoot, config.ssdTransferBytes,
               config.ssdQuotaBytes, config.ssdFreeFloorBytes,
               developmentIoDelay(),
               [&resources](uint64_t bytes) {
                 return resources.memoryGovernor().tryOwnExternal(bytes);
               }) {
    constexpr uint64_t metadataLimit = uint64_t{64} << 20;
    uint64_t metadataBytes = 0;
    std::vector<std::string> keys;
    try {
      keys = store_.listKeys(
          model::CompositeStateStore::ObjectKind::Manifest, 4096);
    } catch (const std::runtime_error &) {
      // An overfull or temporarily unreadable optional index must not make
      // native inference unavailable. Reuse remains cold until repaired.
      ++safeMisses_;
    }
    for (const std::string &key : keys) {
      const auto size = store_.objectSize(
          key, model::CompositeStateStore::ObjectKind::Manifest);
      if (!size || !*size || *size > 1024 * 1024 ||
          *size > metadataLimit - metadataBytes)
        continue;
      auto pending = store_.readBytesAsync(
          *size, *size, key,
          model::CompositeStateStore::ObjectKind::Manifest);
      if (!pending) continue;
      auto bytes = pending->get();
      if (!bytes || restoreBundleKey(bytes->bytes) != key)
        continue;
      metadataBytes += *size;
      manifests_.push_back({key, std::move(bytes->bytes)});
    }
    metadataBytes_ = metadataBytes;
  }

  ~QwenRestoreCoordinator() override {
    if (job_) {
      job_->cancelToken->store(true);
      if (job_->pending) job_->pending->wait();
      if (job_->cacheRequestStarted)
        resources_.cache().endRequest(job_->id);
    }
    if (writer_) {
      if (writer_->pending) {
        writer_->pending->cancel->store(true);
        writer_->pending->completion.wait();
      }
      resources_.cache().releaseKvBundle(writer_->block);
    }
  }

  bool beginLookup(uint64_t id, uint64_t generation,
                   std::span<const uint32_t> prompt,
                   std::span<const ImageSpan> images) override {
    if (job_ || !images.empty() || prompt.size() < 2)
      return false;
    const auto began = Clock::now();
    Job job;
    job.began = began;
    job.id = id;
    job.generation = generation;
    job.prompt.assign(prompt.begin(), prompt.end());
    for (auto &record : manifests_) {
      try {
        auto candidate = decodeRestoreBundle(
            record.bytes, record.key,
            resources_.cacheIdentity().cacheNamespace.digest,
            job.prompt);
        job.candidates.push_back(std::move(candidate));
        record.lastUse = ++recency_;
      } catch (const std::invalid_argument &) {
      }
    }
    if (job.candidates.empty())
      return false;
    std::sort(job.candidates.begin(), job.candidates.end(),
              [](const RestoreBundle &left, const RestoreBundle &right) {
                return left.boundary > right.boundary;
              });
    job.timing.lookupUs = elapsedUs(began);
    job_.emplace(std::move(job));
    if (!startCandidate()) {
      job_.reset();
      return false;
    }
    return true;
  }

  void beginPublish(uint64_t block, uint32_t boundary,
                    std::span<const uint32_t> tokens,
                    std::span<const ImageSpan> images,
                    std::shared_ptr<const CompositeState> state) override {
    if (writer_) {
      ++skippedPublications_;
      return;
    }
    if (!state || !images.empty() || !boundary ||
        boundary % kv::kPageTokens || tokens.size() != boundary)
      return;
    if (completeBundleExists(block, boundary, tokens))
      return;
    auto transfer = store_.reserveState(states_);
    if (!transfer) {
      ++skippedPublications_;
      if (store_.gcOwned() && store_.diskUsed() + states_.serializedBytes() >
                                  store_.diskQuota()) {
        if (!gcPending_) gcProtectedManifests_.clear();
        for (const auto &record : manifests_) {
          try {
            static_cast<void>(decodeRestoreBundle(
                record.bytes, record.key,
                resources_.cacheIdentity().cacheNamespace.digest, tokens));
            gcProtectedManifests_.insert(record.key);
          } catch (const std::invalid_argument &) {
          }
        }
        gcPending_ = true;
        gcRequestedBoundary_ = std::max(gcRequestedBoundary_, boundary);
        if (notifier_) notifier_();
      }
      return; // Optional persistence is skipped before capture.
    }
    resources_.cache().retainKvBundle(block);
    try {
      WriteJob writer;
      writer.block = block;
      writer.boundary = boundary;
      writer.tokens.assign(tokens.begin(), tokens.end());
      writer.state = std::move(state);
      writer.transfer.emplace(std::move(*transfer));
      writer_.emplace(std::move(writer));
    } catch (...) {
      resources_.cache().releaseKvBundle(block);
      throw;
    }
    if (notifier_) notifier_();
  }

  bool beginPublishFromCapture(
      uint64_t block, uint32_t boundary,
      std::span<const uint32_t> tokens,
      std::span<const ImageSpan> images,
      const std::function<std::shared_ptr<const CompositeState>()> &capture) override {
    // Reserve before copying the active GDN/DFlash state. An unavailable
    // optional writer, transfer, or host-growth budget leaves the prior bundle
    // untouched and never blocks a decode command on filesystem work.
    ++decodeCaptureOpportunities_;
    if (writer_) {
      ++decodeSkipWriter_;
      ++skippedPublications_;
      return false;
    }
    if (!images.empty() || !boundary ||
        boundary % kv::kPageTokens || tokens.size() != boundary) {
      ++decodeSkipInvalid_;
      ++skippedPublications_;
      return false;
    }
    if (!resources_.memoryGovernor().snapshot().hostGrowthAllowed) {
      ++decodeSkipPressure_;
      ++skippedPublications_;
      return false;
    }
    if (completeBundleExists(block, boundary, tokens)) {
      ++decodeSkipExisting_;
      return true;
    }
    auto transfer = store_.reserveState(states_);
    if (!transfer) {
      ++decodeSkipTransfer_;
      ++skippedPublications_;
      return false;
    }
    std::shared_ptr<const CompositeState> state;
    try {
      state = capture();
    } catch (const std::exception &) {
      ++decodeSkipSnapshot_;
      ++skippedPublications_;
      return false;
    }
    if (!state) {
      ++decodeSkipSnapshot_;
      ++skippedPublications_;
      return false;
    }
    resources_.cache().retainKvBundle(block);
    try {
      WriteJob writer;
      writer.block = block;
      writer.boundary = boundary;
      writer.tokens.assign(tokens.begin(), tokens.end());
      writer.state = std::move(state);
      writer.transfer.emplace(std::move(*transfer));
      writer_.emplace(std::move(writer));
    } catch (...) {
      resources_.cache().releaseKvBundle(block);
      throw;
    }
    if (notifier_) notifier_();
    return true;
  }
  bool captureReady() const noexcept override {
    return !writer_ &&
           resources_.memoryGovernor().snapshot().hostGrowthAllowed;
  }

  std::vector<StorageCompletion> poll() override {
    lastPollProgressed_ = false;
    if (!job_) {
      progressWriter();
      if (!writer_ && gcPending_) runGcCycle();
      return {};
    }
    Job &job = *job_;
    if (job.cancelled && !job.pending) {
      cleanup();
      return {};
    }
    if (job.stage == Stage::KvReady) {
      try {
        const uint64_t pageBytes = kv::pageObjectBytes(
            resources_.targetKvPages().layout());
        const auto payload = std::span<const uint8_t>(job.extent->bytes).subspan(
            (job.pageIndex - job.extentStart) * pageBytes, pageBytes);
        installKvPage(job, payload);
        lastPollProgressed_ = true;
        return afterPage();
      } catch (const metal::MetalBackendError &) {
        throw;
      } catch (const std::exception &) {
        return advance();
      }
    }
    if (!job.pending ||
        job.pending->wait_for(std::chrono::milliseconds(0)) !=
            std::future_status::ready)
      return {};
    lastPollProgressed_ = true;
    try {
      auto object = job.pending->get();
      job.pending.reset();
      if (job.cancelled) {
        cleanup();
        return {};
      }
      if (!object)
        return advance();
      job.timing.bufferPrepareUs += object->bufferPrepareUs;
      if (job.stage == Stage::State) {
        job.timing.stateReadWallUs += elapsedUs(job.readBegan);
        job.timing.stateFileReadUs += object->filesystemReadUs;
      } else {
        job.timing.kvReadWallUs += elapsedUs(job.readBegan);
        job.timing.kvFileReadUs += object->filesystemReadUs;
      }
      const auto &bundle = job.candidates[job.candidateIndex];
      if (job.stage == Stage::State) {
        const auto validateBegan = Clock::now();
        if (kv::pageObjectKey(object->bytes) != bundle.stateKey) {
          invalidStateKeys_.insert(bundle.stateKey);
          return advance();
        }
        states_.validateSerialized(object->bytes);
        job.timing.validationUs += elapsedUs(validateBegan);
        const auto deserializeBegan = Clock::now();
        job.state = states_.deserialize(object->bytes);
        job.timing.stateDeserializeUs += elapsedUs(deserializeBegan);
        object.reset();
        if (!job.state || !startPage())
          return advance();
        return {};
      }
      const kv::Layout layout = resources_.targetKvPages().layout();
      const std::string &key = bundle.kvKeys[job.pageIndex];
      const auto validateBegan = Clock::now();
      if (kv::pageObjectKey(object->bytes) != key)
        return advance();
      if (job.stage == Stage::KvExtent) {
        const uint64_t pageBytes = kv::pageObjectBytes(layout);
        if (object->bytes.empty() ||
            object->bytes.size() % pageBytes ||
            object->bytes.size() > 64 * pageBytes)
          return advance();
        const size_t count = object->bytes.size() / pageBytes;
        if (count > bundle.kvKeys.size() - job.pageIndex)
          return advance();
        for (size_t offset = 0; offset < count; ++offset)
          if (bundle.kvKeys[job.pageIndex + offset] != key)
            return advance();
        job.timing.validationUs += elapsedUs(validateBegan);
        job.extent.emplace(std::move(*object));
        job.extentStart = job.pageIndex;
        job.extentPages = count;
        job.stage = Stage::KvReady;
        return {};
      }
      job.timing.validationUs += elapsedUs(validateBegan);
      installKvPage(job, object->bytes);
      object.reset();
      return afterPage();
    } catch (const metal::MetalBackendError &) {
      throw;
    } catch (const std::exception &) {
      return advance();
    }
  }

  bool pending() const noexcept override {
    return job_.has_value() || writer_.has_value() || gcPending_;
  }
  bool pollProgressed() const noexcept override { return lastPollProgressed_; }
  StorageSnapshot snapshot() const noexcept override {
    return {store_.transferUsed(), store_.transferPeak(),
            store_.diskUsed(), store_.diskPeak(), store_.diskQuota(),
            store_.freeFloor(),
            static_cast<uint32_t>(manifests_.size()),
            job_ ? 1U : 0U, writer_ ? 1U : 0U,
            restores_, walkbacks_, safeMisses_, publications_,
            skippedPublications_, gcCycles_, gcObjects_, gcBytes_, gcBundles_,
            lastRestoreTiming_, decodeCaptureOpportunities_, decodeSkipWriter_,
            decodeSkipInvalid_, decodeSkipPressure_, decodeSkipTransfer_,
            decodeSkipSnapshot_, decodeSkipExisting_};
  }
  void cancel(uint64_t id) noexcept override {
    if (job_ && job_->id == id) {
      job_->cancelled = true;
      job_->cancelToken->store(true);
    }
  }
  void setNotifier(std::function<void()> notifier) override {
    notifier_ = std::move(notifier);
  }

private:
  bool durableKvObjectAvailable(const std::string &key,
                                 uint64_t pageBytes) const {
    if (store_.objectSize(key,
            model::CompositeStateStore::ObjectKind::KvPage) == pageBytes)
      return true;
    const auto extent = store_.objectSize(
        key, model::CompositeStateStore::ObjectKind::KvExtent);
    return extent && *extent >= pageBytes &&
           *extent <= 64 * pageBytes && *extent % pageBytes == 0;
  }

  void retainLineage(uint64_t block,
                     const std::vector<std::string> &keys) {
    if (durableKvLineage_.size() >= 4096)
      durableKvLineage_.erase(durableKvLineage_.begin());
    durableKvLineage_[block] = keys;
  }

  bool completeBundleExists(uint64_t block, uint32_t boundary,
                            std::span<const uint32_t> tokens) {
    const uint64_t pageBytes = kv::pageObjectBytes(
        resources_.targetKvPages().layout());
    for (const auto &record : manifests_) {
      try {
        const auto bundle = decodeRestoreBundle(
            record.bytes, record.key,
            resources_.cacheIdentity().cacheNamespace.digest, tokens);
        if (bundle.boundary != boundary ||
            invalidStateKeys_.contains(bundle.stateKey) ||
            store_.objectSize(bundle.stateKey,
                model::CompositeStateStore::ObjectKind::CompositeState) !=
                states_.serializedBytes())
          continue;
        bool complete = true;
        for (const auto &key : bundle.kvKeys) {
          if (!durableKvObjectAvailable(key, pageBytes)) {
            complete = false;
            break;
          }
        }
        if (complete) {
          retainLineage(block, bundle.kvKeys);
          return true;
        }
      } catch (const std::invalid_argument &) {
      }
    }
    return false;
  }
  static uint32_t developmentIoDelay() {
    const char *value = std::getenv("SPLASH_DEV_SSD_DELAY_MS");
    if (!value) return 0;
    const std::string_view text(value);
    uint32_t delay = 0;
    const auto parsed = std::from_chars(text.data(),
                                        text.data() + text.size(), delay);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size() || delay > 5000)
      throw std::invalid_argument("invalid development SSD delay");
    return delay;
  }
  enum class Stage : uint8_t { State, Kv, KvExtent, KvReady };
  enum class WriteStage : uint8_t {
    StateEncode, StatePending, KvCapture, KvPending,
    ManifestEncode, ManifestPending
  };
  struct ManifestRecord {
    std::string key;
    std::vector<uint8_t> bytes;
    uint64_t lastUse = 0;
  };

  void runGcCycle() noexcept {
    // Each cycle scans a bounded index and retires at most sixteen orphan
    // objects OR one cold manifest. Re-scanning recovers after a crash between
    // manifest retirement and last-owner deletion; no RAM-only GC queue.
    if (job_ || writer_ || !store_.gcOwned()) return;
    try {
      ++gcCycles_;
      using Kind = model::CompositeStateStore::ObjectKind;
      const auto diskManifests = store_.listKeys(Kind::Manifest, 4096);
      std::vector<std::string> indexed;
      indexed.reserve(manifests_.size());
      for (const auto &record : manifests_) indexed.push_back(record.key);
      std::sort(indexed.begin(), indexed.end());
      if (diskManifests != indexed) {
        gcPending_ = false;
        gcRequestedBoundary_ = 0;
        gcProtectedManifests_.clear();
        return;
      }
      std::vector<RestoreBundle> bundles;
      bundles.reserve(manifests_.size());
      std::unordered_set<std::string> liveStates, liveKv;
      for (const auto &record : manifests_) {
        bundles.push_back(inspectRestoreBundle(
            record.bytes, record.key,
            resources_.cacheIdentity().cacheNamespace.digest));
        liveStates.insert(bundles.back().stateKey);
        liveKv.insert(bundles.back().kvKeys.begin(), bundles.back().kvKeys.end());
      }
      size_t removed = 0;
      const auto sweep = [&](Kind kind, const std::unordered_set<std::string> &live) {
        for (const auto &key : store_.listKeys(kind, 16384)) {
          if (removed == 16) break;
          if (!live.contains(key)) {
            if (const auto bytes = store_.removeUnreferencedObject(key, kind)) {
              ++removed;
              ++gcObjects_;
              gcBytes_ += *bytes;
            }
          }
        }
      };
      sweep(Kind::CompositeState, liveStates);
      if (removed < 16) sweep(Kind::KvExtent, liveKv);
      if (removed < 16) sweep(Kind::KvPage, liveKv);
      if (removed) {
        if (removed < 16 &&
            store_.diskUsed() + states_.serializedBytes() <= store_.diskQuota())
          gcPending_ = false;
        if (!gcPending_) gcRequestedBoundary_ = 0;
        if (!gcPending_) gcProtectedManifests_.clear();
        return;
      }
      if (manifests_.empty() ||
          store_.diskUsed() + states_.serializedBytes() <= store_.diskQuota()) {
        gcPending_ = false;
        gcRequestedBoundary_ = 0;
        gcProtectedManifests_.clear();
        return;
      }
      size_t victim = manifests_.size();
      for (size_t i = 0; i < manifests_.size(); ++i) {
        if (bundles[i].boundary >= gcRequestedBoundary_ ||
            gcProtectedManifests_.contains(manifests_[i].key)) continue;
        if (victim == manifests_.size() ||
            std::pair{manifests_[i].lastUse, bundles[i].boundary} <
                std::pair{manifests_[victim].lastUse, bundles[victim].boundary})
          victim = i;
      }
      if (victim == manifests_.size()) {
        // Never trade a longer, useful cached prefix for a shorter optional
        // publication merely to make the quota arithmetic fit.
        gcPending_ = false;
        gcRequestedBoundary_ = 0;
        gcProtectedManifests_.clear();
        return;
      }
      const auto manifestBytes = store_.removeUnreferencedObject(
          manifests_[victim].key, Kind::Manifest);
      if (!manifestBytes) {
        gcPending_ = false;
        gcRequestedBoundary_ = 0;
        gcProtectedManifests_.clear();
        return;
      }
      metadataBytes_ -= manifests_[victim].bytes.size();
      ++gcObjects_;
      ++gcBundles_;
      gcBytes_ += *manifestBytes;
      manifests_.erase(manifests_.begin() + victim);
      durableKvLineage_.clear();
    } catch (...) {
      gcPending_ = false; // Preserve data on malformed/unknown ownership.
      gcRequestedBoundary_ = 0;
      gcProtectedManifests_.clear();
    }
  }
  struct Job {
    uint64_t id = 0;
    uint64_t generation = 0;
    Clock::time_point began{};
    Clock::time_point readBegan{};
    RestoreTimingSnapshot timing;
    std::vector<uint32_t> prompt;
    std::vector<RestoreBundle> candidates;
    size_t candidateIndex = 0;
    size_t pageIndex = 0;
    Stage stage = Stage::State;
    bool cacheRequestStarted = false;
    bool cancelled = false;
    std::vector<uint32_t> pages;
    std::shared_ptr<const model::QwenCompositeState> state;
    std::shared_ptr<std::atomic_bool> cancelToken =
        std::make_shared<std::atomic_bool>(false);
    std::optional<std::future<std::optional<
        model::CompositeStateStore::Transfer>>> pending;
    std::optional<model::CompositeStateStore::Transfer> extent;
    size_t extentStart = 0;
    size_t extentPages = 0;
  };

  void installKvPage(Job &job, std::span<const uint8_t> objectBytes) {
    const kv::Layout layout = resources_.targetKvPages().layout();
    metal::MetalBuffer staging;
    const auto admissionBegan = Clock::now();
    auto admitted = resources_.memoryGovernor().allocationAdmission()(
        layout.bytesPerModelPage(), [&] {
          staging = resources_.backend().allocateBuffer(
              layout.bytesPerModelPage(), metal::BufferStorage::Shared,
              "serving SSD KV staging");
        });
    if (!admitted || !staging)
      throw std::runtime_error("SSD KV staging refused");
    job.timing.gpuAdmissionUs += elapsedUs(admissionBegan);
    const auto cpuInstallBegan = Clock::now();
    kv::decodePageObject(
        layout,
        static_cast<uint32_t>((job.pageIndex + 1) * kv::kPageTokens),
        resources_.cacheIdentity().cacheNamespace.digest,
        std::span<const uint32_t>(job.prompt).subspan(
            job.pageIndex * kv::kPageTokens, kv::kPageTokens),
        objectBytes, staging);
    job.timing.kvCpuInstallUs += elapsedUs(cpuInstallBegan);
    const auto gpuInstallBegan = Clock::now();
    metal::CommandGraph graph;
    resources_.targetKvPages().encodePageCopy(
        graph, job.pages[job.pageIndex], staging, true);
    (void)resources_.backend().submitCommand(graph.dispatches());
    job.timing.kvGpuInstallUs += elapsedUs(gpuInstallBegan);
  }

  std::vector<StorageCompletion> afterPage() {
    Job &job = *job_;
    ++job.pageIndex;
    if (job.extent &&
        job.pageIndex == job.extentStart + job.extentPages) {
      job.extent.reset();
      job.extentPages = 0;
    }
    const auto &bundle = job.candidates[job.candidateIndex];
    if (job.pageIndex < bundle.kvKeys.size()) {
      if (!startPage()) return advance();
      return {};
    }
    const auto publishBegan = Clock::now();
    const uint64_t block = resources_.cache().publishCommittedBlocks(
        job.id, job.prompt, bundle.boundary);
    resources_.cache().publishCompositeState(block, std::move(job.state));
    job.timing.statePublishUs += elapsedUs(publishBegan);
    retainLineage(block, bundle.kvKeys);
    resources_.cache().endRequest(job.id);
    const StorageCompletion result{job.id, job.generation};
    ++restores_;
    walkbacks_ += job.candidateIndex > 0;
    job.timing.totalUs = elapsedUs(job.began);
    const uint64_t attributed = job.timing.lookupUs +
        job.timing.stateReadWallUs + job.timing.kvReadWallUs +
        job.timing.validationUs + job.timing.stateDeserializeUs +
        job.timing.gpuAdmissionUs + job.timing.kvCpuInstallUs +
        job.timing.kvGpuInstallUs + job.timing.statePublishUs;
    job.timing.ownerGapUs = job.timing.totalUs > attributed
        ? job.timing.totalUs - attributed : 0;
    lastRestoreTiming_ = job.timing;
    job_.reset();
    if (notifier_) notifier_();
    return {result};
  }

  struct WriteJob {
    uint64_t block = 0;
    uint32_t boundary = 0;
    WriteStage stage = WriteStage::StateEncode;
    size_t pageIndex = 0;
    size_t extentStart = 0;
    size_t extentPages = 0;
    size_t extentCaptured = 0;
    std::vector<uint32_t> tokens;
    std::vector<uint32_t> pages;
    std::vector<std::string> kvKeys;
    std::string stateKey;
    std::string extentKey;
    std::string manifestKey;
    std::vector<uint8_t> manifestBytes;
    std::shared_ptr<const CompositeState> state;
    std::optional<model::CompositeStateStore::Transfer> transfer;
    std::optional<model::CompositeStateStore::Transfer> extentTransfer;
    std::optional<model::CompositeStateStore::Write> pending;
    bool aborting = false;
  };

  void finishWriter() noexcept {
    if (!writer_) return;
    resources_.cache().releaseKvBundle(writer_->block);
    writer_.reset();
    lastPollProgressed_ = true;
    if (notifier_) notifier_();
  }

  void progressWriter() {
    if (!writer_) return;
    WriteJob &writer = *writer_;
    if (!resources_.memoryGovernor().snapshot().hostGrowthAllowed)
      writer.aborting = true;
    if (writer.aborting) {
      if (writer.pending) {
        writer.pending->cancel->store(true);
        if (writer.pending->completion.wait_for(
                std::chrono::milliseconds(0)) != std::future_status::ready)
          return;
        static_cast<void>(writer.pending->completion.get());
      }
      finishWriter();
      return;
    }
    try {
      switch (writer.stage) {
      case WriteStage::StateEncode: {
        lastPollProgressed_ = true;
        states_.serialize(*writer.state, writer.transfer->bytes);
        writer.stateKey = kv::pageObjectKey(writer.transfer->bytes);
        writer.pending = store_.writeBytesAsync(
            std::move(*writer.transfer), writer.stateKey,
            model::CompositeStateStore::ObjectKind::CompositeState);
        writer.transfer.reset();
        if (!writer.pending) throw std::runtime_error("SSD state write refused");
        writer.stage = WriteStage::StatePending;
        return;
      }
      case WriteStage::StatePending: {
        if (writer.pending->completion.wait_for(std::chrono::milliseconds(0)) !=
            std::future_status::ready) return;
        const bool written = writer.pending->completion.get();
        lastPollProgressed_ = true;
        writer.pending.reset();
        if (!written) throw std::runtime_error("SSD state write failed");
        invalidStateKeys_.erase(writer.stateKey);
        writer.state.reset();
        const auto chain = resources_.cache().residentKvChain(writer.block);
        if (chain.pages.size() != writer.boundary / kv::kPageTokens)
          throw std::logic_error("SSD KV source chain changed");
        writer.pages = std::move(chain.pages);
        for (size_t depth = chain.blocks.size(); depth > 0; --depth) {
          auto found = durableKvLineage_.find(chain.blocks[depth - 1]);
          if (found != durableKvLineage_.end() &&
              found->second.size() == depth) {
            writer.kvKeys = found->second;
            writer.pageIndex = depth;
            break;
          }
        }
        writer.kvKeys.reserve(writer.pages.size());
        writer.stage = WriteStage::KvCapture;
        return;
      }
      case WriteStage::KvCapture: {
        lastPollProgressed_ = true;
        if (writer.pageIndex == writer.pages.size()) {
          writer.stage = WriteStage::ManifestEncode;
          return;
        }
        const kv::Layout layout = resources_.targetKvPages().layout();
        const uint64_t pageBytes = kv::pageObjectBytes(layout);
        if (!writer.extentTransfer) {
          writer.extentStart = writer.pageIndex;
          writer.extentPages = std::min<size_t>(
              64, writer.pages.size() - writer.pageIndex);
          writer.extentCaptured = 0;
          const uint64_t extentBytes = pageBytes * writer.extentPages;
          auto transfer = store_.reserveWriteBytes(
              extentBytes + layout.bytesPerModelPage(), extentBytes);
          if (!transfer)
            throw std::runtime_error("SSD KV extent transfer refused");
          writer.extentTransfer.emplace(std::move(*transfer));
        }
        metal::MetalBuffer staging;
        auto admitted = resources_.memoryGovernor().allocationAdmission()(
            layout.bytesPerModelPage(), [&] {
              staging = resources_.backend().allocateBuffer(
                  layout.bytesPerModelPage(), metal::BufferStorage::Shared,
                  "serving SSD KV capture");
            });
        if (!admitted || !staging)
          throw std::runtime_error("SSD KV staging refused");
        metal::CommandGraph graph;
        resources_.targetKvPages().encodePageCopy(
            graph, writer.pages[writer.pageIndex], staging, false);
        (void)resources_.backend().submitCommand(graph.dispatches());
        kv::encodePageObject(
            layout,
            static_cast<uint32_t>((writer.pageIndex + 1) * kv::kPageTokens),
            resources_.cacheIdentity().cacheNamespace.digest,
            std::span<const uint32_t>(writer.tokens).subspan(
                writer.pageIndex * kv::kPageTokens, kv::kPageTokens),
            staging,
            std::span<uint8_t>(writer.extentTransfer->bytes).subspan(
                writer.extentCaptured * pageBytes, pageBytes));
        ++writer.pageIndex;
        ++writer.extentCaptured;
        if (writer.extentCaptured < writer.extentPages)
          return;
        writer.extentKey = kv::pageObjectKey(
            writer.extentTransfer->bytes);
        writer.pending = store_.writeBytesAsync(
            std::move(*writer.extentTransfer), writer.extentKey,
            model::CompositeStateStore::ObjectKind::KvExtent);
        writer.extentTransfer.reset();
        if (!writer.pending)
          throw std::runtime_error("SSD KV extent write refused");
        writer.stage = WriteStage::KvPending;
        return;
      }
      case WriteStage::KvPending: {
        if (writer.pending->completion.wait_for(std::chrono::milliseconds(0)) !=
            std::future_status::ready) return;
        const bool written = writer.pending->completion.get();
        lastPollProgressed_ = true;
        writer.pending.reset();
        if (!written) throw std::runtime_error("SSD KV extent write failed");
        writer.kvKeys.insert(writer.kvKeys.end(), writer.extentPages,
                             writer.extentKey);
        writer.extentPages = 0;
        writer.extentCaptured = 0;
        writer.extentKey.clear();
        writer.stage = WriteStage::KvCapture;
        return;
      }
      case WriteStage::ManifestEncode: {
        lastPollProgressed_ = true;
        RestoreBundle bundle;
        bundle.namespaceDigest = resources_.cacheIdentity().cacheNamespace.digest;
        bundle.boundary = writer.boundary;
        bundle.stateKey = writer.stateKey;
        bundle.kvKeys = writer.kvKeys;
        const uint64_t bytes = restoreBundleBytes(bundle);
        auto transfer = store_.reserveWriteBytes(bytes, bytes);
        if (!transfer) throw std::runtime_error("SSD manifest refused");
        encodeRestoreBundle(bundle, writer.tokens, transfer->bytes);
        writer.manifestKey = restoreBundleKey(transfer->bytes);
        writer.manifestBytes = transfer->bytes;
        writer.pending = store_.writeBytesAsync(
            std::move(*transfer), writer.manifestKey,
            model::CompositeStateStore::ObjectKind::Manifest);
        if (!writer.pending) throw std::runtime_error("SSD manifest write refused");
        writer.stage = WriteStage::ManifestPending;
        return;
      }
      case WriteStage::ManifestPending: {
        if (writer.pending->completion.wait_for(std::chrono::milliseconds(0)) !=
            std::future_status::ready) return;
        const bool written = writer.pending->completion.get();
        lastPollProgressed_ = true;
        writer.pending.reset();
        if (!written) throw std::runtime_error("SSD manifest write failed");
        constexpr uint64_t metadataLimit = uint64_t{64} << 20;
        if (writer.manifestBytes.size() <= metadataLimit - metadataBytes_) {
          metadataBytes_ += writer.manifestBytes.size();
          manifests_.push_back({writer.manifestKey,
                                std::move(writer.manifestBytes), ++recency_});
        }
        retainLineage(writer.block, writer.kvKeys);
        ++publications_;
        finishWriter();
        return;
      }
      }
    } catch (const metal::MetalBackendError &) {
      throw;
    } catch (const std::exception &) {
      writer.aborting = true;
      ++skippedPublications_;
      if (!writer.pending) finishWriter();
    }
  }

  bool startCandidate() {
    Job &job = *job_;
    while (job.candidateIndex < job.candidates.size()) {
      const auto &bundle = job.candidates[job.candidateIndex];
      resources_.cache().beginRequest(job.id);
      job.cacheRequestStarted = true;
      if (!resources_.cache().ensureTokens(job.id, bundle.boundary).granted()) {
        resources_.cache().endRequest(job.id);
        job.cacheRequestStarted = false;
        ++job.candidateIndex;
        continue;
      }
      const auto pages = resources_.cache().pageTable(job.id).pages;
      job.pages.assign(pages.begin(), pages.end());
      job.cancelToken = std::make_shared<std::atomic_bool>(false);
      auto read = store_.readAsync(states_.serializedBytes(),
                                    bundle.stateKey, job.cancelToken);
      if (!read) {
        resources_.cache().endRequest(job.id);
        job.cacheRequestStarted = false;
        ++job.candidateIndex;
        continue;
      }
      job.pending = std::move(*read);
      job.readBegan = Clock::now();
      job.stage = Stage::State;
      job.pageIndex = 0;
      return true;
    }
    return false;
  }

  bool startPage() {
    Job &job = *job_;
    if (job.extent) {
      job.stage = Stage::KvReady;
      return true;
    }
    const auto &bundle = job.candidates[job.candidateIndex];
    const uint64_t bytes = kv::pageObjectBytes(
        resources_.targetKvPages().layout());
    const std::string &key = bundle.kvKeys[job.pageIndex];
    std::optional<std::future<std::optional<
        model::CompositeStateStore::Transfer>>> read;
    if (store_.objectSize(
            key, model::CompositeStateStore::ObjectKind::KvPage) == bytes) {
      read = store_.readBytesAsync(
          bytes, 2 * bytes, key,
          model::CompositeStateStore::ObjectKind::KvPage,
          job.cancelToken);
      job.stage = Stage::Kv;
    } else {
      const auto extent = store_.objectSize(
          key, model::CompositeStateStore::ObjectKind::KvExtent);
      if (!extent || *extent < bytes || *extent > 64 * bytes ||
          *extent % bytes)
        return false;
      read = store_.readBytesAsync(
          *extent,
          *extent + resources_.targetKvPages().layout().bytesPerModelPage(),
          key, model::CompositeStateStore::ObjectKind::KvExtent,
          job.cancelToken);
      job.stage = Stage::KvExtent;
    }
    if (!read) return false;
    job.pending = std::move(*read);
    job.readBegan = Clock::now();
    return true;
  }

  void cleanup() {
    if (job_->cacheRequestStarted)
      resources_.cache().endRequest(job_->id);
    job_.reset();
  }

  std::vector<StorageCompletion> advance() {
    Job &job = *job_;
    if (job.cacheRequestStarted) {
      resources_.cache().endRequest(job.id);
      job.cacheRequestStarted = false;
    }
    job.state.reset();
    job.extent.reset();
    job.extentPages = 0;
    job.pages.clear();
    ++job.candidateIndex;
    if (startCandidate()) return {};
    const StorageCompletion result{job.id, job.generation};
    ++safeMisses_;
    job_.reset();
    if (notifier_) notifier_();
    return {result}; // safe cold miss after bounded walkback
  }

  RuntimeResources &resources_;
  model::QwenStateStorage &states_;
  model::CompositeStateStore store_;
  std::vector<ManifestRecord> manifests_;
  uint64_t recency_ = 0;
  bool gcPending_ = false;
  bool lastPollProgressed_ = false;
  uint32_t gcRequestedBoundary_ = 0;
  std::unordered_set<std::string> gcProtectedManifests_;
  std::unordered_set<std::string> invalidStateKeys_;
  std::unordered_map<uint64_t, std::vector<std::string>> durableKvLineage_;
  uint64_t metadataBytes_ = 0;
  std::optional<Job> job_;
  std::optional<WriteJob> writer_;
  uint64_t restores_ = 0;
  uint64_t walkbacks_ = 0;
  uint64_t safeMisses_ = 0;
  uint64_t publications_ = 0;
  uint64_t skippedPublications_ = 0;
  uint64_t decodeCaptureOpportunities_ = 0;
  uint64_t decodeSkipWriter_ = 0;
  uint64_t decodeSkipInvalid_ = 0;
  uint64_t decodeSkipPressure_ = 0;
  uint64_t decodeSkipTransfer_ = 0;
  uint64_t decodeSkipSnapshot_ = 0;
  uint64_t decodeSkipExisting_ = 0;
  uint64_t gcCycles_ = 0;
  uint64_t gcObjects_ = 0;
  uint64_t gcBytes_ = 0;
  uint64_t gcBundles_ = 0;
  RestoreTimingSnapshot lastRestoreTiming_;
  std::function<void()> notifier_;
};

} // namespace

RuntimeResourcesError::RuntimeResourcesError(RuntimeResourceStage stage,
                                             std::string message,
                                             std::string statusJson,
                                             std::string budgetDescription,
                                             RuntimeResourceFailure failure)
    : std::runtime_error(errorText(stage, message, budgetDescription)),
      failure_(failure),
      message_(std::move(message)), statusJson_(std::move(statusJson)),
      budgetDescription_(std::move(budgetDescription)) {}

RuntimeResources::RuntimeResources(
    std::unique_ptr<metal::MetalBackend> backend, model::ModelPackage model,
    ops::ExecutionPlans operators,
    EngineMemoryPlan memoryPlan, model::ModelMemoryPlan modelMemoryPlan,
    RuntimeCacheIdentity cacheIdentity,
    std::unique_ptr<MemoryGovernor> memoryGovernor,
    std::unique_ptr<kv::PageStorage> kvPages,
    std::unique_ptr<model::StateStorage> stateStorage,
    std::unique_ptr<KvPool> kvPool, std::unique_ptr<engine::Cache> cache,
    uint32_t maximumImagePatches)
    : backend_(std::move(backend)), model_(std::move(model)),
      operators_(std::move(operators)),
      memoryPlan_(std::move(memoryPlan)),
      modelMemoryPlan_(std::move(modelMemoryPlan)),
      cacheIdentity_(std::move(cacheIdentity)),
      memoryGovernor_(std::move(memoryGovernor)), kvPages_(std::move(kvPages)),
      stateStorage_(std::move(stateStorage)), kvPool_(std::move(kvPool)),
      cache_(std::move(cache)), maximumImagePatches_(maximumImagePatches) {}

std::unique_ptr<RuntimeResources>
RuntimeResources::create(const RuntimeResourcesConfig &config) {
  if (config.metallibPath.empty() || config.modelRoot.empty() ||
      !kv::validFormat(config.kvFormat) ||
      !config.model.valid() ||
      config.buildId.empty() || !config.maximumImagePatches ||
      config.maximumImagePatches % 4) {
    throw RuntimeResourcesError(
        RuntimeResourceStage::Configuration,
        "metallib path, model root, build id, and a merge-aligned image "
        "patch limit are required");
  }
  std::unique_ptr<metal::MetalBackend> backend;
  try {
    backend =
        std::make_unique<metal::MetalBackend>(config.metallibPath.string());
  } catch (const metal::MetalAllocationError &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::BackendCreation,
                                error.what(), {}, {},
                                resourceAllocationFailure(error.failure()));
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::BackendCreation,
                                error.what());
  }

  const DeviceCapabilities &device = backend->capabilities();
  if (auto error = device.validationError()) {
    throw RuntimeResourcesError(RuntimeResourceStage::CapabilityValidation,
                                *error, deviceStatusJson(device));
  }

  const uint64_t hostReserveBytes =
      EngineMemoryPolicy::hostAvailableReserveBytes(device.physicalMemoryBytes);
  MemoryGovernor::HostAvailableMemoryProvider hostAvailableMemory =
      config.hostAvailableMemory ? config.hostAvailableMemory
                                 : queryHostAvailableMemory;
  backend->setOperationGuard(
      [cancelled = config.cancelled, pressure = config.memoryPressure,
       hostAvailableMemory, hostReserveBytes] {
        if (cancelled && cancelled())
          throw metal::MetalBackendError("Metal operation cancelled");
        requireStartupHeadroom(hostAvailableMemory, hostReserveBytes,
            pressure ? pressure() : MemoryPressure::Normal);
      });
  try {
    const uint64_t modelBytes = packedModelFileBytes(config.modelRoot);
    const uint64_t hardBudgetBytes = EngineMemoryPolicy::hardBudgetBytes(
        device.recommendedMaxWorkingSetBytes, config.maximumMemoryBytes);
    // Reject an impossible weight budget before registering model buffers.
    // The full plan below still uses measured allocations and runtime costs.
    if (modelBytes > hardBudgetBytes) {
      throw RuntimeResourcesError(
          RuntimeResourceStage::MemoryPlanning,
          "model weights require " + std::to_string(modelBytes) +
              " bytes but the Metal memory budget is " +
              std::to_string(hardBudgetBytes) + " bytes",
          deviceStatusJson(device), {}, RuntimeResourceFailure::EngineCapacity);
    }
    // Fail before opening the package when the machine has no headroom at
    // all; the guard installed above keeps checking as residency grows.
    requireStartupHeadroom(hostAvailableMemory, hostReserveBytes,
        config.memoryPressure ? config.memoryPressure() : MemoryPressure::Normal);
  } catch (const RuntimeResourcesError &) {
    throw;
  } catch (const metal::MetalAllocationError &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                error.what(), deviceStatusJson(device), {},
                                resourceAllocationFailure(error.failure()));
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                error.what(), deviceStatusJson(device));
  }

  model::ModelPackage package;
  try {
    package = model::loadModelPackage(*backend, config.modelRoot, config.model);
    requireLoadedModel(package);
  } catch (const metal::MetalAllocationError &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                error.what(), deviceStatusJson(device), {},
                                resourceAllocationFailure(error.failure()));
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                error.what(), deviceStatusJson(device));
  }

  // One selection owner is used both before allocation and during encoding.
  // The engine lends it to model execution without inspecting kernel choices.
  ops::ExecutionPlans operators(device);
  model::ModelMemoryPlan modelMemoryPlan;
  auto prepareMemory = [&]() -> EngineMemoryPlan {
    try {
      modelMemoryPlan = model::plannedRuntimeMemory(device, package, operators, config.kvFormat);
      if (auto error = modelMemoryPlan.validationError()) {
        throw std::invalid_argument(*error);
      }
    } catch (const std::exception &error) {
      throw RuntimeResourcesError(
          RuntimeResourceStage::MemoryPlanning,
          std::string("model allocated-size plan is invalid: ") + error.what(),
          deviceStatusJson(device));
    }

    ModelMemoryFootprint footprint{
        package.targetActualAllocatedBytes(),
        package.draft.actualAllocatedBytes,
        package.vision.actualAllocatedBytes,
        modelMemoryPlan.activeStateCellPlannedAllocatedBytes,
        modelMemoryPlan.sharedPrefillPlannedAllocatedBytes,
        modelMemoryPlan.sharedDecodePlannedAllocatedBytes,
        modelMemoryPlan.pipelineReserveBytes,
        modelMemoryPlan.runtimeOverheadReserveBytes,
    };

    ModelMemoryProfile modelProfile{
        package.name(), package.maximumContextTokens(),
        package.targetKvLayout(config.kvFormat), footprint};
    EngineMemoryPlanResult planResult =
        evaluateEngineMemoryPlan(device, modelProfile, config.maximumMemoryBytes);
    if (!planResult.plan) {
      throw RuntimeResourcesError(
          RuntimeResourceStage::MemoryPlanning, planResult.status.message,
          planResult.status.toStatusJson(), planResult.status.describe());
    }
    EngineMemoryPlan memoryPlan = std::move(*planResult.plan);
    return memoryPlan;
  };
  // Establish the serving baseline and the one real memory governor before
  // installing the shipped choices. Selected workspace never gets a separate
  // allowance or replaces the immutable engine-wide ceiling.
  EngineMemoryPlan memoryPlan = prepareMemory();

  RuntimeCacheIdentity cacheIdentity;
  try {
    cacheIdentity = makeRuntimeCacheIdentity(
        package.manifestFingerprintSha256,
        package.targetManifestFingerprint(), config.buildId,
        package.targetKvLayout(config.kvFormat),
        ops::q8Split2M32Sg16Profile(device));
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::ModelLoading,
                                error.what(), memoryPlan.toStatusJson(),
                                memoryPlan.breakdown().describe());
  }

  try {
    const auto baselineMemoryPlan = memoryPlan;
    const auto baselineModelMemoryPlan = modelMemoryPlan;
    const EngineMemoryBreakdown &baselineBudget = baselineMemoryPlan.breakdown();
    const uint64_t runtimeReserve =
        baselineBudget.pipelineReserveBytes + baselineBudget.runtimeOverheadReserveBytes;
    if (baselineBudget.hardBudgetBytes <= runtimeReserve) {
      throw std::logic_error("runtime reserves consume the Metal budget");
    }
    // The governor observes the complete Metal footprint. Keeping explicit
    // pipeline/allocator reserves outside its growth ceiling prevents elastic
    // state and KV from silently consuming the startup safety margin.
    const uint64_t elasticGrowthCeiling =
        baselineBudget.hardBudgetBytes - runtimeReserve;
    auto memoryGovernor = std::make_unique<MemoryGovernor>(
        *backend, elasticGrowthCeiling, hostReserveBytes, hostAvailableMemory);
    if (config.memoryPressure)
      memoryGovernor->setPressure(config.memoryPressure());
    std::string rejected;
    auto adoptChoices = [&](const ops::OperatorChoices &choices) {
      try {
        operators.install(choices);
        auto selectedMemoryPlan = prepareMemory();
        const auto &selected = selectedMemoryPlan.breakdown();
        if (selected.pipelineReserveBytes + selected.runtimeOverheadReserveBytes !=
                runtimeReserve || selected.hardBudgetBytes != baselineBudget.hardBudgetBytes)
          throw std::logic_error("operator choices changed the memory governor ceiling");
        memoryPlan = std::move(selectedMemoryPlan);
        return true;
      } catch (const std::exception &error) {
        // An illegal table entry or a host/user limit that no longer fits the
        // selected scratch keeps the operator defaults, never a partial table.
        rejected = error.what();
        operators.install({});
        modelMemoryPlan = baselineModelMemoryPlan;
        memoryPlan = baselineMemoryPlan;
        return false;
      }
    };
    logKernelStartup("Kernel policy for GPU family ", device.appleGpuFamily,
                     " with ", device.gpuCoreCount, " cores.");
    if (config.operatorChoices && !config.operatorChoices->empty()) {
      if (adoptChoices(*config.operatorChoices))
        logKernelStartup("Installed supplied kernel choices.");
      else
        logKernelStartup("Supplied kernel choices rejected (", rejected,
                         "); using the kernel policy.");
    }

    const EngineMemoryBreakdown &budget = memoryPlan.breakdown();
    auto kvPages = std::make_unique<kv::PageStorage>(
        *backend, memoryGovernor->allocationAdmission(), package.targetKvLayout(config.kvFormat),
        budget.kvVirtualPages);
    auto stateStorage = model::createStateStorage(
        *backend, memoryGovernor->allocationAdmission(), package);
    if (!stateStorage) {
      throw std::runtime_error("model factory returned no state storage");
    }
    auto kvPool = std::make_unique<KvPool>(*kvPages);
    auto cache =
        std::make_unique<engine::Cache>(*kvPool, cacheIdentity.cacheNamespace);

    if (kvPages->declaredBytes() != budget.kvVirtualBytes ||
        kvPages->actualAllocatedBytes() > budget.kvVirtualBytes) {
      throw std::runtime_error(
          "actual KV page storage exceeds its planned category");
    }
    if (stateStorage->actualAllocatedBytes() != 0) {
      throw std::runtime_error("state cells were allocated eagerly");
    }
    metal::MetalMemoryStats memory = backend->memoryStats();
    if (!backend->healthy()) {
      throw std::runtime_error(
          "Metal backend became unhealthy during resource allocation: " +
          backend->unhealthyReason());
    }
    if (memory.allocatedBytes > budget.hardBudgetBytes ||
        memory.deviceCurrentAllocatedBytes > budget.hardBudgetBytes) {
      throw std::runtime_error(
          "base Metal allocation exceeds immutable hard budget");
    }

    auto result = std::unique_ptr<RuntimeResources>(new RuntimeResources(
        std::move(backend), std::move(package), std::move(operators),
        std::move(memoryPlan),
        std::move(modelMemoryPlan), std::move(cacheIdentity),
        std::move(memoryGovernor), std::move(kvPages), std::move(stateStorage),
        std::move(kvPool), std::move(cache), config.maximumImagePatches));
    return result;
  } catch (const metal::MetalAllocationError &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::StorageAllocation,
                                error.what(), memoryPlan.toStatusJson(),
                                memoryPlan.breakdown().describe(),
                                resourceAllocationFailure(error.failure()));
  } catch (const std::exception &error) {
    throw RuntimeResourcesError(RuntimeResourceStage::StorageAllocation,
                                error.what(), memoryPlan.toStatusJson(),
                                memoryPlan.breakdown().describe());
  }
}

model::RuntimeContext RuntimeResources::modelContext() noexcept {
  const EngineMemoryBreakdown &budget = memoryPlan_.breakdown();
  return {
      *backend_,
      memoryGovernor_->allocationAdmission(),
      model_,
      *kvPages_,
      *stateStorage_,
      operators_,
      maximumImagePatches_,
      budget.pipelineReserveBytes,
      budget.runtimeOverheadReserveBytes,
  };
}

std::shared_ptr<StorageCoordinator>
RuntimeResources::createStorageCoordinator(
    const RuntimeResourcesConfig &config) {
  if (!config.ssdCacheRoot)
    throw std::invalid_argument("SSD cache root is required");
  return std::make_shared<QwenRestoreCoordinator>(*this, config);
}

ActualMemoryReport RuntimeResources::actualMemoryReport(
    const model::ModelMemoryActual &modelMemory,
    uint64_t estimatedWarmupPeakBytes) const {
  ActualMemoryReport report;
  report.targetWeightsBytes = model_.targetActualAllocatedBytes();
  report.draftWeightsBytes = model_.draft.actualAllocatedBytes;
  report.visionWeightsBytes = model_.vision.actualAllocatedBytes;
  report.stateResidentBytes = modelMemory.stateActualAllocatedBytes;
  report.sharedPrefillBytes = modelMemory.sharedPrefillActualAllocatedBytes;
  report.sharedDecodeBytes = modelMemory.sharedDecodeActualAllocatedBytes;
  report.kvResidentBytes = kvPages_->actualAllocatedBytes();
  // Optional warmup may end with a rolled-back allocation and no subsequent
  // command. Refresh current residency after that rollback; peaks stay intact.
  metal::MetalMemoryStats memory = backend_->refreshMemoryStats();
  if (memory.sparseResidentBytes >
      std::numeric_limits<uint64_t>::max() - memory.allocatedBytes) {
    throw std::overflow_error("backend memory accounting overflows");
  }
  report.backendAllocatedBytes =
      memory.allocatedBytes + memory.sparseResidentBytes;
  report.deviceCurrentAllocatedBytes = memory.deviceCurrentAllocatedBytes;
  report.devicePeakAllocatedBytes = memory.devicePeakAllocatedBytes;
  // A capacity-limited warmup can roll back a partial allocation before it
  // returns a result. Preserve that tracked high-water mark independently of
  // the device-wide measurement used by the audit.
  const auto &budget = memoryPlan_.breakdown();
  const uint64_t reserves =
      budget.pipelineReserveBytes + budget.runtimeOverheadReserveBytes;
  if (memory.peakResidentBytes >
      std::numeric_limits<uint64_t>::max() - reserves) {
    throw std::overflow_error("warmup memory estimate overflows");
  }
  report.estimatedWarmupPeakBytes =
      std::max(estimatedWarmupPeakBytes, memory.peakResidentBytes + reserves);
  return report;
}

} // namespace splash::engine
