#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include "model/Model.hpp"
#include <vector>

namespace splash::engine {

struct StorageCompletion final {
  uint64_t requestId = 0;
  uint64_t generation = 0;
};

// Last successful restore, microseconds. Filesystem read is worker-side
// elapsed read(2) time and may be served by the OS page cache; owner wait is
// a residual timeline interval, not physical-media service time.
struct RestoreTimingSnapshot final {
  uint64_t totalUs = 0;
  uint64_t lookupUs = 0;
  uint64_t stateReadWallUs = 0;
  uint64_t kvReadWallUs = 0;
  uint64_t stateFileReadUs = 0;
  uint64_t kvFileReadUs = 0;
  uint64_t bufferPrepareUs = 0;
  uint64_t validationUs = 0;
  uint64_t stateDeserializeUs = 0;
  uint64_t gpuAdmissionUs = 0;
  uint64_t kvCpuInstallUs = 0;
  uint64_t kvGpuInstallUs = 0;
  uint64_t statePublishUs = 0;
  uint64_t ownerGapUs = 0;
};

struct StorageSnapshot final {
  uint64_t transferUsedBytes = 0;
  uint64_t transferPeakBytes = 0;
  uint64_t diskUsedBytes = 0;
  uint64_t diskPeakBytes = 0;
  uint64_t diskQuotaBytes = 0;
  uint64_t diskFreeFloorBytes = 0;
  uint32_t indexedBundles = 0;
  uint32_t pendingReads = 0;
  uint32_t pendingWrites = 0;
  uint64_t restores = 0;
  uint64_t walkbacks = 0;
  uint64_t safeMisses = 0;
  uint64_t publications = 0;
  uint64_t skippedPublications = 0;
  uint64_t gcCycles = 0;
  uint64_t gcObjects = 0;
  uint64_t gcBytes = 0;
  uint64_t gcBundles = 0;
  RestoreTimingSnapshot lastRestore;
  uint64_t decodeCaptureOpportunities = 0;
  uint64_t decodeSkipWriter = 0;
  uint64_t decodeSkipInvalid = 0;
  uint64_t decodeSkipPressure = 0;
  uint64_t decodeSkipTransfer = 0;
  uint64_t decodeSkipSnapshot = 0;
  uint64_t decodeSkipExisting = 0;
};

// A storage worker returns immutable completions to the native owner. poll()
// may install at most one ready bounded chunk when no model command is in
// flight; it never mutates Engine slots or the scheduler from a worker thread.
class StorageCoordinator {
public:
  virtual ~StorageCoordinator() = default;
  [[nodiscard]] virtual bool beginLookup(
      uint64_t requestId, uint64_t generation,
      std::span<const uint32_t> prompt,
      std::span<const ImageSpan> images) = 0;
  virtual void beginPublish(
      uint64_t kvBlock, uint32_t boundary,
      std::span<const uint32_t> exactTokens,
      std::span<const ImageSpan> images,
      std::shared_ptr<const CompositeState> state) = 0;
  // Decode captures only after storage has reserved its bounded state transfer.
  // The factory runs on the owner thread and may return null under pressure.
  [[nodiscard]] virtual bool beginPublishFromCapture(
      uint64_t kvBlock, uint32_t boundary,
      std::span<const uint32_t> exactTokens,
      std::span<const ImageSpan> images,
      const std::function<std::shared_ptr<const CompositeState>()> &capture) = 0;
  [[nodiscard]] virtual bool captureReady() const noexcept = 0;
  [[nodiscard]] virtual std::vector<StorageCompletion> poll() = 0;
  // An owner-thread chunk/page may progress without completing a request.
  // A true result lets the transport tick again immediately instead of
  // inserting its storage-poll sleep between every installed KV page.
  [[nodiscard]] virtual bool pollProgressed() const noexcept { return false; }
  [[nodiscard]] virtual bool pending() const noexcept = 0;
  [[nodiscard]] virtual StorageSnapshot snapshot() const noexcept = 0;
  virtual void cancel(uint64_t requestId) noexcept = 0;
  virtual void setNotifier(std::function<void()> notifier) = 0;
};

} // namespace splash::engine
