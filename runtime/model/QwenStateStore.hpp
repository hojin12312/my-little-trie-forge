#pragma once

#include "model/QwenState.hpp"

#include <atomic>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace splash::model {

// R2's single-object transfer lane. A transfer owns its byte reservation
// until the asynchronous I/O and, for reads, the owner-thread restore finish.
class CompositeStateStore final {
  struct RootLease;
  struct DiskBudget;
  struct DiskTicket;
public:
  enum class ObjectKind : uint8_t {
    CompositeState, KvPage, KvExtent, Manifest
  };
  struct Budget final {
    explicit Budget(uint64_t maximum) : maximum(maximum) {}
    const uint64_t maximum;
    std::atomic<uint64_t> used{0};
    std::atomic<uint64_t> peak{0};
  };

  struct Transfer final {
    std::vector<uint8_t> bytes;
    std::shared_ptr<void> externalLease;
    std::shared_ptr<Budget> budget;
    std::shared_ptr<DiskTicket> diskTicket;
    uint64_t reservedBytes = 0;
    uint64_t bufferPrepareUs = 0;
    uint64_t filesystemReadUs = 0;
    Transfer(uint64_t reservation, std::shared_ptr<Budget> owner)
        : budget(std::move(owner)), reservedBytes(reservation) {}
    Transfer(Transfer &&other) noexcept
        : bytes(std::move(other.bytes)),
          externalLease(std::move(other.externalLease)),
          budget(std::move(other.budget)),
          diskTicket(std::move(other.diskTicket)),
          reservedBytes(std::exchange(other.reservedBytes, 0)),
          bufferPrepareUs(other.bufferPrepareUs),
          filesystemReadUs(other.filesystemReadUs) {}
    Transfer &operator=(Transfer &&) = delete;
    Transfer(const Transfer &) = delete;
    ~Transfer() {
      std::vector<uint8_t>().swap(bytes);
      externalLease.reset();
      if (budget)
        budget->used.fetch_sub(reservedBytes);
    }
  };

  struct Write final {
    std::future<bool> completion;
    std::shared_ptr<std::atomic_bool> cancel;
  };

  CompositeStateStore(std::filesystem::path root, uint64_t transferLimit,
                      uint64_t diskQuotaBytes = uint64_t{1} << 30,
                      uint64_t freeFloorBytes = uint64_t{8} << 30,
                      uint32_t workerDelayMilliseconds = 0,
                      std::function<std::shared_ptr<void>(uint64_t)>
                          externalAdmission = {});
  [[nodiscard]] std::optional<Transfer>
  reserveState(const QwenStateStorage &storage);
  [[nodiscard]] std::optional<Transfer>
  reserveBytes(uint64_t reservationBytes, uint64_t objectBytes);
  [[nodiscard]] std::optional<Transfer>
  reserveWriteBytes(uint64_t reservationBytes, uint64_t objectBytes);
  [[nodiscard]] std::optional<Write>
  writeAsync(const QwenStateStorage &storage,
             const CompositeState &state, Transfer transfer,
             const std::string &key,
             std::shared_ptr<std::atomic_bool> cancel = {});
  [[nodiscard]] std::optional<Write>
  writeBytesAsync(Transfer transfer, const std::string &key, ObjectKind kind,
                  std::shared_ptr<std::atomic_bool> cancel = {});
  [[nodiscard]] std::optional<std::future<std::optional<Transfer>>>
  readAsync(uint64_t expectedBytes, const std::string &key,
            std::shared_ptr<std::atomic_bool> cancel = {});
  [[nodiscard]] std::optional<std::future<std::optional<Transfer>>>
  readBytesAsync(uint64_t expectedBytes, uint64_t reservationBytes,
                 const std::string &key, ObjectKind kind,
                 std::shared_ptr<std::atomic_bool> cancel = {});
  [[nodiscard]] std::vector<std::string> listKeys(ObjectKind kind,
                                                  size_t maximumKeys) const;
  [[nodiscard]] std::optional<uint64_t> objectSize(
      const std::string &key, ObjectKind kind) const;
  // Only roots created and marked by this store can reclaim cache objects.
  // Callers must prove no live manifest, request or writer owns the object.
  [[nodiscard]] bool gcOwned() const noexcept { return gcOwned_; }
  [[nodiscard]] std::optional<uint64_t> removeUnreferencedObject(
      const std::string &key, ObjectKind kind);
  [[nodiscard]] uint64_t transferUsed() const noexcept {
    return budget_->used.load();
  }
  [[nodiscard]] uint64_t transferPeak() const noexcept {
    return budget_->peak.load();
  }
  [[nodiscard]] uint64_t diskUsed() const noexcept;
  [[nodiscard]] uint64_t diskPeak() const noexcept;
  [[nodiscard]] uint64_t diskQuota() const noexcept;
  [[nodiscard]] uint64_t freeFloor() const noexcept;

private:
  [[nodiscard]] bool reserve(uint64_t bytes) noexcept;
  [[nodiscard]] bool reserveDisk(uint64_t bytes) noexcept;
  [[nodiscard]] std::filesystem::path objectPath(const std::string &key,
                                                  ObjectKind kind) const;
  std::filesystem::path root_;
  std::shared_ptr<RootLease> rootLease_;
  std::shared_ptr<Budget> budget_;
  std::shared_ptr<DiskBudget> diskBudget_;
  std::atomic<uint64_t> sequence_{0};
  uint32_t workerDelayMilliseconds_ = 0;
  bool gcOwned_ = false;
  std::function<std::shared_ptr<void>(uint64_t)> externalAdmission_;
};

} // namespace splash::model
