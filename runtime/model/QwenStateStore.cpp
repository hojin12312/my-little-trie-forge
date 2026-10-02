#include "model/QwenStateStore.hpp"
#include "ops/KvPageCodec.hpp"

#include <cerrno>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <system_error>
#include <thread>
#include <unistd.h>

namespace splash::model {

struct CompositeStateStore::RootLease final {
  explicit RootLease(int directory) : directory(directory) {}
  ~RootLease() {
    if (lock >= 0) close(lock);
    if (directory >= 0) close(directory);
  }
  int directory = -1;
  int lock = -1;
  std::mutex publication;
};

struct CompositeStateStore::DiskBudget final {
  DiskBudget(uint64_t maximum, uint64_t freeFloor)
      : maximum(maximum), freeFloor(freeFloor) {}
  const uint64_t maximum;
  const uint64_t freeFloor;
  std::atomic<uint64_t> used{0};
  std::atomic<uint64_t> peak{0};
};

struct CompositeStateStore::DiskTicket final {
  DiskTicket(std::shared_ptr<DiskBudget> owner, uint64_t bytes)
      : owner(std::move(owner)), bytes(bytes) {}
  ~DiskTicket() {
    if (active && !committed)
      owner->used.fetch_sub(bytes);
  }
  std::shared_ptr<DiskBudget> owner;
  uint64_t bytes = 0;
  bool active = false;
  bool committed = false;
};

CompositeStateStore::CompositeStateStore(std::filesystem::path root,
                                         uint64_t transferLimit,
                                         uint64_t diskQuotaBytes,
                                         uint64_t freeFloorBytes,
                                         uint32_t workerDelayMilliseconds,
                                         std::function<std::shared_ptr<void>(uint64_t)>
                                             externalAdmission)
    : root_(std::move(root)), budget_(std::make_shared<Budget>(transferLimit)),
      diskBudget_(std::make_shared<DiskBudget>(diskQuotaBytes, freeFloorBytes)),
      workerDelayMilliseconds_(workerDelayMilliseconds),
      externalAdmission_(std::move(externalAdmission)) {
  if (!transferLimit || !diskQuotaBytes || root_.empty())
    throw std::invalid_argument("state store root and budget are required");
  if (std::filesystem::is_symlink(std::filesystem::symlink_status(root_)))
    throw std::invalid_argument("state store root must not be a symlink");
  const bool created = std::filesystem::create_directories(root_);
  if (created)
    std::filesystem::permissions(root_, std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::replace);
  const int directory = open(root_.c_str(), O_RDONLY | O_DIRECTORY |
                                             O_NOFOLLOW | O_CLOEXEC);
  if (directory < 0)
    throw std::runtime_error("state store root cannot be opened safely");
  rootLease_ = std::shared_ptr<RootLease>(new RootLease(directory));
  struct stat info{};
  if (fstat(directory, &info) != 0 || info.st_uid != geteuid() ||
      (info.st_mode & 0077) != 0)
    throw std::invalid_argument("state store root must be private and owned");
  constexpr const char *marker = ".splash-cache-owned-v1";
  if (created) {
    const int fd = openat(directory, marker, O_WRONLY | O_CREAT | O_EXCL |
                         O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) throw std::runtime_error("state store ownership marker failed");
    if (close(fd) != 0 || fsync(directory) != 0)
      throw std::runtime_error("state store ownership marker sync failed");
  }
  struct stat markerInfo{};
  gcOwned_ = fstatat(directory, marker, &markerInfo, AT_SYMLINK_NOFOLLOW) == 0 &&
      S_ISREG(markerInfo.st_mode) && markerInfo.st_uid == geteuid() &&
      markerInfo.st_size == 0 && (markerInfo.st_mode & 0077) == 0;
  rootLease_->lock = openat(directory, ".lock", O_RDWR | O_CREAT |
                           O_CLOEXEC | O_NOFOLLOW, 0600);
  if (rootLease_->lock < 0 ||
      flock(rootLease_->lock, LOCK_EX | LOCK_NB) != 0)
    throw std::runtime_error("state store root already has an owner");
  const int scanFd = openat(directory, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  DIR *scan = scanFd < 0 ? nullptr : fdopendir(scanFd);
  if (!scan) {
    if (scanFd >= 0) close(scanFd);
    throw std::runtime_error("state store disk usage cannot be scanned");
  }
  uint64_t existingBytes = 0;
  while (dirent *entry = readdir(scan)) {
    if (std::string_view(entry->d_name) == "." ||
        std::string_view(entry->d_name) == "..")
      continue;
    struct stat object{};
    if (fstatat(directory, entry->d_name, &object,
                AT_SYMLINK_NOFOLLOW) != 0 || !S_ISREG(object.st_mode) ||
        object.st_size < 0 ||
        static_cast<uint64_t>(object.st_size) >
            std::numeric_limits<uint64_t>::max() - existingBytes) {
      closedir(scan);
      throw std::runtime_error("state store contains an unsafe object");
    }
    existingBytes += static_cast<uint64_t>(object.st_size);
  }
  closedir(scan);
  if (existingBytes > diskQuotaBytes)
    throw std::runtime_error("state store already exceeds disk quota");
  diskBudget_->used.store(existingBytes);
  diskBudget_->peak.store(existingBytes);
}

uint64_t CompositeStateStore::diskUsed() const noexcept {
  return diskBudget_->used.load();
}

uint64_t CompositeStateStore::diskPeak() const noexcept {
  return diskBudget_->peak.load();
}

uint64_t CompositeStateStore::diskQuota() const noexcept {
  return diskBudget_->maximum;
}

uint64_t CompositeStateStore::freeFloor() const noexcept {
  return diskBudget_->freeFloor;
}

bool CompositeStateStore::reserveDisk(uint64_t bytes) noexcept {
  if (!bytes || bytes > diskBudget_->maximum)
    return false;
  struct statvfs filesystem{};
  if (fstatvfs(rootLease_->directory, &filesystem) != 0)
    return false;
  const uint64_t available =
      filesystem.f_frsize &&
              filesystem.f_bavail >
                  std::numeric_limits<uint64_t>::max() / filesystem.f_frsize
          ? std::numeric_limits<uint64_t>::max()
          : uint64_t{filesystem.f_bavail} * filesystem.f_frsize;
  if (available < diskBudget_->freeFloor ||
      available - diskBudget_->freeFloor < bytes)
    return false;
  uint64_t used = diskBudget_->used.load();
  while (used <= diskBudget_->maximum - bytes) {
    if (diskBudget_->used.compare_exchange_weak(used, used + bytes)) {
      uint64_t peak = diskBudget_->peak.load();
      while (peak < used + bytes &&
             !diskBudget_->peak.compare_exchange_weak(peak, used + bytes)) {}
      return true;
    }
  }
  return false;
}

bool CompositeStateStore::reserve(uint64_t bytes) noexcept {
  if (!bytes || bytes > budget_->maximum)
    return false;
  uint64_t used = budget_->used.load();
  while (used <= budget_->maximum - bytes) {
    if (budget_->used.compare_exchange_weak(used, used + bytes)) {
      uint64_t peak = budget_->peak.load();
      while (peak < used + bytes &&
             !budget_->peak.compare_exchange_weak(peak, used + bytes)) {}
      return true;
    }
  }
  return false;
}

std::filesystem::path
CompositeStateStore::objectPath(const std::string &key,
                                ObjectKind kind) const {
  if (key.empty() || key.size() > 128 ||
      key.find_first_not_of("0123456789abcdef") != std::string::npos)
    throw std::invalid_argument("state object key must be lowercase hex");
  const char *suffix = kind == ObjectKind::KvPage ? ".kv" :
                       kind == ObjectKind::KvExtent ? ".kve" :
                       kind == ObjectKind::Manifest ? ".bundle" : ".state";
  return root_ / (key + suffix);
}

std::vector<std::string> CompositeStateStore::listKeys(
    ObjectKind kind, size_t maximumKeys) const {
  if (!maximumKeys)
    throw std::invalid_argument("object index limit must be positive");
  const int descriptor = openat(rootLease_->directory, ".",
                                O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (descriptor < 0)
    throw std::runtime_error("object directory cannot be scanned");
  DIR *directory = fdopendir(descriptor);
  if (!directory) {
    close(descriptor);
    throw std::runtime_error("object directory cannot be scanned");
  }
  const std::string suffix =
      kind == ObjectKind::KvPage ? ".kv" :
      kind == ObjectKind::KvExtent ? ".kve" :
      kind == ObjectKind::Manifest ? ".bundle" : ".state";
  std::vector<std::string> keys;
  while (dirent *entry = readdir(directory)) {
    const std::string_view name(entry->d_name);
    if (name.size() != 64 + suffix.size() ||
        name.substr(64) != suffix ||
        name.substr(0, 64).find_first_not_of("0123456789abcdef") !=
            std::string_view::npos)
      continue;
    if (keys.size() == maximumKeys) {
      closedir(directory);
      throw std::runtime_error("object index limit exceeded");
    }
    keys.emplace_back(name.substr(0, 64));
  }
  closedir(directory);
  std::sort(keys.begin(), keys.end());
  return keys;
}

std::optional<uint64_t> CompositeStateStore::objectSize(
    const std::string &key, ObjectKind kind) const {
  const std::string name = objectPath(key, kind).filename().string();
  struct stat info{};
  if (fstatat(rootLease_->directory, name.c_str(), &info,
              AT_SYMLINK_NOFOLLOW) != 0 || !S_ISREG(info.st_mode) ||
      info.st_size < 0)
    return std::nullopt;
  return static_cast<uint64_t>(info.st_size);
}

std::optional<uint64_t> CompositeStateStore::removeUnreferencedObject(
    const std::string &key, ObjectKind kind) {
  if (!gcOwned_) return std::nullopt;
  const std::string name = objectPath(key, kind).filename().string();
  struct stat info{};
  if (fstatat(rootLease_->directory, name.c_str(), &info,
              AT_SYMLINK_NOFOLLOW) != 0 || !S_ISREG(info.st_mode) ||
      info.st_nlink != 1 || info.st_size < 0)
    return std::nullopt;
  if (unlinkat(rootLease_->directory, name.c_str(), 0) != 0)
    return std::nullopt;
  const uint64_t bytes = static_cast<uint64_t>(info.st_size);
  diskBudget_->used.fetch_sub(bytes);
  if (fsync(rootLease_->directory) != 0)
    throw std::runtime_error("state store GC directory sync failed");
  return bytes;
}

std::optional<CompositeStateStore::Transfer>
CompositeStateStore::reserveState(const QwenStateStorage &storage) {
  return reserveWriteBytes(storage.serializedBytes(),
                           storage.serializedBytes());
}

std::optional<CompositeStateStore::Transfer>
CompositeStateStore::reserveBytes(uint64_t reservationBytes,
                                  uint64_t objectBytes) {
  if (!objectBytes || objectBytes > reservationBytes ||
      !reserve(reservationBytes))
    return std::nullopt;
  Transfer transfer(reservationBytes, budget_);
  if (externalAdmission_ &&
      !(transfer.externalLease = externalAdmission_(reservationBytes)))
    return std::nullopt;
  transfer.bytes.resize(objectBytes);
  return transfer;
}

std::optional<CompositeStateStore::Transfer>
CompositeStateStore::reserveWriteBytes(uint64_t reservationBytes,
                                       uint64_t objectBytes) {
  if (!objectBytes || objectBytes > reservationBytes ||
      !reserve(reservationBytes))
    return std::nullopt;
  Transfer transfer(reservationBytes, budget_);
  // Own the disk reservation before charging or allocating the CPU transfer.
  // A quota refusal must not report a phantom CPU high-water mark.
  auto ticket = std::make_shared<DiskTicket>(diskBudget_, objectBytes);
  if (!reserveDisk(objectBytes))
    return std::nullopt;
  ticket->active = true;
  transfer.diskTicket = std::move(ticket);
  if (externalAdmission_ &&
      !(transfer.externalLease = externalAdmission_(reservationBytes)))
    return std::nullopt;
  transfer.bytes.resize(objectBytes);
  return transfer;
}

std::optional<CompositeStateStore::Write> CompositeStateStore::writeAsync(
    const QwenStateStorage &storage, const CompositeState &state,
    Transfer transfer, const std::string &key,
    std::shared_ptr<std::atomic_bool> cancel) {
  if (transfer.budget != budget_ ||
      transfer.reservedBytes != storage.serializedBytes() ||
      transfer.bytes.size() != transfer.reservedBytes)
    throw std::invalid_argument("state write lacks a complete transfer lease");
  storage.serialize(state, transfer.bytes);
  return writeBytesAsync(std::move(transfer), key,
                         ObjectKind::CompositeState, std::move(cancel));
}

std::optional<CompositeStateStore::Write> CompositeStateStore::writeBytesAsync(
    Transfer transfer, const std::string &key, ObjectKind kind,
    std::shared_ptr<std::atomic_bool> cancel) {
  const auto path = objectPath(key, kind);
  if (transfer.budget != budget_ || transfer.bytes.empty() ||
      transfer.bytes.size() > transfer.reservedBytes ||
      !transfer.diskTicket || transfer.diskTicket->owner != diskBudget_ ||
      transfer.diskTicket->bytes != transfer.bytes.size())
    throw std::invalid_argument("object write lacks complete transfer and disk leases");
  auto diskLease = std::move(transfer.diskTicket);
  const std::string name = path.filename().string();
  const std::string temp = name + ".tmp." + std::to_string(getpid()) + "." +
                           std::to_string(sequence_.fetch_add(1));
  if (!cancel) cancel = std::make_shared<std::atomic_bool>(false);
  auto completion = std::async(std::launch::async,
      [root = rootLease_, name, temp, key, cancel, diskLease,
       delay = workerDelayMilliseconds_,
       transfer = std::move(transfer)]() mutable {
        for (uint32_t elapsed = 0; elapsed < delay && !cancel->load();
             elapsed += 10)
          std::this_thread::sleep_for(std::chrono::milliseconds(
              std::min<uint32_t>(10, delay - elapsed)));
        if (cancel->load()) return false;
        const int fd = openat(root->directory, temp.c_str(),
                              O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC |
                                  O_NOFOLLOW, 0600);
        if (fd < 0)
          return false;
        bool ok = true;
        size_t offset = 0;
        while (offset < transfer.bytes.size() && !cancel->load()) {
          const ssize_t count = write(fd, transfer.bytes.data() + offset,
                                      transfer.bytes.size() - offset);
          if (count > 0) offset += static_cast<size_t>(count);
          else if (count < 0 && errno == EINTR) continue;
          else { ok = false; break; }
        }
        if (cancel->load() || offset != transfer.bytes.size()) ok = false;
        if (ok && fsync(fd) != 0) ok = false;
        if (close(fd) != 0) ok = false;
        // The temporary object remains private until every byte is synced.
        // Serialize same-process publishers of a content key. The root's
        // flock already excludes another store process.
        std::lock_guard publication(root->publication);
        if (ok && !cancel->load() &&
            linkat(root->directory, temp.c_str(), root->directory,
                   name.c_str(), 0) == 0) {
          diskLease->committed = true;
          unlinkat(root->directory, temp.c_str(), 0);
          return fsync(root->directory) == 0;
        }
        if (ok && !cancel->load() && errno == EEXIST) {
          // A shared immutable ancestor may already have this object key.
          // Only byte-identical publication can be deduplicated.
          const int existing = openat(root->directory, name.c_str(),
                                       O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
          if (existing >= 0) {
            struct stat info{};
            const bool regular = fstat(existing, &info) == 0 &&
                S_ISREG(info.st_mode) && info.st_size >= 0;
            const uint64_t previousBytes = regular ?
                static_cast<uint64_t>(info.st_size) : 0;
            const bool replaceable = regular && info.st_nlink == 1;
            bool same = regular && previousBytes == transfer.bytes.size();
            bool readFailed = false;
            std::array<uint8_t, 64 * 1024> chunk{};
            size_t compared = 0;
            while (same && compared < transfer.bytes.size()) {
              const size_t wanted = std::min(chunk.size(),
                                            transfer.bytes.size() - compared);
              const ssize_t count = read(existing, chunk.data(), wanted);
              if (count > 0) {
                same = std::memcmp(chunk.data(),
                    transfer.bytes.data() + compared,
                    static_cast<size_t>(count)) == 0;
                compared += static_cast<size_t>(count);
              } else if (count < 0 && errno == EINTR) {
                continue;
              } else {
                same = false;
                readFailed = true;
              }
            }
            close(existing);
            if (!same && !readFailed && replaceable &&
                kv::pageObjectKey(transfer.bytes) == key &&
                renameat(root->directory, temp.c_str(), root->directory,
                         name.c_str()) == 0) {
              // An incomplete or corrupt content-key object is replaced
              // atomically only with verified bytes of that same key. This
              // also repairs partial KV extents and manifests. Refund only
              // the observed old size: external same-process truncation may
              // leave conservative quota over-accounting until reopen.
              diskLease->committed = true;
              diskLease->owner->used.fetch_sub(previousBytes);
              return fsync(root->directory) == 0;
            }
            unlinkat(root->directory, temp.c_str(), 0);
            return same;
          }
        }
        unlinkat(root->directory, temp.c_str(), 0);
        return false;
      });
  return Write{std::move(completion), std::move(cancel)};
}

std::optional<std::future<std::optional<CompositeStateStore::Transfer>>>
CompositeStateStore::readAsync(uint64_t expectedBytes, const std::string &key,
                               std::shared_ptr<std::atomic_bool> cancel) {
  return readBytesAsync(expectedBytes, expectedBytes, key,
                        ObjectKind::CompositeState, std::move(cancel));
}

std::optional<std::future<std::optional<CompositeStateStore::Transfer>>>
CompositeStateStore::readBytesAsync(uint64_t expectedBytes,
                                    uint64_t reservationBytes,
                                    const std::string &key, ObjectKind kind,
                                    std::shared_ptr<std::atomic_bool> cancel) {
  const auto path = objectPath(key, kind);
  const std::string name = path.filename().string();
  if (!expectedBytes || expectedBytes > reservationBytes ||
      !reserve(reservationBytes))
    return std::nullopt;
  Transfer transfer(reservationBytes, budget_);
  if (externalAdmission_ &&
      !(transfer.externalLease = externalAdmission_(reservationBytes)))
    return std::nullopt;
  if (!cancel) cancel = std::make_shared<std::atomic_bool>(false);
  auto future = std::async(std::launch::async,
      [root = rootLease_, name, expectedBytes, cancel,
       delay = workerDelayMilliseconds_,
       transfer = std::move(transfer)]() mutable
          -> std::optional<Transfer> {
        const auto prepareBegan = std::chrono::steady_clock::now();
        transfer.bytes.resize(expectedBytes);
        transfer.bufferPrepareUs = std::chrono::duration_cast<
            std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                       prepareBegan).count();
        for (uint32_t elapsed = 0; elapsed < delay && !cancel->load();
             elapsed += 10)
          std::this_thread::sleep_for(std::chrono::milliseconds(
              std::min<uint32_t>(10, delay - elapsed)));
        if (cancel->load()) return std::nullopt;
        const int fd = openat(root->directory, name.c_str(),
                              O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0)
          return std::nullopt;
        const auto readBegan = std::chrono::steady_clock::now();
        bool ok = true;
        size_t offset = 0;
        while (offset < transfer.bytes.size() && !cancel->load()) {
          const ssize_t count = read(fd, transfer.bytes.data() + offset,
                                     transfer.bytes.size() - offset);
          if (count > 0) offset += static_cast<size_t>(count);
          else if (count < 0 && errno == EINTR) continue;
          else { ok = false; break; }
        }
        uint8_t extra;
        if (ok && !cancel->load() && read(fd, &extra, 1) != 0) ok = false;
        close(fd);
        transfer.filesystemReadUs = std::chrono::duration_cast<
            std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                       readBegan).count();
        if (!ok || cancel->load()) return std::nullopt;
        return std::move(transfer);
      });
  return future;
}

} // namespace splash::model
