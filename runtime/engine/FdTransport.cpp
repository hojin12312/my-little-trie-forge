#include "engine/FdTransport.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cmath>
#include <fcntl.h>
#include <poll.h>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <utility>

namespace splash::engine {
namespace {

class RestoreFdFlags final {
public:
  RestoreFdFlags(int fd, int flags) : fd_(fd), flags_(flags) {}
  ~RestoreFdFlags() { static_cast<void>(fcntl(fd_, F_SETFL, flags_)); }

  RestoreFdFlags(const RestoreFdFlags &) = delete;
  RestoreFdFlags &operator=(const RestoreFdFlags &) = delete;

private:
  int fd_;
  int flags_;
};

[[noreturn]] void throwIo(const char *operation) {
  throw std::system_error(errno, std::generic_category(), operation);
}

// Retry pending control work while the engine is idle.
constexpr int kControlRetryMilliseconds = 25;

int pollTimeout(const NativeRuntime &loop, bool controlPending) {
  auto delay = loop.millisecondsUntilNextWakeup();
  int timeout = -1;
  if (delay) {
    if (*delay <= 0.0)
      timeout = 0;
    else if (*delay >= double(INT_MAX))
      timeout = INT_MAX;
    else
      timeout = static_cast<int>(std::ceil(*delay));
  }
  if (controlPending && !loop.commandInFlight() &&
      (timeout < 0 || timeout > kControlRetryMilliseconds)) {
    timeout = kControlRetryMilliseconds;
  }
  return timeout;
}

NativeProcessExit loopFailure(const NativeRuntime &loop) {
  return loop.engineHealthy() ? NativeProcessExit::ProtocolFailure
                              : NativeProcessExit::EngineFailure;
}

} // namespace

struct FdTransport::CompletionWake {
  int readFd = -1;
  int writeFd = -1;
  std::atomic<bool> controlPending{false};
  std::atomic<bool> shutdownRequested{false};

  CompletionWake() {
    int descriptors[2];
    if (pipe(descriptors) < 0)
      throwIo("pipe(completion wake)");
    readFd = descriptors[0];
    writeFd = descriptors[1];
    auto makeNonBlocking = [&](int fd) {
      int flags = fcntl(fd, F_GETFL);
      if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        int saved = errno;
        close(readFd);
        close(writeFd);
        readFd = -1;
        writeFd = -1;
        errno = saved;
        throwIo("fcntl(completion wake)");
      }
    };
    makeNonBlocking(readFd);
    makeNonBlocking(writeFd);
  }

  ~CompletionWake() {
    if (readFd >= 0)
      close(readFd);
    if (writeFd >= 0)
      close(writeFd);
  }

  void notify() const noexcept {
    constexpr uint8_t byte = 1;
    while (writeFd >= 0) {
      ssize_t written = write(writeFd, &byte, sizeof(byte));
      if (written > 0 ||
          (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))) {
        return;
      }
      if (written < 0 && errno == EINTR)
        continue;
      return;
    }
  }

  void notifyControl() noexcept {
    controlPending.store(true, std::memory_order_release);
    notify();
  }

  bool takeControl() noexcept {
    return controlPending.exchange(false, std::memory_order_acq_rel);
  }

  void drain() const noexcept {
    std::array<uint8_t, 64> bytes{};
    while (readFd >= 0) {
      ssize_t count = read(readFd, bytes.data(), bytes.size());
      if (count > 0)
        continue;
      if (count < 0 && errno == EINTR)
        continue;
      return;
    }
  }
};

FdTransport::FdTransport(int inputFd, int outputFd)
    : inputFd_(inputFd), outputFd_(outputFd),
      completionWake_(std::make_shared<CompletionWake>()) {
  if (inputFd_ < 0 || outputFd_ < 0) {
    throw std::invalid_argument("native transport requires valid fds");
  }
}

NativeRuntime::ByteSink FdTransport::outputSink() {
  return [this](std::span<const uint8_t> bytes) { writeAll(bytes); };
}

std::function<void()> FdTransport::controlNotifier() {
  std::shared_ptr<CompletionWake> wake = completionWake_;
  return [wake] { wake->notifyControl(); };
}

void FdTransport::setControlHandler(ControlHandler handler) {
  controlHandler_ = std::move(handler);
}

NativeProcessExit FdTransport::run(NativeRuntime &loop) {
  std::shared_ptr<CompletionWake> completionWake = completionWake_;
  loop.setCompletionNotifier([completionWake] { completionWake->notify(); });
  int originalFlags = fcntl(inputFd_, F_GETFL);
  if (originalFlags < 0)
    throwIo("fcntl(F_GETFL)");
  if (fcntl(inputFd_, F_SETFL, originalFlags | O_NONBLOCK) < 0) {
    throwIo("fcntl(F_SETFL)");
  }
  RestoreFdFlags restore(inputFd_, originalFlags);
  std::array<uint8_t, 64 * 1024> input{};
  bool deferredControl = false;

  while (!loop.connectionMustClose()) {
    if (shutdownRequested())
      return NativeProcessExit::CleanEof;
    while (true) {
      ssize_t count = read(inputFd_, input.data(), input.size());
      if (count > 0) {
        if (!loop.receive(std::span<const uint8_t>(
                input.data(), static_cast<size_t>(count)))) {
          return loopFailure(loop);
        }
        continue;
      }
      if (count == 0) {
        return loop.finishInput() ? NativeProcessExit::CleanEof
                                  : loopFailure(loop);
      }
      if (errno == EINTR)
        continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK)
        break;
      return NativeProcessExit::IoFailure;
    }

    deferredControl = completionWake->takeControl() || deferredControl;
    if (deferredControl && !loop.commandInFlight()) {
      deferredControl = loop.runControl(controlHandler_);
    }

    if (loop.tick())
      continue;
    if (loop.connectionMustClose())
      return loopFailure(loop);

    std::array<pollfd, 2> descriptors{
        pollfd{inputFd_, short(POLLIN | POLLHUP), 0},
        pollfd{completionWake->readFd, POLLIN, 0}};
    int result;
    do {
      result = poll(descriptors.data(), descriptors.size(),
                    pollTimeout(loop, deferredControl));
    } while (result < 0 && errno == EINTR && !shutdownRequested());
    if (result < 0 && errno == EINTR)
      return NativeProcessExit::CleanEof;
    if (result < 0)
      return NativeProcessExit::IoFailure;
    if (descriptors[1].revents & POLLIN)
      completionWake->drain();
    // A zero result is a deadline or resource-retry wake-up; tick() at the
    // top of the next iteration performs the transition.
  }
  return loopFailure(loop);
}

bool FdTransport::shutdownRequested() const noexcept {
  return completionWake_->shutdownRequested.load(std::memory_order_acquire);
}

void FdTransport::requestShutdown() noexcept {
  completionWake_->shutdownRequested.store(true, std::memory_order_release);
  completionWake_->notify();
}

void FdTransport::writeAll(std::span<const uint8_t> bytes) const {
  size_t offset = 0;
  while (offset < bytes.size()) {
    ssize_t count =
        write(outputFd_, bytes.data() + offset, bytes.size() - offset);
    if (count > 0) {
      offset += static_cast<size_t>(count);
      continue;
    }
    if (count < 0 && errno == EINTR)
      continue;
    if (count == 0) {
      throw std::runtime_error("native output accepted zero bytes");
    }
    throwIo("write(native output)");
  }
}

} // namespace splash::engine
