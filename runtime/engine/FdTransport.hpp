#pragma once

#include "engine/NativeRuntime.hpp"

#include <cstdint>
#include <functional>
#include <memory>

namespace splash::engine {

enum class NativeProcessExit : int {
  CleanEof = 0,
  ProtocolFailure = 64,
  EngineFailure = 70,
  IoFailure = 74,
};

// POSIX pipe transport for the native runtime child process.  The input fd is
// nonblocking only while run() is active.  Each iteration drains all already
// available frames before advancing the backend. Constraint-mask responses
// can therefore arrive while the scheduler-owned target-forward command is
// in flight; cancellation remains safe because provisional writes commit only
// after that command drains.
class FdTransport final {
public:
  explicit FdTransport(int inputFd, int outputFd);
  FdTransport(const FdTransport &) = delete;
  FdTransport &operator=(const FdTransport &) = delete;

  [[nodiscard]] NativeRuntime::ByteSink outputSink();
  // Runs between commands after a control notification. Returning true asks
  // for another run at the next command-free point, bounded by a short poll
  // timeout, so paced work (one KV extent release at a time) can continue
  // without a new notification.
  using ControlHandler = std::function<bool()>;
  [[nodiscard]] std::function<void()> controlNotifier();
  void setControlHandler(ControlHandler handler);
  [[nodiscard]] NativeProcessExit run(NativeRuntime &loop);
  // Async-signal-safe. Asks run() to return CleanEof at its next iteration.
  // It does not wait for in-flight GPU work; the process owner bounds teardown.
  void requestShutdown() noexcept;
  [[nodiscard]] bool shutdownRequested() const noexcept;

private:
  struct CompletionWake;
  void writeAll(std::span<const uint8_t> bytes) const;

  int inputFd_ = -1;
  int outputFd_ = -1;
  std::shared_ptr<CompletionWake> completionWake_;
  ControlHandler controlHandler_;
};

} // namespace splash::engine
