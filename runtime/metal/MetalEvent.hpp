#pragma once

#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <algorithm>
#include <chrono>
#include <stop_token>

namespace splash::metal {

// Keep a pending resource dependency off the GPU command queue and the
// submitting thread. The callback owns everything needed until the wait ends.
template <class Completion>
inline void afterMetalEvent(id<MTLSharedEvent> event, uint64_t value,
                            NSUInteger timeoutMilliseconds,
                            Completion completion, std::stop_token stop = {}) {
  if (stop.stop_requested()) {
    completion(false);
    return;
  }
  if (!value || event.signaledValue >= value) {
    completion(true);
    return;
  }
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    @autoreleasepool {
      const auto deadline = std::chrono::steady_clock::now() +
          std::chrono::milliseconds(timeoutMilliseconds);
      bool signaled = false;
      while (!stop.stop_requested()) {
        const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        if (remaining <= 0) break;
        if ([event waitUntilSignaledValue:value
                               timeoutMS:std::min<int64_t>(remaining, 100)]) {
          signaled = true;
          break;
        }
      }
      completion(signaled && !stop.stop_requested());
    }
  });
}

} // namespace splash::metal
