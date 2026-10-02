#pragma once

#include <cstdint>
#include <limits>

namespace splash {

// On overflow, the result is left untouched.
[[nodiscard]] inline bool checkedAdd(uint64_t left, uint64_t right,
                                     uint64_t &result) {
  if (left > std::numeric_limits<uint64_t>::max() - right)
    return false;
  result = left + right;
  return true;
}

[[nodiscard]] inline bool checkedMultiply(uint64_t left, uint64_t right,
                                          uint64_t &result) {
  if (left && right > std::numeric_limits<uint64_t>::max() / left)
    return false;
  result = left * right;
  return true;
}

} // namespace splash
