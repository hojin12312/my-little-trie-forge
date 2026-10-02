#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace splash::engine {

// A sparse checkpoint refers to immutable KV objects and one CompositeState
// object. Exact tokens are checked in the referenced Page32 objects on load.
struct RestoreBundle final {
  std::array<uint8_t, 32> namespaceDigest{};
  uint32_t boundary = 0;
  std::string stateKey;
  std::vector<std::string> kvKeys;
};

[[nodiscard]] uint64_t restoreBundleBytes(const RestoreBundle &bundle);
[[nodiscard]] std::string restoreBundleKey(std::span<const uint8_t> bytes);
void encodeRestoreBundle(const RestoreBundle &bundle,
                         std::span<const uint32_t> exactPrefix,
                         std::span<uint8_t> bytes);
[[nodiscard]] RestoreBundle decodeRestoreBundle(
    std::span<const uint8_t> bytes, const std::string &expectedKey,
    const std::array<uint8_t, 32> &expectedNamespace,
    std::span<const uint32_t> queryTokens);
// Structural/reference inspection for owner-only GC. Does not authorize a
// restore: the caller still needs decodeRestoreBundle's exact token check.
[[nodiscard]] RestoreBundle inspectRestoreBundle(
    std::span<const uint8_t> bytes, const std::string &expectedKey,
    const std::array<uint8_t, 32> &expectedNamespace);

struct BundleOrphans final {
  bool state = false;
  std::vector<std::string> kvKeys;
};
// Exact last-manifest ownership. Extents repeated for many Page32 slots are
// returned once; surviving ancestor/branch references remain protected.
[[nodiscard]] BundleOrphans bundleOrphans(
    const RestoreBundle &victim, std::span<const RestoreBundle> survivors);

} // namespace splash::engine
