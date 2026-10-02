#pragma once

#include "ops/PagedKv.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>

namespace splash::kv {

// Cache-only, lossless one-page envelope. The parent/content-derived object
// key is owned by the persistent prefix index; this header binds the page to
// its numerical namespace, layout and committed logical boundary.
struct PageObjectHeader final {
  char magic[8] = {'S', 'P', 'K', 'V', '0', '0', '0', '2'};
  uint32_t format = 0;
  uint32_t attentionLayers = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;
  uint32_t pageTokens = 0;
  uint32_t boundary = 0;
  uint32_t reserved0 = 0;
  uint32_t reserved1 = 0;
  uint64_t payloadBytes = 0;
  uint64_t payloadHash = 0;
  std::array<uint8_t, 32> namespaceDigest{};
  std::array<uint32_t, kPageTokens> exactTokens{};
  std::array<uint64_t, 2> mediaIdentity{};
};
static_assert(sizeof(PageObjectHeader) == 232);

[[nodiscard]] uint64_t pageObjectBytes(Layout layout) noexcept;
[[nodiscard]] std::string pageObjectKey(std::span<const uint8_t> objectBytes);
void encodePageObject(Layout layout, uint32_t boundary,
                      const std::array<uint8_t, 32> &namespaceDigest,
                      std::span<const uint32_t> exactTokens,
                      const metal::MetalBuffer &staging,
                      std::span<uint8_t> objectBytes,
                      std::array<uint64_t, 2> mediaIdentity = {});
void decodePageObject(Layout layout, uint32_t boundary,
                      const std::array<uint8_t, 32> &namespaceDigest,
                      std::span<const uint32_t> exactTokens,
                      std::span<const uint8_t> objectBytes,
                      const metal::MetalBuffer &staging,
                      std::array<uint64_t, 2> mediaIdentity = {});
void validatePageObject(Layout layout, uint32_t boundary,
                        const std::array<uint8_t, 32> &namespaceDigest,
                        std::span<const uint32_t> exactTokens,
                        std::span<const uint8_t> objectBytes,
                        std::array<uint64_t, 2> mediaIdentity = {});

} // namespace splash::kv
