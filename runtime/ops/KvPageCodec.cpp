#include "ops/KvPageCodec.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace splash::kv {
namespace {

uint64_t hashPayload(std::span<const uint8_t> bytes) noexcept {
  uint64_t hash = 14695981039346656037ULL;
  for (uint8_t value : bytes) {
    hash ^= value;
    hash *= 1099511628211ULL;
  }
  return hash;
}

void validateShape(Layout layout, uint32_t boundary,
                   std::span<const uint32_t> exactTokens,
                   const metal::MetalBuffer &staging, size_t objectBytes) {
  if (!layout.valid() || !boundary || boundary % kPageTokens ||
      exactTokens.size() != kPageTokens || !staging ||
      staging.storage() != metal::BufferStorage::Shared ||
      staging.sizeBytes() < layout.bytesPerModelPage() ||
      objectBytes != pageObjectBytes(layout) || !staging.contents())
    throw std::invalid_argument("incompatible KV Page32 object shape");
}

} // namespace

uint64_t pageObjectBytes(Layout layout) noexcept {
  return sizeof(PageObjectHeader) + layout.bytesPerModelPage();
}

std::string pageObjectKey(std::span<const uint8_t> objectBytes) {
  if (objectBytes.empty() || objectBytes.size() > UINT32_MAX)
    throw std::invalid_argument("KV page object cannot be hashed");
  std::array<uint8_t, CC_SHA256_DIGEST_LENGTH> digest{};
  if (!CC_SHA256(objectBytes.data(), static_cast<CC_LONG>(objectBytes.size()),
                 digest.data()))
    throw std::runtime_error("KV page object SHA-256 failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string key;
  key.reserve(digest.size() * 2);
  for (uint8_t value : digest) {
    key.push_back(hex[value >> 4]);
    key.push_back(hex[value & 15]);
  }
  return key;
}

void encodePageObject(Layout layout, uint32_t boundary,
                      const std::array<uint8_t, 32> &namespaceDigest,
                      std::span<const uint32_t> exactTokens,
                      const metal::MetalBuffer &staging,
                      std::span<uint8_t> objectBytes,
                      std::array<uint64_t, 2> mediaIdentity) {
  validateShape(layout, boundary, exactTokens, staging, objectBytes.size());
  PageObjectHeader header{};
  header.format = static_cast<uint32_t>(layout.format);
  header.attentionLayers = layout.attentionLayers;
  header.kvHeads = layout.kvHeads;
  header.headDimension = layout.headDimension;
  header.pageTokens = kPageTokens;
  header.boundary = boundary;
  header.payloadBytes = layout.bytesPerModelPage();
  header.namespaceDigest = namespaceDigest;
  std::copy(exactTokens.begin(), exactTokens.end(),
            header.exactTokens.begin());
  header.mediaIdentity = mediaIdentity;
  auto payload = objectBytes.subspan(sizeof(header));
  std::memcpy(payload.data(), staging.contents(), payload.size());
  header.payloadHash = hashPayload(payload);
  std::memcpy(objectBytes.data(), &header, sizeof(header));
}

void decodePageObject(Layout layout, uint32_t boundary,
                      const std::array<uint8_t, 32> &namespaceDigest,
                      std::span<const uint32_t> exactTokens,
                      std::span<const uint8_t> objectBytes,
                      const metal::MetalBuffer &staging,
                      std::array<uint64_t, 2> mediaIdentity) {
  validateShape(layout, boundary, exactTokens, staging, objectBytes.size());
  validatePageObject(layout, boundary, namespaceDigest, exactTokens,
                     objectBytes, mediaIdentity);
  auto payload = objectBytes.subspan(sizeof(PageObjectHeader));
  std::memcpy(staging.contents(), payload.data(), payload.size());
}

void validatePageObject(Layout layout, uint32_t boundary,
                        const std::array<uint8_t, 32> &namespaceDigest,
                        std::span<const uint32_t> exactTokens,
                        std::span<const uint8_t> objectBytes,
                        std::array<uint64_t, 2> mediaIdentity) {
  if (!layout.valid() || !boundary || boundary % kPageTokens ||
      exactTokens.size() != kPageTokens ||
      objectBytes.size() != pageObjectBytes(layout))
    throw std::invalid_argument("incompatible KV Page32 object shape");
  PageObjectHeader header;
  std::memcpy(&header, objectBytes.data(), sizeof(header));
  constexpr char magic[8] = {'S', 'P', 'K', 'V', '0', '0', '0', '2'};
  auto payload = objectBytes.subspan(sizeof(header));
  if (std::memcmp(header.magic, magic, sizeof(magic)) != 0 ||
      header.format != static_cast<uint32_t>(layout.format) ||
      header.attentionLayers != layout.attentionLayers ||
      header.kvHeads != layout.kvHeads ||
      header.headDimension != layout.headDimension ||
      header.pageTokens != kPageTokens || header.boundary != boundary ||
      header.reserved0 || header.reserved1 ||
      header.payloadBytes != layout.bytesPerModelPage() ||
      header.namespaceDigest != namespaceDigest ||
      !std::equal(header.exactTokens.begin(), header.exactTokens.end(),
                  exactTokens.begin()) ||
      header.mediaIdentity != mediaIdentity ||
      header.payloadHash != hashPayload(payload))
    throw std::invalid_argument(
        "KV Page32 object identity or checksum mismatch");
}

} // namespace splash::kv
