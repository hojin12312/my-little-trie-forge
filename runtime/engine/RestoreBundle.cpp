#include "engine/RestoreBundle.hpp"

#include "ops/KvPageCodec.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace splash::engine {
namespace {

struct BundleHeader final {
  char magic[8] = {'S', 'P', 'R', 'B', '0', '0', '0', '1'};
  uint32_t boundary = 0;
  uint32_t kvCount = 0;
  std::array<uint8_t, 32> namespaceDigest{};
  std::array<uint8_t, 32> prefixHash{};
  char stateKey[64]{};
};
static_assert(sizeof(BundleHeader) == 144);

bool hexKey(std::string_view key) noexcept {
  return key.size() == 64 &&
      key.find_first_not_of("0123456789abcdef") == std::string_view::npos;
}

std::array<uint8_t, 32> hashPrefix(std::span<const uint32_t> tokens) {
  if (tokens.size() > std::numeric_limits<CC_LONG>::max() / sizeof(uint32_t))
    throw std::invalid_argument("prefix token hash is too large");
  std::array<uint8_t, 32> digest{};
  if (!CC_SHA256(tokens.data(),
                 static_cast<CC_LONG>(tokens.size() * sizeof(uint32_t)),
                 digest.data()))
    throw std::runtime_error("prefix token SHA-256 failed");
  return digest;
}

void validateBundle(const RestoreBundle &bundle) {
  if (!bundle.boundary || bundle.boundary > 262144 ||
      bundle.boundary % kv::kPageTokens ||
      bundle.kvKeys.size() != bundle.boundary / kv::kPageTokens ||
      !hexKey(bundle.stateKey) ||
      !std::all_of(bundle.kvKeys.begin(), bundle.kvKeys.end(), hexKey))
    throw std::invalid_argument("restore bundle shape or object key is invalid");
}

} // namespace

uint64_t restoreBundleBytes(const RestoreBundle &bundle) {
  validateBundle(bundle);
  return sizeof(BundleHeader) + bundle.kvKeys.size() * 64;
}

std::string restoreBundleKey(std::span<const uint8_t> bytes) {
  return kv::pageObjectKey(bytes);
}

void encodeRestoreBundle(const RestoreBundle &bundle,
                         std::span<const uint32_t> exactPrefix,
                         std::span<uint8_t> bytes) {
  validateBundle(bundle);
  if (exactPrefix.size() != bundle.boundary ||
      bytes.size() != restoreBundleBytes(bundle))
    throw std::invalid_argument("restore bundle token or byte length mismatch");
  BundleHeader header{};
  header.boundary = bundle.boundary;
  header.kvCount = static_cast<uint32_t>(bundle.kvKeys.size());
  header.namespaceDigest = bundle.namespaceDigest;
  header.prefixHash = hashPrefix(exactPrefix);
  std::memcpy(header.stateKey, bundle.stateKey.data(), 64);
  std::memcpy(bytes.data(), &header, sizeof(header));
  uint8_t *cursor = bytes.data() + sizeof(header);
  for (const std::string &key : bundle.kvKeys) {
    std::memcpy(cursor, key.data(), 64);
    cursor += 64;
  }
}

RestoreBundle decodeRestoreBundle(
    std::span<const uint8_t> bytes, const std::string &expectedKey,
    const std::array<uint8_t, 32> &expectedNamespace,
    std::span<const uint32_t> queryTokens) {
  RestoreBundle result = inspectRestoreBundle(bytes, expectedKey,
                                               expectedNamespace);
  BundleHeader header;
  std::memcpy(&header, bytes.data(), sizeof(header));
  if (queryTokens.size() < result.boundary ||
      header.prefixHash != hashPrefix(queryTokens.first(result.boundary)))
    throw std::invalid_argument("restore bundle prefix mismatch");
  return result;
}

RestoreBundle inspectRestoreBundle(
    std::span<const uint8_t> bytes, const std::string &expectedKey,
    const std::array<uint8_t, 32> &expectedNamespace) {
  if (!hexKey(expectedKey) || bytes.size() < sizeof(BundleHeader) ||
      restoreBundleKey(bytes) != expectedKey)
    throw std::invalid_argument("restore bundle object key mismatch");
  BundleHeader header;
  std::memcpy(&header, bytes.data(), sizeof(header));
  constexpr char magic[8] = {'S', 'P', 'R', 'B', '0', '0', '0', '1'};
  if (std::memcmp(header.magic, magic, sizeof(magic)) != 0 ||
      header.namespaceDigest != expectedNamespace ||
      !header.boundary || header.boundary > 262144 ||
      header.boundary % kv::kPageTokens ||
      header.kvCount != header.boundary / kv::kPageTokens ||
      bytes.size() != sizeof(header) + uint64_t{header.kvCount} * 64)
    throw std::invalid_argument("restore bundle namespace or shape mismatch");
  RestoreBundle result;
  result.namespaceDigest = header.namespaceDigest;
  result.boundary = header.boundary;
  result.stateKey.assign(header.stateKey, 64);
  result.kvKeys.reserve(header.kvCount);
  const char *cursor = reinterpret_cast<const char *>(bytes.data() + sizeof(header));
  for (uint32_t index = 0; index < header.kvCount; ++index) {
    result.kvKeys.emplace_back(cursor, 64);
    cursor += 64;
  }
  validateBundle(result);
  return result;
}

BundleOrphans bundleOrphans(
    const RestoreBundle &victim, std::span<const RestoreBundle> survivors) {
  validateBundle(victim);
  std::unordered_set<std::string> liveStates, liveKv, emitted;
  for (const auto &bundle : survivors) {
    validateBundle(bundle);
    liveStates.insert(bundle.stateKey);
    liveKv.insert(bundle.kvKeys.begin(), bundle.kvKeys.end());
  }
  BundleOrphans result;
  result.state = !liveStates.contains(victim.stateKey);
  for (const auto &key : victim.kvKeys)
    if (!liveKv.contains(key) && emitted.insert(key).second)
      result.kvKeys.push_back(key);
  return result;
}

} // namespace splash::engine
