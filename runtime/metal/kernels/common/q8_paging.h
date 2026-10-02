#pragma once

#include "metal/abi/ExecutionGeometry.h"
#include <metal_stdlib>

using namespace metal;

// Q8 KV page geometry shared by the attention and store kernels. A page holds
// SplashQ8PageTokens tokens of every KV head: keys token-major, values
// dimension-major, and one fp32 scale per (KV head, token) for each tensor.
constant uint SplashQ8PageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS;
constant uint SplashQ8HeadDimension = 256;

template <uint KVHeads>
inline ulong splash_q8_key_index(uint page, uint head, uint token,
                                   uint dimension) {
  constexpr ulong ElementsPerPage =
      ulong(KVHeads) * SplashQ8PageTokens * SplashQ8HeadDimension;
  return ulong(page) * ElementsPerPage +
         (ulong(head) * SplashQ8PageTokens + token) *
             SplashQ8HeadDimension +
         dimension;
}

template <uint KVHeads>
inline ulong splash_q8_value_index(uint page, uint head, uint token,
                                     uint dimension) {
  constexpr ulong ElementsPerPage =
      ulong(KVHeads) * SplashQ8PageTokens * SplashQ8HeadDimension;
  return ulong(page) * ElementsPerPage +
         (ulong(head) * SplashQ8HeadDimension + dimension) *
             SplashQ8PageTokens +
         token;
}

template <uint KVHeads>
inline ulong splash_q8_scale_index(uint page, uint head, uint token) {
  constexpr ulong ScalesPerPage = ulong(KVHeads) * SplashQ8PageTokens;
  return ulong(page) * ScalesPerPage + ulong(head) * SplashQ8PageTokens +
         token;
}
