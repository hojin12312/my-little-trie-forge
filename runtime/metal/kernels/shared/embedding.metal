#include "metal/abi/KernelABI.h"

template <uint Hidden>
inline void q4_embedding_impl(device const uint *tokens,
                              device const uchar *weights,
                              device const bfloat *scales,
                              device const bfloat *biases,
                              device bfloat *output,
                              constant Q4EmbeddingParams &params, uint index,
                              uint grid_size) {
  constexpr uint QuantGroups = Hidden / 64;
  uint elements = params.rows * Hidden;
  for (uint element = index; element < elements; element += grid_size) {
    uint row = element / Hidden;
    uint dim = element % Hidden;
    uint raw_token = tokens[row];
    // Runtime validates every token. Keep the bounds guard local to the
    // storage table so a malformed direct operator call cannot read past it.
    uint token = raw_token < params.vocabulary_size ? raw_token : 0;
    uchar packed = weights[ulong(token) * (Hidden / 2) + dim / 2];
    float quantized = float((packed >> ((dim & 1) * 4)) & 15);
    ulong parameter = ulong(token) * QuantGroups + dim / 64;
    output[element] =
        bfloat(quantized * float(scales[parameter]) + float(biases[parameter]));
  }
}

#define Q4_EMBEDDING_ENTRY(Name, Hidden)                                      \
  kernel void Name(                                                          \
      device const uint *tokens [[buffer(0)]],                               \
      device const uchar *weights [[buffer(1)]],                             \
      device const bfloat *scales [[buffer(2)]],                             \
      device const bfloat *biases [[buffer(3)]],                             \
      device bfloat *output [[buffer(4)]],                                   \
      constant Q4EmbeddingParams &params [[buffer(5)]],                      \
      uint index [[thread_position_in_grid]],                                \
      uint grid_size [[threads_per_grid]]) {                                 \
    q4_embedding_impl<Hidden>(tokens, weights, scales, biases, output,       \
                              params, index, grid_size);                     \
  }

Q4_EMBEDDING_ENTRY(embedding_q4_h5120, 5120)
Q4_EMBEDDING_ENTRY(embedding_q4_h2048, 2048)
#undef Q4_EMBEDDING_ENTRY

// Dense Q8 tables store signed int8 codes row-major with fp32 scales and
// fp32 folded biases (b' = z + 128*s), one parameter per 64-input group, or
// the compact bf16 s and z from which b' = fma(128, s, z) is rebuilt bit for
// bit. q * s is exact for an int8 code and a bf16 scale, so the row value
// rounds once whatever the compiler contracts.
inline float q8_embedding_scale(device const float *scales, ulong parameter) {
  return scales[parameter];
}
inline float q8_embedding_scale(device const bfloat *scales, ulong parameter) {
  return float(scales[parameter]);
}
inline float q8_embedding_bias(device const float *, device const float *biases,
                               ulong parameter) {
  return biases[parameter];
}
inline float q8_embedding_bias(device const bfloat *scales,
                               device const bfloat *zeros, ulong parameter) {
  return fma(128.0f, float(scales[parameter]), float(zeros[parameter]));
}

template <uint Hidden, typename P>
inline void q8_embedding_impl(device const uint *tokens,
                              device const char *weights,
                              device const P *scales,
                              device const P *biases,
                              device bfloat *output,
                              constant Q4EmbeddingParams &params, uint index,
                              uint grid_size) {
  constexpr uint QuantGroups = Hidden / 64;
  uint elements = params.rows * Hidden;
  for (uint element = index; element < elements; element += grid_size) {
    uint row = element / Hidden;
    uint dim = element % Hidden;
    uint raw_token = tokens[row];
    uint token = raw_token < params.vocabulary_size ? raw_token : 0;
    float quantized = float(weights[ulong(token) * Hidden + dim]);
    ulong parameter = ulong(token) * QuantGroups + dim / 64;
    output[element] =
        bfloat(quantized * q8_embedding_scale(scales, parameter) +
               q8_embedding_bias(scales, biases, parameter));
  }
}

#define Q8_EMBEDDING_ENTRY(Name, Hidden, Param)                               \
  kernel void Name(                                                          \
      device const uint *tokens [[buffer(0)]],                               \
      device const char *weights [[buffer(1)]],                              \
      device const Param *scales [[buffer(2)]],                              \
      device const Param *biases [[buffer(3)]],                              \
      device bfloat *output [[buffer(4)]],                                   \
      constant Q4EmbeddingParams &params [[buffer(5)]],                      \
      uint index [[thread_position_in_grid]],                                \
      uint grid_size [[threads_per_grid]]) {                                 \
    q8_embedding_impl<Hidden>(tokens, weights, scales, biases, output,       \
                              params, index, grid_size);                     \
  }

Q8_EMBEDDING_ENTRY(embedding_q8_h5120, 5120, float)
Q8_EMBEDDING_ENTRY(embedding_q8_h5120_bf16p, 5120, bfloat)
#undef Q8_EMBEDDING_ENTRY
