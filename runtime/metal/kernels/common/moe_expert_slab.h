#pragma once

#include "metal/abi/KernelABI.h"

// One expert's Q4 slab, [weights][BF16 scales][BF16 biases] in the same
// StorageN=256 affine package as the dense kernels. Routed experts sit at
// their stride in the packed buffer; expert `experts` is the shared expert,
// whose single slab lives in its own buffer.
struct MoeQ4Slab {
  device uchar *weights;
  device bfloat *scales;
  device bfloat *biases;
};

inline MoeQ4Slab moe_q4_slab(device uchar *packed, device uchar *shared,
                             uint expert, uint experts,
                             ulong expert_stride_bytes, uint output_size,
                             uint input_size) {
  device uchar *base = expert == experts
                           ? shared
                           : packed + ulong(expert) * expert_stride_bytes;
  ulong elements = ulong(output_size) * input_size;
  ulong weight_bytes = elements / 2;
  ulong parameter_bytes = elements / 32;
  return {base, reinterpret_cast<device bfloat *>(base + weight_bytes),
          reinterpret_cast<device bfloat *>(base + weight_bytes +
                                            parameter_bytes)};
}
