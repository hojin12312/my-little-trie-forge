#pragma once

#include "metal/abi/ExecutionGeometry.h"

// Batched decode kernels take one buffer argument per lane (Metal has no array
// of buffers as a kernel argument). SPLASH_LANE_ARGS(type, prefix, base) expands
// to `type prefix0 [[buffer(base)]], ..., prefixN [[buffer(base + N)]]` for the
// compiled lane ceiling, SPLASH_LANE_PICK selects one by lane, and
// SPLASH_LANE_COUNT is the number of bindings a following argument must skip.
// At width 4 the expansion is exactly the historical four-lane signature.
#define SPLASH_LANE_COUNT SPLASH_MAXIMUM_BATCH_WIDTH

#define SPLASH_LANE_ARG_(type, prefix, base, lane) \
  type prefix##lane [[buffer((base) + lane)]]

#if SPLASH_MAXIMUM_BATCH_WIDTH == 4
#define SPLASH_LANE_ARGS(type, prefix, base)                                   \
  SPLASH_LANE_ARG_(type, prefix, base, 0), SPLASH_LANE_ARG_(type, prefix, base, 1), \
  SPLASH_LANE_ARG_(type, prefix, base, 2), SPLASH_LANE_ARG_(type, prefix, base, 3)
#define SPLASH_LANE_PICK(prefix, lane)                                         \
  ((lane) == 0 ? prefix##0 : ((lane) == 1 ? prefix##1 :                        \
   ((lane) == 2 ? prefix##2 : prefix##3)))
#elif SPLASH_MAXIMUM_BATCH_WIDTH == 8
#define SPLASH_LANE_ARGS(type, prefix, base)                                   \
  SPLASH_LANE_ARG_(type, prefix, base, 0), SPLASH_LANE_ARG_(type, prefix, base, 1), \
  SPLASH_LANE_ARG_(type, prefix, base, 2), SPLASH_LANE_ARG_(type, prefix, base, 3), \
  SPLASH_LANE_ARG_(type, prefix, base, 4), SPLASH_LANE_ARG_(type, prefix, base, 5), \
  SPLASH_LANE_ARG_(type, prefix, base, 6), SPLASH_LANE_ARG_(type, prefix, base, 7)
#define SPLASH_LANE_PICK(prefix, lane)                                         \
  ((lane) < 4 ?                                                                \
   ((lane) == 0 ? prefix##0 : ((lane) == 1 ? prefix##1 :                       \
    ((lane) == 2 ? prefix##2 : prefix##3))) :                                  \
   ((lane) == 4 ? prefix##4 : ((lane) == 5 ? prefix##5 :                       \
    ((lane) == 6 ? prefix##6 : prefix##7))))
#else
#error "unsupported SPLASH_MAXIMUM_BATCH_WIDTH: lane bindings exist for 4 and 8"
#endif
