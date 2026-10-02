#pragma once

// C/Metal ABI constants shared by host and shader compilation. Startup
// static assertions check the corresponding model and operator contracts.
#define SPLASH_DRAFT_QUERY_ROWS 8u
#define SPLASH_DRAFT_PROPOSAL_TOKENS 7u
#define SPLASH_TARGET_VERIFY_ROWS 8u
#define SPLASH_MAXIMUM_CONTEXT_TOKENS 262144u
#define SPLASH_SPECULATIVE_SCRATCH_TOKENS                                  \
  (SPLASH_TARGET_VERIFY_ROWS - 1u)
#define SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS                                  \
  (SPLASH_MAXIMUM_CONTEXT_TOKENS + SPLASH_SPECULATIVE_SCRATCH_TOKENS)
// Development builds may raise this (host and shader compilation must agree)
// while native B8 support is brought up; production ships four.
#ifndef SPLASH_MAXIMUM_BATCH_WIDTH
#define SPLASH_MAXIMUM_BATCH_WIDTH 4u
#endif
#define SPLASH_PREFILL_TOKEN_BUDGET 2048u
#define SPLASH_DRAFT_SLIDING_WINDOW 2048u
#define SPLASH_TARGET_KV_BLOCK_TOKENS 32u
#define SPLASH_PREFILL_ATTENTION_TILE_ROWS 8u
// Development builds may lower this (e.g. to 1u) to test whether a prefill's
// output depends on how its chunk splits the attention history; production
// never overrides it.
#ifndef SPLASH_PREFILL_ATTENTION_MAXIMUM_SPLITS
#define SPLASH_PREFILL_ATTENTION_MAXIMUM_SPLITS 32u
#endif
#define SPLASH_VERIFY_ATTENTION_SPLITS 32u
// Verify attention starts from the configured split count and adds one split
// per this many visible Page32 blocks, capped at the maximum that sizes the
// partial workspace. Development builds may override the last two (both host
// and shader compilation must agree) to measure the split partition; a changed
// partition changes reduction order, so production never overrides them.
#ifndef SPLASH_VERIFY_ATTENTION_PAGES_PER_SPLIT
#define SPLASH_VERIFY_ATTENTION_PAGES_PER_SPLIT 16u
#endif
#ifndef SPLASH_VERIFY_ATTENTION_MAXIMUM_SPLITS
#define SPLASH_VERIFY_ATTENTION_MAXIMUM_SPLITS 128u
#endif
#define SPLASH_TARGET_SAMPLING_SHARDS 16u
#define SPLASH_DRAFT_SAMPLING_SHARDS 8u
// Rows of one value head's recurrent state a prefill GDN scan threadgroup
// carries through the chunk: four simdgroups whose lanes each own sixteen key
// columns of one row. The thread count follows from the rows: 128 columns /
// 16 per lane = 8 lanes per row, times the 16 rows; a static_assert in
// prefill/gdn.metal ties the two literals together.
#define SPLASH_GDN_SCAN_STATE_ROWS 16u
#define SPLASH_GDN_SCAN_THREADS 128u
#define SPLASH_ALLOCATION_EXTENT_TARGET_BYTES (128ull * 1024ull * 1024ull)
