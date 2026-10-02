# Design acknowledgements

MLTF derives from Splash. Splash's DFlash2, INT8 KV, scheduler/cache,
serving/API and long-context foundations are credited in UPSTREAM.md.

The [oMLX sidecar work](https://github.com/jundot/omlx/pull/2569), authored
by the same developer and merged into jundot/omlx, informed durable
state/KV publication, restart indexing, bounded asynchronous ownership and
safe walkback/fallback. MLTF's analogous native cache paths are
conservatively treated as algorithm adaptations under oMLX's Apache-2.0
terms. This acknowledgement does not claim that the Python source was
translated line for line. No oMLX package is bundled as a runtime dependency.

MLX and MLX-LM were evaluation references, not declared runtime packages.
The inherited Splash q4_sgmatrix.h comment refers to an “MLX steel pattern”
for a short-lived matrix object and register accumulator. The design influence
is acknowledged without claiming MLX source inclusion or clean-room
independence.
