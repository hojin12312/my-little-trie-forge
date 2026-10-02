# Upstream and maintenance

MLTF (My Little Trie Forge) is derived from [incoai/splash](https://github.com/incoai/splash),
whose incorporated parent in this private development line is
`134807b80bd6f64b1533cb1306358c042bb750d5` (Apache-2.0). This line is
independently maintained. This relationship does not imply endorsement by
Splash or its contributors. Existing upstream DFlash2, INT8 KV, 256Ki,
scheduler, cache, and serving/API foundations remain credited to Splash.
MLTF's modifications and extensions are recorded in the source history.

The [oMLX SSD sidecar PR](https://github.com/jundot/omlx/pull/2569) informed
durable cache design and failure-scenario review. The related native
implementation is conservatively acknowledged as an algorithm adaptation;
see ACKNOWLEDGEMENTS.md and THIRD_PARTY_NOTICES.md. oMLX is not a declared
runtime dependency.

## Incorporated MLTF 0.1.1 changes

The incorporated Splash revision remains the exact parent above, not latest
upstream. This release carries MLTF q8c preparation/verification, W8A8 prefill,
W8A16 split2-rounded execution, B3 padded-M32, Page32/GDN/KV restore and
bounded asynchronous SSD storage, memory/admission and packaging extensions.
Common wide execution implementations are retained for source stability;
release compilation is fixed to B4 and cost-cap research is disabled.
New lane-binding, q8 slab and paged-attention common helpers retain the
Splash-derived project context. oMLX cache adaptation labels remain unchanged;
new B4 release/installer changes do not add oMLX source translation.

MLX/MLX-LM are evaluation/design references and the target asset input format;
neither is in the pinned runtime Python dependency list. No clean-room,
upstream endorsement or universal hardware support claim is made.
