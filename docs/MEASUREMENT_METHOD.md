# MLTF coding measurement method

Fixed before official timings on 2026-10-01. MLTF only, q8c/DFlash2/INT8 KV/split2-rounded; greedy T0, top_p1/top_k1 (native greedy argmax ignores sampling cutoffs), thinking OFF, cap512, recording OFF.

Input counts: 2048,8192,32768,65536,131072,261568 including the complete chat template; 261568+512+64=262144. C1/C2/C3/C4, three fixed bugfix/feature/refactor rounds, with two fixed repeats for 2Ki C1/C4 (n5). All 152 independent S/D samples are retained; no outlier removal or workload replacement. Corpus contains original Apache-2.0 queue/cache code and distinct pinned CPython3.13.14 dependency implementation blocks with PSF notices. No repeated filler.

S uses the exact qualified installed server/native payload under ordinary scheduling. TTFT is submit to first nonempty SSE content. PP and service-native decode use native batch-ticket wall counters, verified against actual usage and per-batch traces. Full-wave includes prefill/queue/decode/drain.

D is a bench-only adapter linked to an unchanged qualified static library and metallib. It prepares independent cold KV/GDN/draft states, verifies readiness, releases all lanes and records actual physical width. Primary window: first complete full-width cycle until immediately before first lane completion; at least16 cycles and1s. Native batch-ticket wall, call wall and elapsed time remain separate. No production kernel, cache layout or scheduler default is changed. Four pilot lanes matched natural non-held references in output tokens and final GDN/draft digests.

Draft acceptance uses shipping retained accepted proposal count divided by all seven proposals per active lane in the same window. Rejections remain in the denominator; anchor/bonus and inactive padded rows are not accepted proposals. Terminal boundary cycles are outside the window. This is a workload acceptance metric, not a task success rate.

96GiB fixed ceiling on M5 Max40GPU/128GiB/macOS27.0(26A428), AC; native OS-safe governor also applies. Primary memory is the maximum fresh-process lifetime tracked dense+sparse Metal peak, including warmup; request-phase samples and per-process physical footprint/process-tree RSS are retained separately. No domains are added.

Timeout seconds=max(900,ceil(input*C/100*1.5+240)); socket +60, D startup +180. Resume preserves attempt identity. Original logs >=1MiB are retained as lossless verified gzip with original/compressed hashes. Rates use median/min/max/n; draft ratios use summed counts. Quality/early EOS/cap hits are recorded; no silent truncation, EOS override or successful-test claim from a throughput completion.

