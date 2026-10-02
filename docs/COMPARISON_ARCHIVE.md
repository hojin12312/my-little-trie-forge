# Historical comparison archive

Preserved from the owner-reviewed snapshot `144f849d4c3340b2bd6a51493ae5d1c420357f2e`, file `BENCHMARKS.md`. The original root document remains available. These historical sampling measurements are separate from the new greedy coding matrix. No competitor was rerun.

# Same-machine8bit benchmark PASS_IN_SCOPE

M5 Max40GPU cores/128GiB, one macOS27.0/26A428 host, AC, system-controlled fans. oMLX commit4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40; MTPLX commit50ab114253e133616bf8c9ba090734d40fad5b3b. MLTF0.1.1 frozen profile; oMLX0.7.0 official macOS26/27 bundled runtime; MTPLX2.12.0 official wheel. ThinkingOFF; T1/top_p.95/top_k20; cap512; distinct nonce before Page32; cold primary cache0. No outliers removed. Client streaming event timing; event batching is a limitation. Native engine TG/prefill remain separate.

Splash's official Qwen3.8-27B reference uses 4-bit weights and is therefore excluded from the primary 8-bit head-to-head benchmark.

MLTF q8c source and both oMLX targets: mlx-community/Qwen3.8-27B-8bit@815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9. DFlash2 draft: incoai/Qwen3.8-27B-DFlash2@dedf8df68adfb1afeaf7b7480c0a0243108177b4. MTPLX: Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality@300a4ac6c6058585e80571ff6c910819711843ec, affine8bit/group64/native recommended depth3. MTPLX is same architecture/8bit quality class, not identical weight bytes. oMLX plain uses the same source target; accelerated DFlash is the preregistered calibration-v2 selection, not a post-result switch.

| Profile | Actual prompt | TTFT median s | Client decode median tok/s [range] | Full wall median s | n |
|---|---|---:|---:|---:|---:|
| MLTF0.1.1 q8c/DFlash2 | [1024] | 1.174 | 65.08 [57.77–65.55] | 9.026 | 3 |
| MLTF0.1.1 q8c/DFlash2 | [4096] | 4.489 | 67.67 [59.72–78.62] | 12.243 | 5 |
| MLTF0.1.1 q8c/DFlash2 | [16384] | 18.727 | 69.39 [65.23–70.45] | 26.561 | 3 |
| MLTF0.1.1 q8c/DFlash2 | [32768] | 40.243 | 52.37 [50.05–65.16] | 49.838 | 5 |
| MLTF0.1.1 q8c/DFlash2 | [65536] | 92.157 | 60.17 [59.20–69.91] | 100.649 | 3 |
| oMLX0.7.0 plain8bit | [1024] | 1.518 | 18.96 [18.96–18.96] | 28.466 | 3 |
| oMLX0.7.0 plain8bit | [4096] | 5.136 | 18.81 [18.72–18.81] | 32.320 | 5 |
| oMLX0.7.0 plain8bit | [16384] | 21.328 | 18.00 [17.84–18.00] | 49.724 | 3 |
| oMLX0.7.0 plain8bit | [32768] | 47.420 | 17.05 [16.75–17.18] | 77.390 | 5 |
| oMLX0.7.0 plain8bit | [65536] | 106.960 | 15.66 [15.13–15.68] | 139.586 | 3 |
| oMLX0.7.0 DFlash8bit | [1024] | 1.646 | 49.11 [41.58–49.63] | 12.061 | 3 |
| oMLX0.7.0 DFlash8bit | [4096] | 5.558 | 43.04 [37.05–46.97] | 17.432 | 5 |
| oMLX0.7.0 DFlash8bit | [16384] | 22.613 | 38.72 [37.94–41.81] | 35.763 | 3 |
| oMLX0.7.0 DFlash8bit | [32768] | 47.510 | 36.81 [34.69–41.20] | 61.324 | 5 |
| oMLX0.7.0 DFlash8bit | [65536] | 108.570 | 35.98 [31.71–38.97] | 122.774 | 3 |
| MTPLX2.12.0 Quality8bit/native MTP depth3 | [1024] | 1.676 | 48.87 [48.08–52.17] | 12.137 | 3 |
| MTPLX2.12.0 Quality8bit/native MTP depth3 | [4096] | 5.824 | 49.98 [43.63–51.83] | 16.205 | 5 |
| MTPLX2.12.0 Quality8bit/native MTP depth3 | [16384] | 24.177 | 47.31 [44.92–48.62] | 34.779 | 3 |
| MTPLX2.12.0 Quality8bit/native MTP depth3 | [32768] | 50.767 | 42.62 [40.82–45.82] | 63.262 | 5 |
| MTPLX2.12.0 Quality8bit/native MTP depth3 | [65536] | 115.842 | 41.96 [39.25–42.00] | 128.248 | 3 |

C4 HTTP barrier/distinct4K/cap256, one separate warmup wave each boot. Aggregate delivered tokens divided by complete-wave wall; no internal fixed-work substitution.

| Profile | Aggregate median tok/s [range] | Wave median s | TTFT median s | n |
|---|---:|---:|---:|---:|
| MLTF0.1.1 q8c/DFlash2 | 39.88 [38.89–40.76] | 25.677 | 12.688 | 3 |
| oMLX0.7.0 plain8bit | 21.74 [21.63–21.79] | 47.102 | 29.523 | 3 |
| oMLX0.7.0 DFlash8bit | 25.60 [25.55–25.71] | 39.996 | 27.703 | 3 |
| MTPLX2.12.0 Quality8bit/native MTP depth3 | 22.77 [21.92–22.86] | 44.968 | 22.529 | 3 |

8bit classification refers to target weights. The official MTPLX turbo/native-MTP runtime uses its default4bit/affine/group64 draft LM head; its target remains the8bit Optimized Quality pack, not the4bit Speed pack. MLTF uses INT8 KV; competitors use their stock unquantized cache configuration (oMLX TurboQuant OFF; MTPLX q8/q4 paged KV opt-in OFF). Per-layer competitor dtype is not separately instrumented. MTPLX uses its stock serial external request scheduler; no unsupported experimental27B batching is enabled. These are tested default/recommended profiles, not a claim of identical KV formats. Native TG/prefill and available native memory/host snapshots are separate fields in the private statistics; unavailable telemetry is not synthesized.

Ratios are limited to local medians on this one tested host and configuration. No universal superiority, other-Apple-Silicon, world-fastest or ratio against external published values is supported. Published external references are kept separate from local samples and are not used for ratios. Stock MTPLX anonymous C4 requests emitted session-postcommit wait/defer messages; complete-wave wall includes this public service behavior. No session/cache setting was changed after seeing results. Other engine concurrency is not inferred merely from four submitted clients. Optional128K and shared-prefixC4 are UNTESTED.

## Package identities

- oMLX 0.7.0 macOS26/27 official DMG SHA256: `2e3bb06ac6ee7f50986ba1417e909d432ccd2be471db752a4a2d3b5651e3bce0`
- MTPLX 2.12.0 official wheel SHA256: `c90258e11e4d2aad96ae91493f6038faf650c8457d523867e547a7a1242fcae6`
- MLTF measured native executable SHA256: `0d3d56f7a8133b01d9bcf6ea8414a3e05bfb50ee248aaa8b8695a3db9f06a7bb`
- MLTF measured metallib SHA256: `7d8388c89c7a8bdac1cfc065b9daa1e1b02653cb5b7a78020cb9cb4bc0efa3e3`

The README memory table distinguishes native Metal allocation from sampled process-tree RSS. RSS is not total unified/GPU memory and must not be used alone to rank engine memory needs. Native Metal peaks for competitors were not reported by this harness.

## Hardware support and verification

Supported devices: M5+ family, with required GPU features and sufficient memory. Verified device: M5 Max40c128GB, MacBook Pro (physical memory128GiB). Other GPU/RAM configurations and successor chips have not been measured. The declared support scope is not a performance or every-memory-configuration guarantee.

Memory guidance:48GB or more is an estimated minimum for short contexts/bounded concurrency, not a verified48GB SKU or a guarantee of all context/cohort sizes. The measured hardware had128GiB physical memory. The34GiB governor-budget gate ran on that128GiB host and does not simulate or qualify a34/48GB machine.

## MLTF native phase throughput

These tables separate prefill and decode **batch wall time** using runtime request counters. They exclude HTTP/tokenization/queue gaps and the other phase; they are not GPU-only time, TTFT or full-wave throughput. Counts were checked against actual usage. Rates are aggregate delivered tokens per phase, not per-client rates. Sampling/host/model/production profile are unchanged.

| Actual input tokens | C | Output/client | n | Prefill tok/s median [range] | Decode tok/s median [range] |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 512 | 3 | 938.41 [827.71–1057.91] | 65.12 [57.83–65.65] |
| 4096 | 1 | 512 | 5 | 930.33 [880.38–1038.68] | 67.65 [59.79–78.62] |
| 16384 | 1 | 512 | 3 | 880.00 [853.30–920.92] | 69.36 [65.27–70.43] |
| 32768 | 1 | 512 | 5 | 817.15 [782.97–839.65] | 52.44 [50.11–65.11] |
| 65536 | 1 | 512 | 3 | 712.75 [669.51–715.13] | 60.14 [59.17–69.79] |

Concurrency slice: fixed4K per client, distinct-prefix simultaneous barrier, output256/client, warmup1 and n3/condition. C1–C3 newly measured using the qualified final installed payload; C4 derives from exact same-profile/cap/context existing three HTTP waves. This is a representative4K concurrency slice, not the entire context×concurrency matrix.

| Context/client | C | Output/client | n | Aggregate prefill tok/s median [range] | Aggregate decode tok/s median [range] |
|---:|---:|---:|---:|---:|---:|
| 4096 | 1 | 256 | 3 | 954.98 [953.98–977.63] | 68.01 [54.50–68.21] |
| 4096 | 2 | 256 | 3 | 1023.11 [993.85–1046.46] | 113.68 [96.84–118.40] |
| 4096 | 3 | 256 | 3 | 982.07 [954.49–987.31] | 117.63 [116.89–118.67] |
| 4096 | 4 | 256 | 3 | 954.32 [945.57–957.49] | 121.52 [113.53–133.64] |

Context and concurrency tables have different output caps, explicitly shown. Never mix their rows into one scaling ratio. The historical approximately395 tok/s number was an older native fixed-work control:39-token prompt,greedy,64 output/lane,physicalB4,one sample,aggregate decode-only256 tokens/648.041ms. Its high draft acceptance/workload/artifact differ from these sampling/code-prompt measurements, so it is not substituted for a current production or competitor row. Historical437 tok/s was B8 research outside production scope.
