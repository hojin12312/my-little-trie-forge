# Actual Pi coding · reasoning ON

Pi 0.87.1 · MLTF Qwen3.8-27B q8c · M5 Max 40 GPU / 128 GiB

Reasoning **ON · xhigh** · temperature **1** · top_p **0.95** · top_k **20** · min_p **0** · presence/frequency penalty **0** · repetition penalty **1** · preserve_thinking **true** · Max Tokens **131,072** · logical context limit **262,144**.

Uses the official [Qwen3.8-27B settings](https://huggingface.co/Qwen/Qwen3.8-27B). Actual HTTP requests and reasoning SSE/usage were verified. The output cap is not a forced generation length and remains subject to available context.

Pi installation/global hashes match before and after. Separate config, sessions and workspaces disable extension/skill/prompt/theme/context discovery while retaining read/bash/edit/write. Installed core is retained; wrapper/preload is bypassed.

**3840×2160 · 30fps · original timing at 1× · no idle compression.** Reasoning is visible (`hideThinkingBlock: false`). Timestamped PTY replay, software H.264 CRF16 rendered after inference. Microphone/camera/audio OFF. Captured braille codepoints are drawn as exact 2×4 dots to avoid missing Menlo glyphs; no intermediate spinner motion is invented.

Only account names/home prefixes are masked with equal-width replacements. Raw masters, timestamps, order, failures and retries stay in retained measurement evidence. Throughput includes thinking, visible answers and tool-call tokens; it is not answer-only speed. The fixed 24-cell benchmark retains T0, reasoning OFF, recording OFF.

Isolated frontend: start the loopback backend first. The launcher does not record automatically.

```sh
./benchmarks/run-pi-demo.sh --profile usage --workspace ./synthetic-workspace
```

## C1

[![C1 continuous 1× excerpt](media/pi-c1-xhigh-t1-4k30-preview.gif)](media/pi-c1-xhigh-t1-4k30.mp4)

[4K30 uncut 1× MP4](media/pi-c1-xhigh-t1-4k30.mp4) · [Metadata](media/pi-c1-xhigh-t1-4k30-metadata.json)

1/1 lanes: actual read/edit-or-write/bash tests + post-take tests + independent cancellation/supersession checks PASS. Source 95.366s / video 95.400s. 12 model requests; 2,646 reasoning / 4,597 total completion tokens.

Native prefill 842.95 tok/s · native decode 67.99 tok/s · accepted draft 3525/7420 (47.51%) · actual physical dispatch counts {'1': 1072, '2': 0, '3': 0, '4': 0}.

## C4

[![C4 continuous 1× excerpt](media/pi-c4-xhigh-t1-4k30-preview.gif)](media/pi-c4-xhigh-t1-4k30.mp4)

[4K30 uncut 1× MP4](media/pi-c4-xhigh-t1-4k30.mp4) · [Metadata](media/pi-c4-xhigh-t1-4k30-metadata.json)

4/4 lanes: actual read/edit-or-write/bash tests + post-take tests + independent cancellation/supersession checks PASS. Source 220.336s / video 220.400s. 46 model requests; 11,824 reasoning / 21,527 total completion tokens.

Native prefill 776.73 tok/s · native decode 166.86 tok/s · accepted draft 16672/33663 (49.53%) · actual physical dispatch counts {'1': 207, '2': 158, '3': 348, '4': 822}.

## Recorder OFF/ON controls

| C | OFF tok/s median | ON tok/s median | Paired slowdown median [range] | n |
|---|---:|---:|---:|---:|
| C1 | 63.61 | 64.91 | -1.98% [-3.82, 7.31] | 3 |
| C4 | 163.01 | 164.13 | 3.30% [-19.22, 11.34] | 3 |

Controls used the earlier common high footer label while actual HTTP reasoning_effort was xhigh in every arm; final coding videos correct the UI map to xhigh. Three pairs per C use identical actual request hashes, TUI, tools and caps, ordered OFF/ON, ON/OFF, OFF/ON. T1 stochastic outputs prevent attributing TPS differences solely to recording. Slowdown=100×(1−ON/OFF). [Full paired metrics](data/usage-recording-overhead.json) preserve PP, decode, acceptance, B, memory, client rates and raw hashes. PTY capture has no raster capture drop-frame count.

[Earlier T0 / reasoning OFF recordings](DEMO-T0-ARCHIVE.md)

## Actual-use profile metrics

| C | TTFT first output median (s) | Native PP tok/s | Native decode tok/s | Peak Metal GiB |
|---|---:|---:|---:|---:|
| C1 | 0.925 | 842.95 | 67.99 | 34.429 |
| C4 | 1.573 | 776.73 | 166.86 | 47.116 |

TTFT ends at the first nonempty reasoning/content/tool-call delta. It is the median over all model turns, including cache hits, rather than cold-first TTFT. See [per-turn data](data/usage-demo-metrics.json) for input/output/thinking tokens, cache, separate TTFT types, actual B dispatches and nonadditive RSS/footprint.

Actual 4K frames with visible reasoning: [C1](media/pi-c1-reasoning-4k.png) · [C4](media/pi-c4-reasoning-4k.png)
