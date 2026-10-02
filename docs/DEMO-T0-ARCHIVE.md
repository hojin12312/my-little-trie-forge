# Pi coding on local MLTF

Actual Pi file reads, edits and tests recorded as timestamped terminal output, replayed at1× without idle compression. This is a real-time terminal replay, not a screen capture.

Pi0.87.1 · local MLTF Qwen3.8-27B q8c · M5 Max40GPU/128GiB · T0 · thinking OFF. Extension/skill/prompt/theme/context discovery0; wrapper/preload bypassed; installed core retained; read/bash/edit/write tools enabled. User Pi installation/global configuration unchanged.

Isolated frontend launcher: [run-pi-demo.sh](../benchmarks/run-pi-demo.sh). Start the local backend first; the launcher creates the included audited synthetic fixture in a fresh temporary workspace by default. The launcher prints and preserves its fresh config/session directory without changing global settings. Recordings were made separately by the private campaign harness.

```sh
./benchmarks/run-pi-demo.sh --port 8000
```

The review replay masks local account names/home prefixes with equal-width replacements. Event order, original timestamps, waits, failures and retries are preserved. Unmodified masters remain private; metadata distinguishes raw-master and privacy-projection hashes.

## C1 actual coding

Outcome: 1/1 lanes changed files, passed the recorded model-run tests and the independent cancellation/supersession checks. Failures/retries/tool waits remain in the original recording.

[Uncut1× MP4](media/pi-c1-1x.mp4) · [Timing/artifact metadata](media/pi-c1-metadata.json)

[![Continuous1× excerpt](media/pi-c1-preview-1x.gif)](media/pi-c1-1x.mp4)

Cold first model request and per-turn cache reuse are preserved in the private request/session evidence; cache hits are described only where usage reports them. C1 uses one workspace; actual physical dispatch counts are {'1': 299, '2': 0, '3': 0, '4': 0}.

Actual cached tokens on the first request of each lane: [0]. Cache reuse occurred in 11/12 model requests. Media metadata records per-request input, output, cached tokens and TTFT.

## C4 actual coding

Outcome: 4/4 lanes changed files, passed the recorded model-run tests and the independent cancellation/supersession checks. Failures/retries/tool waits remain in the original recording.

[Uncut1× MP4](media/pi-c4-1x.mp4) · [Timing/artifact metadata](media/pi-c4-metadata.json)

[![Continuous1× excerpt](media/pi-c4-preview-1x.gif)](media/pi-c4-1x.mp4)

Cold first model request and per-turn cache reuse are preserved in the private request/session evidence; cache hits are described only where usage reports them. C4 uses four independent workspaces; actual physical dispatch counts are {'1': 122, '2': 94, '3': 33, '4': 181}.

Actual cached tokens on the first request of each lane: [0, 0, 1024, 1024]. Cache reuse occurred in 49/51 model requests. Media metadata records per-request input, output, cached tokens and TTFT.

The first two C4 requests reported0 cached tokens; the later two initial requests on the same fresh backend reused1024 shared-prefix tokens. This is not described as four cache-cold requests.

## Recorder overhead

Official matrix recording is OFF. The balanced controls use the same Pi TUI, fixed code-review request, fresh sessions/workspaces/backend and three matched pairs for each C. Default tools remain available.

| C | OFF native decode median tok/s | ON native decode median tok/s | Paired slowdown median [range] | Pairs |
|---|---:|---:|---:|---:|
| 1 | 70.39 | 70.40 | -0.05% [-0.29–0.13] | 3 |
| 4 | 200.14 | 200.25 | 0.01% [-1.92–1.31] | 3 |

Slowdown=100×(1−TPS_ON/TPS_OFF). Output, acceptance and actual B variation are retained in [paired data](data/recording-overhead.json); unequal outputs prevent isolated recorder attribution. PTY capture has no raster capture FPS/drop-frame metric; replay is rendered offline at12fps and source/media duration is verified within two frames.

Original casts, sessions, request events, counters, test output, source/media hashes and privacy checks stay in retained measurement evidence. Microphone/system audio/camera are OFF. 

Additional isolation control: native TPS changes in the two C4 pairs with differing outputs are not attributed solely to recording. Three balanced OFF/ON pairs replayed identical actual C4 events to isolate JSON serialization/write/flush CPU and I/O cost. This unpaced microcontrol is not a model benchmark or a video demo; GPU requests were0. Median additional CPU cost per original C4 trace was 12.12ms. See the [separate scope and original source hashes](data/recorder-fixed-trace-control.json).

Four earlier takes excluded for recorder EOF/turn-limit controller faults remain in retained measurement evidence with raw masters, sessions, workspaces and hashes. The first complete correctly controlled takes used the same tasks/model/sampling/caps. Model failures, corrections and retries were not edited out.
