# Pi 실제 코딩 · Reasoning ON

Pi 0.87.1 · MLTF Qwen3.8-27B q8c · M5 Max 40 GPU / 128 GiB

Reasoning **ON · xhigh** · 온도 **1** · top_p **0.95** · top_k **20** · min_p **0** · presence/frequency penalty **0** · repetition penalty **1** · preserve_thinking **true** · Max Tokens **131,072** · 논리 컨텍스트 상한 **262,144**.

공식 [Qwen3.8-27B 설정](https://huggingface.co/Qwen/Qwen3.8-27B)을 사용했습니다. 실제 HTTP 요청과 reasoning SSE·usage를 검증했습니다. 상한은 강제 출력 길이가 아니며, 출력 cap은 실제 문맥 여유의 제약을 받습니다.

사용자 Pi 설치·전역 설정의 전후 hash는 같습니다. 별도 config·session·작업공간에서 확장·스킬·프롬프트·테마·컨텍스트 자동 발견을 끄고 기본 read/bash/edit/write 도구를 유지했습니다. 설치된 core를 유지하고 wrapper/preload를 우회했습니다.

**3840×2160 · 30fps · 원래 시간 간격 1× · idle 압축 없음.** Reasoning 내용을 화면에 노출한(`hideThinkingBlock: false`) timestamp 기반 PTY 재생본입니다. 모델 실행 중 화면 인코딩은 없으며 추론 종료 후 software H.264 CRF16으로 렌더링했습니다. 마이크·카메라·오디오는 OFF입니다. 실제 braille codepoint를 2×4 점으로 그려 Menlo의 누락 글리프를 보완했습니다. 중간 스피너 움직임을 만들어내지 않습니다.

로컬 계정명·home prefix만 동일 문자 폭으로 가린 재생본입니다. 원본 raw master, timestamp, 순서, 실패·재시도는 비공개 증거에 보존합니다. 처리량은 thinking·답변·도구 호출 토큰을 포함하며 답변만의 속도가 아닙니다. 24셀 고정 벤치마크는 기존 T0·Reasoning OFF·녹화 OFF 조건 그대로입니다.

격리 실행: backend를 loopback에서 먼저 실행한 뒤 아래 명령을 사용하세요. 격리 launcher는 자동 녹화하지 않습니다.

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

## 녹화 OFF/ON 비교

| C | OFF tok/s median | ON tok/s median | Paired slowdown median [range] | n |
|---|---:|---:|---:|---:|
| C1 | 63.61 | 64.91 | -1.98% [-3.82, 7.31] | 3 |
| C4 | 163.01 | 164.13 | 3.30% [-19.22, 11.34] | 3 |

비교 쌍은 UI 매핑 보완 전의 high 표시를 공통으로 사용했지만 실제 HTTP reasoning_effort는 모두 xhigh였습니다. 최종 코딩 영상의 UI는 xhigh로 보완했습니다. 동일 실제 요청 hash와 동일 TUI·기본 도구·출력 상한을 사용해 각 C 3쌍을 OFF/ON, ON/OFF, OFF/ON 순서로 실행했습니다. T1 확률적 출력이므로 TPS 차이를 녹화만의 인과 효과로 볼 수 없습니다. Slowdown=100×(1−ON/OFF). [전체 쌍별 메트릭](data/usage-recording-overhead.json)은 PP·decode·acceptance·B·메모리·client rates와 원본 hash를 보존합니다. PTY capture에는 화면 capture의 drop-frame 수가 없습니다.

[Earlier T0 / reasoning OFF recordings](DEMO-T0-ARCHIVE.ko.md)

## 실제 사용 프로필 메트릭

| C | TTFT first output median (s) | Native PP tok/s | Native decode tok/s | Peak Metal GiB |
|---|---:|---:|---:|---:|
| C1 | 0.925 | 842.95 | 67.99 | 34.429 |
| C4 | 1.573 | 776.73 | 166.86 | 47.116 |

TTFT는 각 모델 turn에서 처음 도착한 비어 있지 않은 reasoning·content·tool-call 출력까지이며 cache hit를 포함한 전체 turn의 중앙값입니다. cold 최초 요청만의 값이 아닙니다. 입력·출력·thinking token·cache·각 종류의 TTFT·실제 B dispatch·비가산 RSS/footprint는 [요청별 데이터](data/usage-demo-metrics.json)를 확인하세요.

Reasoning이 화면에 보이는 실제 4K 프레임: [C1](media/pi-c1-reasoning-4k.png) · [C4](media/pi-c4-reasoning-4k.png)
