# MLTF · My Little Trie Forge

<p align="center"><img src="docs/media/mltf-app-icon.svg" alt="MLTF" width="144"></p>

[English](README.md) · [한국어](README.ko.md) · [完整基准测试](docs/BENCHMARKS.md) · [Pi 演示](docs/DEMO.md) · [安装](docs/INSTALL.zh.md)

**8 位 Qwen3.8-27B。本地编程，速度飞快。256Ki 上下文。**

MLTF 在 M5 Max MacBook Pro 上运行 Qwen3.8-27B，专注于速度和内存效率。它的 Metal 内核和缓存只针对一个模型和一台机器做了专门优化。

## 实测性能

**MacBook Pro · M5 Max · 40 核 GPU · 128GiB 统一内存**

| 单请求冷预填充（2Ki） | 单请求 TTFT | 单请求解码 | 四请求合计解码 |
|---:|---:|---:|---:|
| **1,033.8 PP/s** | **2.094 s** | **71.7 tok/s** | **218.9 tok/s** |

## 演示视频

**C1 演示**

[![C1 实际编程](docs/media/pi-c1-xhigh-t1-4k30-preview.gif)](docs/media/pi-c1-xhigh-t1-4k30.mp4)

**C4 演示**

[![C4 四个并发编程任务](docs/media/pi-c4-xhigh-t1-4k30-preview.gif)](docs/media/pi-c4-xhigh-t1-4k30.mp4)

实际编程，推理过程可见：xhigh，temperature 1，输出上限 128Ki。完整录像为 **4K、30fps、1× 速度**播放。

[C1 完整视频](docs/media/pi-c1-xhigh-t1-4k30.mp4) · [C4 完整视频](docs/media/pi-c4-xhigh-t1-4k30.mp4) · [测试与录制开销](docs/DEMO.md)

## 特性

- 针对 Qwen3.8 的 Metal 内核、8 位预填充、DFlash2 验证和 INT8 KV 缓存。
- 基于前缀树的上下文复用，可选 SSD 缓存。
- 根据可用内存决定并发准入。

## 不同上下文长度下的性能

![不同上下文长度下的性能与内存](docs/media/performance-overview.png)

`C` 是并发请求数。PP/s 是每秒处理的输入 token 数；TTFT 计到第一个内容 token 为止。PP/TTFT 使用自然服务模式（S），解码/接受率使用全部就绪的运行（D），内存分别报告两者的 Metal 峰值。

**T0，关闭推理，关闭录制，输出上限 512。** 冷请求不复用 KV 缓存。吞吐/TTFT 为中位数；内存为最大值。

| 每请求输入 | C | 冷 PP/s | TTFT (s) | 解码 tok/s | 接受率 | 内存 S/D (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| 2Ki | 1 | 1,033.8 | 2.094 | 71.7 | 54.4% | 31.135 / 31.134 |
|  | 2 | 1,042.0 | 3.352 | 147.1 | 55.6% | 31.135 / 31.134 |
|  | 3 | 992.7 | 4.823 | 176.1 | 55.8% | 31.483 / 31.134 |
|  | 4 | 992.4 | 5.925 | 218.9 | 53.3% | 32.119 / 31.388 |
| 8Ki | 1 | 963.0 | 8.638 | 68.9 | 52.4% | 31.135 / 31.134 |
|  | 2 | 876.0 | 15.230 | 125.8 | 52.6% | 31.355 / 31.134 |
|  | 3 | 877.9 | 21.073 | 157.3 | 52.9% | 32.118 / 31.569 |
|  | 4 | 877.8 | 26.745 | 205.1 | 53.7% | 32.881 / 32.150 |
| 32Ki | 1 | 774.7 | 42.493 | 63.9 | 51.4% | 31.354 / 31.171 |
|  | 2 | 742.8 | 68.913 | 119.3 | 52.1% | 32.879 / 32.513 |
|  | 3 | 728.4 | 96.560 | 140.7 | 53.0% | 34.077 / 33.855 |
|  | 4 | 722.5 | 125.141 | 174.2 | 52.5% | 35.275 / 35.196 |
| 64Ki | 1 | 669.2 | 98.185 | 68.4 | 55.8% | 32.370 / 32.187 |
|  | 2 | 662.3 | 150.536 | 114.7 | 54.0% | 34.910 / 34.544 |
|  | 3 | 658.8 | 205.471 | 135.2 | 54.8% | 37.124 / 36.901 |
|  | 4 | 655.2 | 260.602 | 157.4 | 54.5% | 39.338 / 39.259 |
| 128Ki | 1 | 546.0 | 240.434 | 53.7 | 50.4% | 34.401 / 34.218 |
|  | 2 | 525.9 | 499.021 | 84.9 | 51.3% | 38.972 / 38.607 |
|  | 3 | 520.1 | 510.612 | 99.3 | 51.7% | 43.544 / 42.995 |
|  | 4 | 551.8 | 731.879 | 122.3 | 52.6% | 48.115 / 47.384 |
| near256Ki | 1 | 349.1 | 749.930 | 44.2 | 47.4% | 38.337 / 38.153 |
|  | 2 | 347.2 | 1,507.903 | 64.8 | 48.9% | 46.971 / 46.605 |
|  | 3 | 385.5 | 1,358.865 | 78.3 | 49.0% | 55.604 / 55.056 |
|  | 4 | 386.0 | 2,047.129 | 88.7 | 49.8% | 64.238 / 63.507 |

[完整样本、范围与实际 B/R](docs/BENCHMARKS.md) · [早期对比](docs/COMPARISON_ARCHIVE.md)

## 与 oMLX 0.7.0 的同机对比

在同一台 40 核 GPU、128GiB 统一内存的 M5 Max MacBook Pro 上测得，
使用相同的 Qwen3.8-27B 8 位目标来源和 DFlash2 draft。

单请求（C1），冷缓存，关闭推理，temperature 1.0，
top_p 0.95，top_k 20，输出上限 512 token。数值为中位数：
4Ki 和 32Ki 为 n=5，其余输入长度为 n=3。
测试的 oMLX 0.7.0 提交：`4d4f5a2`。

### 客户端观测到的解码吞吐

越高越好。百分比为实测吞吐中位数之间的比较。

| 输入 token 数 | MLTF 0.1.1 q8c / DFlash2 (tok/s) | oMLX 0.7.0 8-bit / DFlash2 (tok/s) | MLTF 吞吐提升 |
|---:|---:|---:|---:|
| 1Ki | 65.08 | 49.11 | +33% |
| 4Ki | 67.67 | 43.04 | +57% |
| 16Ki | 69.39 | 38.72 | +79% |
| 32Ki | 52.37 | 36.81 | +42% |
| 64Ki | 60.17 | 35.98 | +67% |

### 首 token 时间

越低越好。两列使用相同的客户端 TTFT 测量方法。

| 输入 token 数 | MLTF 0.1.1 q8c / DFlash2 (s) | oMLX 0.7.0 8-bit / DFlash2 (s) |
|---:|---:|---:|
| 1Ki | 1.174 | 1.646 |
| 4Ki | 4.489 | 5.558 |
| 16Ki | 18.727 | 22.613 |
| 32Ki | 40.243 | 47.510 |
| 64Ki | 92.157 | 108.570 |

以上是此前记录的同机对比，与上面 T0 / 关闭推理的 24 格测试相互独立；
本次 README 更新没有重新运行竞品基准测试。MLTF 使用 INT8
KV，而 oMLX 使用其默认的未量化缓存且关闭 TurboQuant，
因此缓存精度并不一致。结果只适用于所测试的硬件、工作负载
和配置，不代表所有模型或所有 Mac。
客户端流式事件的批处理是一项测量局限。

[完整对比方法、样本范围与产物标识](docs/COMPARISON_ARCHIVE.md)

## 本地启动

使用[已准备好的 q8c 检查点](https://huggingface.co/Ho-Jin-93/Qwen3.8-27B-MLTF-q8c)。请先阅读[模型条款](MODEL_ASSETS.md)。

```sh
# 1. 安装
brew install hojin12312/mltf/mltf
# 2. 下载模型
mltf download --model Ho-Jin-93/Qwen3.8-27B-MLTF-q8c --accept-model-licenses
# 3. 启动服务
mltf serve --model Ho-Jin-93/Qwen3.8-27B-MLTF-q8c --max-context 256K --max-memory 96G
```

默认地址：`127.0.0.1:8000`；SSD 缓存默认关闭。内存/SSD 限制和模型准备要求见[安装说明](docs/INSTALL.zh.md)。

`--max-concurrent-requests` 取 1–4，可选地限制同时进行的生成请求数；超出的请求在现有安全限制内按先进先出排队。不设置则保持自动并发，且不会改变物理批宽度。见[活跃请求数限制](docs/MAX_CONCURRENT_REQUESTS.md)。

MLTF 最多每 24 小时检查一次 GitHub Releases 是否有新的稳定版，仅在存在新版本时，才在启动时打印一行提示，并在 `/status` 和 Web UI 中显示一个小提示。检查结果会被缓存，在服务器 Ready 之后于后台运行，绝不会延迟或影响推理；失败时保持静默。它不会下载或安装任何东西，也不发送任何遥测数据。可用 `mltf serve --no-update-check` 或 `MLTF_NO_UPDATE_CHECK=1` 关闭。见[更新提示](docs/INSTALL.zh.md#更新通知)。


Python 安装：`pip install https://github.com/hojin12312/my-little-trie-forge/releases/download/v0.1.2/my_little_trie_forge-0.1.2-py3-none-macosx_26_0_arm64.whl` 或 `uv tool install https://github.com/hojin12312/my-little-trie-forge/releases/download/v0.1.2/my_little_trie_forge-0.1.2-py3-none-macosx_26_0_arm64.whl`。从源码转换见 [prepare-q8c](MODEL_ASSETS.md)。

## 支持范围

已验证设备：**M5 Max，40 核 GPU / 128GiB**。需要 macOS 26.4 或更新版本以及所需的 GPU 特性。物理解码范围为 B1–B4；262,144 token 的上下文上限包含输出。见[验证与支持范围](docs/VALIDATION.md)。

## 开源基础

MLTF 是**派生自 Splash 的 Apache-2.0 项目**。DFlash2、KV、服务和缓存方面的上游贡献与声明保留在 [LICENSE](LICENSE)、[UPSTREAM.md](UPSTREAM.md)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) 中。
