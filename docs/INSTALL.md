# Installation

[한국어](INSTALL.ko.md) · [README](../README.md)

Reference validation: M5 Max40GPU/128GiB, macOS27.0(26A428). Native minimum: macOS26.4 with required GPU features; Python3.13. Other GPU/RAM configurations are not performance-certified.

From reviewed source, use `make install-environment`, `make -j4 MAX_BATCH_WIDTH=4`, and `make check`. The README serves a locally prepared q8c model and separates model data from installed program files. Default port8000, loopback bind; Ctrl+C drains and exits.

Code packages contain no weights. Read [MODEL_ASSETS.md](../MODEL_ASSETS.md) and explicitly accept model terms before preparation. Fresh source preparation checks110GiB, or65GiB with verified target-blob reuse: preparation disk space, not runtime RAM. This campaign reused an existing verified model and does not claim a fresh network weight download. Raw MLX8bit repositories do not substitute for prepared q8c packages.

## Install and download the model

```sh
brew install hojin12312/mltf/mltf
# Alternatively: pip install https://github.com/hojin12312/my-little-trie-forge/releases/download/v0.1.1/my_little_trie_forge-0.1.1-py3-none-macosx_26_0_arm64.whl
mltf download --model Ho-Jin-93/Qwen3.8-27B-MLTF-q8c --accept-model-licenses
mltf serve --model Ho-Jin-93/Qwen3.8-27B-MLTF-q8c --max-context 256K --max-memory 96G
```

The downloader verifies the manifest and SHA-256. Review the model card and terms before consenting. Source conversion uses `prepare-q8c` from [MODEL_ASSETS.md](../MODEL_ASSETS.md).

README256K/96G matches the validated target-machine profile. 32K/48G is a separate conservative profile and the prior soak scope. Signing is ad-hoc; Developer ID/notarization is not included.

## Memory and SSD cache

`--max-memory` defaults to `auto`: Metal's recommended working set minus `max(1GiB, 2%)`, capped further by a smaller user limit. A larger user value cannot exceed the automatic ceiling. This machine reports a 120GiB recommended working set, giving an automatic ceiling of about 117.6GiB; measurements and the README command explicitly use 96GiB.

The budget covers engine Metal allocations and governor-managed memory. It is not an OS limit on all processes' RAM. Host availability and system memory pressure are also checked; low headroom or critical pressure denies new allocations. The host reserve is `min(10% of physical RAM, 2GiB)`, with another 1GiB margin checked for growth admission. Inspect effective values in `/status` under `memory_plan` and `memory_governor`.

| Option | Default | Controls |
|---|---|---|
| `--max-memory` | `auto` | MLTF memory budget, e.g. `48G` or `96G` (`G` means GiB) |
| `--ssd-root` | unset, SSD cache OFF | Opt-in SSD cache at an absolute private 0700 directory |
| `--ssd-quota-bytes` | when enabled: 1,073,741,824 (1GiB) | Cache quota, positive integer bytes |
| `--ssd-free-floor-bytes` | when enabled: 8,589,934,592 (8GiB) | Filesystem free-space floor, nonnegative integer bytes |

Default cache disk usage is zero while SSD caching is disabled; model weights/downloads use separate disk space. SSD byte-limit flags require `--ssd-root`. For a 20GiB cache quota with an 8GiB free-space floor:

```sh
mltf serve --model local/Qwen3.8-27B-q8c --model-path "$HOME/mltf-models/q8c" --max-context 256K --max-memory 96G --ssd-root "$HOME/mltf-cache" --ssd-quota-bytes 21474836480 --ssd-free-floor-bytes 8589934592
```

The directory must satisfy the private permission checks. `/status` reports storage, restore and quota state under `storage`.
