# 安装

[English](INSTALL.md) · [한국어](INSTALL.ko.md) · [README](../README.md)

> 本文是英文版 [INSTALL.md](INSTALL.md) 的中文翻译，以英文版为准。

参考验证环境：M5 Max 40 核 GPU / 128GiB，macOS 27.0（26A428）。原生最低要求：macOS 26.4，且具备所需的 GPU 特性；Python 3.13。其他 GPU / 内存配置未经过性能认证。

从经过审阅的源码构建时，使用 `make install-environment`、`make -j4 MAX_BATCH_WIDTH=4` 和 `make check`。README 中的命令运行本地准备好的 q8c 模型，并将模型数据与已安装的程序文件分开存放。默认端口 8000，仅绑定回环地址；按 Ctrl+C 会处理完在途请求后退出。

代码包不含权重。准备模型之前，请阅读 [MODEL_ASSETS.md](../MODEL_ASSETS.md) 并明确接受模型条款。从源码全新准备时需要 110GiB 磁盘空间；若复用已验证的 target blob，则需要 65GiB。这是准备阶段的磁盘空间，不是运行时内存。本次测试复用了已有的已验证模型，不声称做过全新的网络权重下载。原始的 MLX 8bit 仓库不能替代准备好的 q8c 包。

## 安装并下载模型

```sh
brew install hojin12312/mltf/mltf
# 或者：pip install https://github.com/hojin12312/my-little-trie-forge/releases/download/v0.1.2/my_little_trie_forge-0.1.2-py3-none-macosx_26_0_arm64.whl
mltf download --model Ho-Jin-93/Qwen3.8-27B-MLTF-q8c --accept-model-licenses
mltf serve --model Ho-Jin-93/Qwen3.8-27B-MLTF-q8c --max-context 256K --max-memory 96G
```

下载器会校验 manifest 和 SHA-256。同意之前，请先阅读模型卡和相关条款。从源码转换使用 [MODEL_ASSETS.md](../MODEL_ASSETS.md) 中的 `prepare-q8c`。

README 中的 256K/96G 对应已验证的目标机器配置。32K/48G 是另一套保守配置，也是此前 soak 测试的范围。签名为 ad-hoc；不包含 Developer ID 和 notarization。

## 登录时自动运行（macOS）

`mltf service` 把 `mltf serve` 安装为当前用户的 LaunchAgent，这样服务器会在登录时启动，无需保持终端打开。引擎使用 Metal 需要图形登录会话，所以这里用的是 LaunchAgent，而不是系统守护进程。

```sh
mltf service install -- --model Ho-Jin-93/Qwen3.8-27B-MLTF-q8c --max-context 256K --max-memory 96G
mltf service status
mltf service stop        # 立即停止；下次登录时仍会自动启动
mltf service start
mltf service restart
mltf service uninstall   # 同时移除登录自启
```

- `--` 之后的所有参数原样传给 `mltf serve`，并用同一个解析器校验，所以写错会在安装时就报错。端口总是被显式记录下来，因为登录服务看不到 `SPLASH_PORT`。每个端口是一个独立的服务。其余操作会自己找到已安装的服务，不需要记端口：`mltf service status` 列出所有已安装的服务，`start`、`stop`、`restart`、`uninstall` 在只有一个服务时直接作用于它，有多个时要求用 `--port` 指定（`--port` 或 `SPLASH_PORT` 始终优先）。
- 服务从 `PATH` 中启动 `mltf`（源码检出目录中则是 `./mltf`），并保持路径原样，所以 Homebrew 的 `bin/mltf` 在升级后依然有效。可用 `--executable PATH` 指定其他可执行文件。与任何正在运行的服务器一样，升级前请先停止该服务。
- 相对路径的 `--model-path` 会被拒绝，因为登录服务不是从你当前的目录启动的。`MLTF_DATA_ROOT` 会被带过去，`SPLASH_API_KEY` 不会。需要 API key 时请显式传 `--api-key`：它会存放在 plist 中，该文件只有你本人可读。
- 崩溃后不会自动重启，因此在内存压力下失败的服务器不会陷入反复重启的循环。用 `mltf service start` 重新启动。
- 请先准备好模型（`mltf download`）：登录服务没有终端来完成模型准备。
- 标签为 `io.github.hojin12312.mltf.serve-<port>`，plist 位于 `~/Library/LaunchAgents/<label>.plist`，输出写入 `~/Library/Logs/mltf-<port>.log`，该日志不会轮转。`mltf service status` 会打印状态和这些路径。端口已监听并不代表 Ready，请查看日志。

## 更新通知

MLTF 会自动检查更新（仅通知；它不是自动更新器）。服务器 Ready 之后，后台守护线程会向公开的 GitHub Releases API（`hojin12312/my-little-trie-forge`，最新已发布、非草稿、非预发布的版本）查询是否有更新的稳定版本。请求是一次不带认证的 HTTPS GET，使用固定的 `User-Agent`；其中不包含模型名、提示词、路径、主机名、硬件或使用数据，也不发送任何遥测数据。

- 不阻塞：服务器从不等待 GitHub。DNS、TLS、限流（403/429）、404、5xx、无效 JSON、超时和离线等情况都不致命，不会改变 `/status` 的健康状态，也不影响模型加载、就绪状态、HTTP API 和 Web UI。状态最多显示 8 秒的 `checking`，之后变为 `unavailable` 并附带简短的 `reason`。
- 带缓存：结果保存在 MLTF 数据目录下的 `update/update-check.json`（源码检出目录中为 `build/runtime/update/`），24 小时内复用。失败的尝试会退避一小时。缺失、损坏、旧 schema 或日期在未来的缓存会被忽略并以原子方式重写。过期的缓存绝不会作为新结果显示。
- 版本比较按数值进行（`0.1.10` 比 `0.1.9` 新）。已安装版本比最新发布版本更新（开发版或本地构建）时报告为 `up_to_date`，不会建议降级。已安装版本取自打包的 `release.json`，源码检出目录中取自 `pyproject.toml`。
- 只有存在更新版本时才会在启动时打印一行，例如 `A new MLTF version is available: 0.1.3 (installed: 0.1.2). See: <release URL>`。安装在 Homebrew Cellar keg 中的会提示 `brew upgrade hojin12312/mltf/mltf`；其他安装方式只给出发布页链接，因为无法可靠识别安装方式。链接始终由经过校验的发布标签生成。
- `/status` 新增一个 `update` 对象，包含 `status`（`unknown`、`checking`、`up_to_date`、`update_available`、`unavailable`、`disabled`）、`installed_version`、`latest_version`、`update_available`、`checked_at`（Unix 秒）、`release_url`，以及在相关时的 `reason`。原有字段不变。Web UI 读取 `/status`，仅在 `update_available` 时显示一个可关闭的小提示和“View release”链接（新标签页打开）；浏览器从不直接访问 GitHub。
- 可用 `mltf serve --no-update-check` 或 `MLTF_NO_UPDATE_CHECK=1`（`1`、`true`、`yes` 或 `on`）关闭。两者任一都会禁用检查：不发起网络请求，不读取缓存，`/status` 报告 `disabled`，也不打印任何内容。直接运行 `server/server.py` 时，只有由启动器携带已安装版本启动才会检查。

升级仍需手动：先停止正在运行的 MLTF 服务器，然后通过你安装时使用的渠道升级（Homebrew 用 `brew upgrade hojin12312/mltf/mltf`；否则参照发布页）。模型数据和缓存会保留。

## 内存与 SSD 缓存

`--max-memory` 默认为 `auto`：Metal 推荐工作集减去 `max(1GiB, 2%)`，并受更小的用户限制进一步约束。更大的用户设定值不能超过自动上限。这台机器报告的推荐工作集为 120GiB，自动上限约为 117.6GiB；测量和 README 中的命令都明确使用 96GiB。

该预算涵盖引擎的 Metal 分配和由 governor 管理的内存。它不是对所有进程内存的操作系统级限制。同时还会检查主机可用内存和系统内存压力；余量过低或压力达到临界时会拒绝新的分配。主机预留为 `min(物理内存的 10%, 2GiB)`，增长准入时还会额外检查 1GiB 余量。有效值可在 `/status` 的 `memory_plan` 和 `memory_governor` 下查看。

| 选项 | 默认值 | 作用 |
|---|---|---|
| `--max-memory` | `auto` | MLTF 内存预算，例如 `48G` 或 `96G`（`G` 表示 GiB） |
| `--ssd-root` | 未设置，SSD 缓存关闭 | 可选启用的 SSD 缓存，位于绝对路径的私有 0700 目录 |
| `--ssd-quota-bytes` | 启用时为 1,073,741,824（1GiB） | 缓存配额，正整数字节数 |
| `--ssd-free-floor-bytes` | 启用时为 8,589,934,592（8GiB） | 文件系统剩余空间下限，非负整数字节数 |

SSD 缓存关闭时，默认的缓存磁盘占用为零；模型权重和下载使用另外的磁盘空间。SSD 字节限制参数需要同时指定 `--ssd-root`。若要设置 20GiB 缓存配额和 8GiB 剩余空间下限：

```sh
mltf serve --model local/Qwen3.8-27B-q8c --model-path "$HOME/mltf-models/q8c" --max-context 256K --max-memory 96G --ssd-root "$HOME/mltf-cache" --ssd-quota-bytes 21474836480 --ssd-free-floor-bytes 8589934592
```

该目录必须满足私有权限检查。`/status` 在 `storage` 下报告存储、恢复和配额状态。
