# MMRebuttal 项目说明

这个目录下主要放了两个相关项目：

- `openpi/`：Physical Intelligence 的 openpi 代码库，包含 pi0、pi0-FAST、pi05 等 VLA 模型的 JAX/PyTorch 实现、训练、推理服务和机器人/仿真示例。
- `realtime-vla-flash/`：基于 openpi 改出来的 Realtime-VLA FLASH。它保留了 openpi 的主体结构，并新增了 diffusion-based VLA 的 speculative inference、draft head、Triton 推理 runtime、LIBERO 评测客户端等代码。

简单理解：`openpi/` 是基础 VLA 框架；`realtime-vla-flash/` 是在 openpi 上做实时加速和投机推理实验的版本。

## 环境安装

两个项目都用 `uv` 管理依赖，Python 要求是 3.11。进入对应目录后安装：

```bash
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

或者：

```bash
cd realtime-vla-flash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

`GIT_LFS_SKIP_SMUDGE=1` 是为了避免 LeRobot 依赖拉取时触发不必要的大文件下载。模型运行通常需要 NVIDIA GPU；FLASH 的 Triton backend 需要 CUDA，`realtime-vla-flash/pyproject.toml` 里固定了 `torch==2.7.1`、`triton==3.3.1`、`jax[cuda12]==0.5.3`。

## openpi 代码组织

核心目录：

- `src/openpi/models/`：JAX/Flax 模型实现。重点看 `pi0.py`、`pi0_fast.py`、`pi0_config.py`、`model.py`、`gemma.py`、`siglip.py`、`vit.py`。
- `src/openpi/models_pytorch/`：PyTorch 模型实现。重点看 `pi0_pytorch.py`、`gemma_pytorch.py`、`preprocessing_pytorch.py`。
- `src/openpi/training/`：训练配置、数据加载、checkpoint、optimizer、sharding。最重要的是 `config.py`，所有 config 名称都在 `_CONFIGS` 里注册，例如 `pi0_libero`、`pi05_libero`、`pi0_fast_droid`、`pi0_aloha_sim`。
- `src/openpi/policies/`：环境和模型之间的输入/输出适配。`libero_policy.py`、`droid_policy.py`、`aloha_policy.py` 定义不同机器人/benchmark 的 observation/action key 如何映射到模型格式。
- `src/openpi/transforms.py`：通用数据 transform，包括图片 resize、prompt tokenization、state/action padding、normalize/unnormalize 等。
- `src/openpi/serving/websocket_policy_server.py`：websocket policy server，把 policy 包成远程推理服务。
- `packages/openpi-client/`：轻量客户端包，机器人侧或评测侧通过 websocket 调服务器。
- `scripts/`：命令行入口。常用的是 `train.py`、`train_pytorch.py`、`serve_policy.py`、`compute_norm_stats.py`。
- `examples/`：具体平台示例，包括 `libero/`、`droid/`、`aloha_sim/`、`aloha_real/`、`simple_client/`。

openpi 的主链路是：

1. `src/openpi/training/config.py` 选定 `TrainConfig`。
2. `policy_config.create_trained_policy()` 根据 config 和 checkpoint 创建模型与 transforms。
3. `Policy.infer()` 做输入 transform、模型采样、输出 transform。
4. `scripts/serve_policy.py` 把这个 policy 挂到 websocket server。
5. 客户端用 `openpi-client` 发 observation，拿回 action chunk。

## openpi 怎么启动

### 启动预训练 policy server

在 `openpi/` 下：

```bash
uv run scripts/serve_policy.py --env LIBERO
```

也可以显式指定 config 和 checkpoint：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

默认端口是 `8000`，可以加 `--port 8001` 改端口。

### 用简单客户端测试服务

开两个终端：

```bash
cd openpi
uv run scripts/serve_policy.py --env DROID
```

```bash
cd openpi
uv run examples/simple_client/main.py --env DROID
```

真实机器人或评测代码中，本质上也是构造 observation dict，然后用：

```python
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
out = client.infer({
    "observation/image": image,
    "observation/wrist_image": wrist_image,
    "observation/state": state,
    "prompt": instruction,
})
actions = out["actions"]
```

### 训练 / 微调

先算 normalization stats：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

JAX 训练：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero \
  --exp-name=my_experiment \
  --overwrite
```

PyTorch 训练：

```bash
uv run scripts/train_pytorch.py pi05_libero --exp_name my_experiment
```

多卡 PyTorch：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_pytorch.py pi05_libero --exp_name my_experiment
```

训练输出默认在 `checkpoints/<config>/<exp_name>/<step>/`。

## realtime-vla-flash 代码组织

`realtime-vla-flash/` 和 `openpi/` 大部分结构相同，但关键新增/修改集中在以下位置：

- `README.md`：FLASH 项目自己的 quick start。
- `README_OPENPI.md`：保留的 openpi 原始说明。
- `docs/flash/`：项目页面静态资源。
- `scripts/spec/`：FLASH/speculative inference 的主要脚本。
- `scripts/spec/spec_serve_policy.py`：FLASH policy server 入口。支持 `compiled` 和 `triton` backend，是服务端最关键的文件。
- `scripts/spec/spec_client_libero.py`：LIBERO 评测客户端，连接 server，运行任务并记录 episode/inference trace。
- `scripts/spec/enc_cache.py`：从 LeRobot/LIBERO 数据中构建 encoder/prefix cache，用于训练 draft head。
- `scripts/spec/spec_draft_train.py`：训练 draft head。
- `scripts/spec/pi0_benchmark.py`：本地推理耗时 benchmark。
- `scripts/spec/exp/run_sweep.py`：批量实验/参数 sweep。
- `scripts/spec/triton/convert_for_triton.py`：把 base pi0 checkpoint 和 draft checkpoint 转成 Triton runtime 使用的权重布局。
- `scripts/spec/triton/triton_pi0_runtime.py`：Triton runtime 的 Python 封装，负责加载权重、prompt cache、runtime pool、spec verify 调度。
- `scripts/spec/triton/pi0_infer.py`、`pi0_spec_infer.py`：Triton kernel/runtime 侧的具体推理实现。
- `src/openpi/models_pytorch/spec_pi0_pytorch.py`：PyTorch speculative pi0 实现，包括 draft、verify、prefix acceptance、fallback 逻辑。
- `src/openpi/models_pytorch/draft.py`：`DraftChunkHead`，一个轻量 Gemma decoder block 风格的 draft action head。
- `src/openpi/models_pytorch/pi0_pytorch.py`、`gemma_pytorch.py`：相对 openpi 有修改，用来支持 FLASH 的 cache/分阶段推理。
- `packages/openpi-client/src/openpi_client/websocket_client_policy.py`：客户端侧也有改动，用于透传 timing、trace、accepted prefix 等额外字段。

FLASH 的主链路是：

1. 用 base pi0 产生 encoder/prefix cache：`enc_cache.py`。
2. 用 cache 训练 draft head：`spec_draft_train.py`。
3. 把 base checkpoint 和 draft checkpoint 转成 Triton 权重：`convert_for_triton.py`。
4. `spec_serve_policy.py` 启动服务。每轮先尝试 draft 预测动作，再用 verifier 检查 prefix，接受一段动作；失败或策略要求时回退 full pipeline。
5. `spec_client_libero.py` 连接服务端，在 LIBERO 中执行 action chunk，并统计 latency、accepted prefix、route type 等。

## realtime-vla-flash 怎么启动

### 转换权重到 Triton 布局

在 `realtime-vla-flash/` 下：

```bash
uv run scripts/spec/triton/convert_for_triton.py \
  --mode base \
  --jax-path /path/to/jax/checkpoint \
  --output converted/base
```

```bash
uv run scripts/spec/triton/convert_for_triton.py \
  --mode draft \
  --draft-ckpt /path/to/draft_model.pt \
  --output converted/draft
```

`converted/base` 是 base pi0 的 Triton 权重目录，`converted/draft` 是 draft head 的 Triton 权重目录。

### 启动 FLASH policy server

```bash
uv run scripts/spec/spec_serve_policy.py \
  --config pi0_libero \
  --base-triton-path converted/base \
  --draft-triton-path converted/draft \
  --task-suite-name libero_goal \
  --backend triton
```

常用参数：

- `--port`：服务端口，默认 `8000`。
- `--pytorch-device cuda:0`：指定 GPU。
- `--max-exec-steps 12`：客户端每次最多执行多少步 action。
- `--t-list 0.10 0.05`：spec verify 使用的 diffusion 时间点。
- `--tau-radius 0.3`：draft 和 verifier 距离阈值。
- `--force-full-each-round`：强制每轮 full pipeline，通常用于对照。
- `--periodic-full-every-n-draft-rounds`：每 N 个 draft round 插一次 full round。

### 启动 LIBERO 客户端评测

另开终端：

```bash
uv run scripts/spec/spec_client_libero.py \
  --host 127.0.0.1 \
  --port 8000 \
  --task-suite-name libero_goal \
  --num-trials-per-task 50 \
  --replan-steps 12 \
  --video-out-path ../outputs/test
```

可以用 `--task` 限制任务，例如：

```bash
uv run scripts/spec/spec_client_libero.py \
  --task-suite-name libero_goal \
  --task 0-3 \
  --num-trials-per-task 5
```

输出会写到 `video_out_path` 下，包括 manifest、episode log、trace 和视频。本项目后续实验统一把 `video_out_path` 指到 `/path/to/MMRebuttal/outputs` 下面，方便统计分析。

### 训练 draft head

先构建 cache：

```bash
uv run scripts/spec/enc_cache.py \
  --config pi0_libero \
  --checkpoint-dir /path/to/pi0_libero_torch \
  --task-suite-name libero_goal \
  --output-dir /tmp/spec_train/libero_goal_cache
```

再训练 draft：

```bash
uv run scripts/spec/spec_draft_train.py \
  --cache-dir /tmp/spec_train/libero_goal_cache \
  --output draft_model_goal_torch.pt
```

训练完的 `draft_model_goal_torch.pt` 可以继续用 `convert_for_triton.py --mode draft` 转换，或者在 compiled backend 中直接作为 `--draft-checkpoint` 使用。

### benchmark

```bash
uv run python scripts/spec/pi0_benchmark.py
```

## 最关键的核心代码

如果只想读最少的代码，建议按这个顺序：

### openpi

1. `openpi/src/openpi/training/config.py`：所有训练/推理配置，理解 config 名称、模型类型、数据集、transforms、checkpoint loader。
2. `openpi/src/openpi/policies/policy_config.py`：从 config + checkpoint 创建可推理 policy。
3. `openpi/src/openpi/policies/policy.py`：`Policy.infer()` 的输入 transform、模型采样、输出 transform 流程。
4. `openpi/src/openpi/models/pi0.py` 和 `openpi/src/openpi/models_pytorch/pi0_pytorch.py`：pi0 模型本体。
5. `openpi/scripts/serve_policy.py`：服务启动入口。
6. `openpi/src/openpi/serving/websocket_policy_server.py` 和 `openpi/packages/openpi-client/`：远程推理通信。
7. `openpi/examples/libero/main.py`、`openpi/examples/droid/main.py`：具体环境如何构造 observation 并消费 actions。

### realtime-vla-flash

1. `realtime-vla-flash/scripts/spec/spec_serve_policy.py`：FLASH server 总入口，包含 backend 选择、policy 封装、runtime 调度。
2. `realtime-vla-flash/src/openpi/models_pytorch/spec_pi0_pytorch.py`：speculative 推理核心逻辑，包括 draft/full round、prefix acceptance、gripper verify、fallback。
3. `realtime-vla-flash/src/openpi/models_pytorch/draft.py`：draft head 网络结构。
4. `realtime-vla-flash/scripts/spec/triton/triton_pi0_runtime.py`：Triton runtime pool、prompt cache、draft/verify runtime 封装。
5. `realtime-vla-flash/scripts/spec/triton/convert_for_triton.py`：权重格式转换，调试 checkpoint 不匹配时优先看这里。
6. `realtime-vla-flash/scripts/spec/spec_client_libero.py`：LIBERO client、action chunk 执行、metrics/trace 记录。
7. `realtime-vla-flash/scripts/spec/enc_cache.py` 和 `spec_draft_train.py`：draft 训练数据和训练流程。

## 两个项目的关系和差异

`realtime-vla-flash/` 不是一个完全独立的新框架，而是 openpi 的 fork/扩展。它复用了 openpi 的模型配置、policy transform、websocket serving、LIBERO/DROID/ALOHA 示例等，同时增加了以下能力：

- 新增 `scripts/spec/` 实验和服务脚本。
- 新增 `DraftChunkHead` 和 `SpecPI0Pytorch`。
- 新增 Triton 权重转换和 Triton runtime。
- 修改 PyTorch pi0/gemma 实现，以支持分阶段推理、cache 和 speculative verify。
- 修改 websocket client/server/policy，使响应里能带 timing、accepted prefix、trace 等信息。
- 增加 UR5 policy、FLASH 项目页和 benchmark/sweep 工具。

所以日常开发时：

- 改基础模型、数据 transform、普通训练逻辑：优先看两个项目共有的 `src/openpi/` 结构。
- 改 FLASH 加速、draft、Triton、LIBERO 评测：主要看 `realtime-vla-flash/scripts/spec/` 和 `realtime-vla-flash/src/openpi/models_pytorch/spec_pi0_pytorch.py`。
- 新增一个机器人/benchmark：通常要加 `policies/*_policy.py`、`training/config.py` 里的 `DataConfig/TrainConfig`，以及对应 `examples/` 或 client 脚本。

## 常见排查点

- 找不到 config：检查 `src/openpi/training/config.py` 末尾 `_CONFIGS` 是否注册了对应 `name`。
- checkpoint 加载失败：确认 checkpoint 是 JAX `params/` 目录还是 PyTorch `model.safetensors`；`policy_config.py` 会用 `model.safetensors` 判断 PyTorch checkpoint。
- norm stats 缺失：检查 checkpoint 下 `assets/<asset_id>/`，或者 config 中 `AssetsConfig` 指向的位置。
- websocket 连不上：确认 server 的 `--host`、`--port`，客户端 host 不要在远程机器上误用 `0.0.0.0`。
- FLASH Triton 启动失败：确认 `--base-triton-path`、`--draft-triton-path` 是转换后的路径，不是原始 checkpoint；确认 CUDA/Triton 环境可用。
- LIBERO 图像方向不对：`spec_client_libero.py` 中对 `agentview_image` 和 `robot0_eye_in_hand_image` 做了 180 度旋转，这是为了对齐训练预处理。

## FLASH LIBERO 复现记录

当前已在 `realtime-vla-flash/` 跑通一次最小 LIBERO smoke test：

- suite/task：`libero_goal` task `0`
- episode 数：`1`
- 结果：`success=true`
- 任务描述：`open the middle drawer of the cabinet`
- 输出目录：`realtime-vla-flash/data/flash_libero_smoke/smoke_goal_task0/`
- episode log：`realtime-vla-flash/data/flash_libero_smoke/smoke_goal_task0/episode_log.json`
- 视频：`episodes/task00_ep000_open_the_middle_drawer_of_the_cabinet_success/rollout_success.mp4`

已准备好的关键路径：

- base checkpoint：`models/openpi/openpi-assets/checkpoints/pi0_libero/pi0_libero`
- Triton base artifact：`models/realtime-vla-flash/triton/pi0_libero_base/base_weights.pkl`
- Triton draft artifact：`models/realtime-vla-flash/triton/draft_libero_goal/draft_triton.pkl`
- LIBERO client venv：`realtime-vla-flash/examples/libero/.venv`

复跑 server：

```bash
cd /path/to/MMRebuttal/realtime-vla-flash

uv run scripts/spec/spec_serve_policy.py \
  --config pi0_libero \
  --base-triton-path /path/to/MMRebuttal/models/realtime-vla-flash/triton/pi0_libero_base \
  --draft-triton-path /path/to/MMRebuttal/models/realtime-vla-flash/triton/draft_libero_goal \
  --task-suite-name libero_goal \
  --backend triton \
  --pytorch-device cuda:0 \
  --max-exec-steps 12 \
  --port 8000
```

复跑 client：

```bash
cd /path/to/MMRebuttal/realtime-vla-flash

PYTHONPATH=$PWD:$PWD/src:$PWD/third_party/libero \
MUJOCO_GL=egl \
examples/libero/.venv/bin/python scripts/spec/spec_client_libero.py \
  --host 127.0.0.1 \
  --port 8000 \
  --task-suite-name libero_goal \
  --task 0 \
  --num-trials-per-task 1 \
  --replan-steps 12 \
  --video-out-path data/flash_libero_smoke \
  --run-name smoke_goal_task0
```

注意：client 退出时观察到过一次 EGL context cleanup 的 ignored exception，包含 `libGLU.so.0` / `EGL_NOT_INITIALIZED`。它发生在 episode 成功之后，不影响本次复现结果；如果后续渲染初始化阶段失败，再尝试 `MUJOCO_GL=glx`。

## RT-Cache 文档入口

RT-Cache mix view 数据库检索链路单独记录在 [docs/rtcache_mix_retrieval.md](docs/rtcache_mix_retrieval.md)。该文档用于后续追踪“用数据库检索替换 FLASH draft model”的实现。

新的 FLASH 视觉编码内存索引、纯 database draft policy 和 retrieval-draft spike 记录在 [docs/flash_memory_retrieval.md](docs/flash_memory_retrieval.md)。当前纯 database draft smoke 输出在 `outputs/database_draft_smoke/libero_goal_task0/`，视频路径为 `outputs/database_draft_smoke/libero_goal_task0/episodes/task00_ep000_open_the_middle_drawer_of_the_cabinet_success/rollout_success.mp4`。
