# FLASH In-Memory Retrieval

当前数据库链路已经从旧 RT-Cache/Qdrant 切到本地内存索引：

- 数据源：`dataset/flash_episodes`
- 索引目录：`database/flash_index`
- 构建脚本：`database/scripts/build_flash_index.py`
- 检索类：`database/scripts/flash_retriever.py`
- 视觉编码器：`database/scripts/flash_vision_embedder.py`
- 纯 database policy：`database/scripts/flash_database_policy.py`
- 当前兼容 server：`database/scripts/serve_flash_database_policy.py`

旧的 `database/qdrant`、Qdrant 配置和 RT-Cache mix 脚本已删除。新检索不启动网络服务，也不在 query 时从磁盘取 payload；`FlashEpisodeRetriever` 初始化时会把 `vectors.npy`、`action_chunks.npy`、`executed_actions.npy`、`states.npy` 和 `records.jsonl` 全量加载到内存。

## Index

构建命令：

```bash
cd /path/to/MMRebuttal/realtime-vla-flash

uv run python ../database/scripts/build_flash_index.py \
  --dataset-root ../dataset/flash_episodes \
  --output-dir ../database/flash_index \
  --base-triton-path ../models/realtime-vla-flash/triton/pi0_libero_base \
  --device cuda:0 \
  --overwrite
```

当前索引结果：

```text
record_count = 21357
vector_dim = 4096
tasks = 40
index_size ~= 235M
```

向量来自 FLASH Triton base 的 PaliGemma vision tower：

```text
third-person projected token mean 2048
wrist projected token mean        2048
total                             4096
```

每条记录对应一个 inference observation，payload 保留：

- `action_chunk`: `[50, 7]`
- `executed_actions`: 实际执行 prefix，长度由 `executed_action_lens` 给出
- `state`: `[8]`
- `record`: suite、task、episode、infer id、frame/env step、accepted prefix 等元数据

## Retrieval

自检：

```bash
cd /path/to/MMRebuttal/realtime-vla-flash
uv run python ../database/scripts/test_flash_retriever.py --index-dir ../database/flash_index
```

代码入口：

```python
from database.scripts.flash_retriever import FlashEpisodeRetriever

retriever = FlashEpisodeRetriever(index_dir="database/flash_index")
result = retriever.retrieve_by_vector(query_vector, suite="libero_goal", task_id=0)
action_chunk = result["results"][0]["action_chunk"]
```

如果需要检索类内部直接编码图像：

```python
retriever = FlashEpisodeRetriever(
    index_dir="database/flash_index",
    load_embedder=True,
    device="cuda:0",
)
result = retriever.retrieve(third_person_image, wrist_image, suite="libero_goal", task_id=0)
```

## Pure Database Draft

初步实验可以不跑 FLASH learned draft、full decoder 和 verifier，只跑数据库检索出的 action chunk：

- policy 类：`database/scripts/flash_database_policy.py::FlashDatabasePolicy`
- 输入：`observation/image`、`observation/wrist_image`、`prompt`
- 输出：最近邻记录里的 `actions: [50, 7]`
- `accepted_prefix_len`：默认 `12`，和 LIBERO client 的 `--replan-steps 12` 对齐
- 实验输出统一放到 `outputs/`

当前 LIBERO client 仍在单独的 Python 3.8 venv 中，而 FLASH/Triton 视觉编码器在 uv 环境中更稳定，所以先用一个很薄的 websocket wrapper 做环境兼容。这个 wrapper 不跑 FLASH full/verifier，只调用 `FlashDatabasePolicy`：

```bash
cd /path/to/MMRebuttal/realtime-vla-flash

uv run python ../database/scripts/serve_flash_database_policy.py \
  --host 0.0.0.0 \
  --port 8020 \
  --index-dir ../database/flash_index \
  --device cuda:0 \
  --suite-name libero_goal \
  --top-k 1 \
  --max-exec-steps 12
```

LIBERO 评测端：

```bash
cd /path/to/MMRebuttal/realtime-vla-flash

source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD:$PWD/src:$PWD/third_party/libero
export MUJOCO_GL=egl

python scripts/spec/spec_client_libero.py \
  --host 127.0.0.1 \
  --port 8020 \
  --task-suite-name libero_goal \
  --task 0 \
  --num-trials-per-task 1 \
  --replan-steps 12 \
  --video-out-path ../outputs/database_draft_smoke \
  --run-name libero_goal_task0
```

已完成一次 smoke：

```text
suite/task = libero_goal / task_00
episodes   = 1
success    = true
output     = outputs/database_draft_smoke/libero_goal_task0
video      = outputs/database_draft_smoke/libero_goal_task0/episodes/task00_ep000_open_the_middle_drawer_of_the_cabinet_success/rollout_success.mp4
mean policy_time ~= 11.55 ms/infer
```

## FLASH Draft Spike

现在已经有一个 in-process spike，可以在 FLASH server 里直接把 learned draft proposal 换成本地检索结果：

- 入口：`realtime-vla-flash/scripts/spec/spec_serve_policy.py`
- 开关：`--rtcache-draft`
- 检索类：`database/scripts/flash_retriever.py::FlashEpisodeRetriever`
- 索引默认路径：`../database/flash_index`

它不是 HTTP 服务，也不走 Qdrant。server 启动时会把 index 全量加载到内存，query 时用当前 third-person + wrist 图像做 FLASH vision embedding，然后按 `suite/task` 检索最近邻的 `[50, 7]` action chunk。

需要注意：索引里的 action chunk 是 client 端 7 维环境动作；FLASH verifier 需要内部 normalized model action。`spec_serve_policy.py` 里的 `_FlashRetrievalDraftProvider` 会用当前 state 和 checkpoint norm stats 把检索到的 7 维 chunk 反变换回 `[50, action_dim]`，再作为 `x0_draft` 交给原 verifier。full path、cache snapshot、verify、fallback 逻辑保持不变。

启动 server 示例：

```bash
cd /path/to/MMRebuttal/realtime-vla-flash

uv run scripts/spec/spec_serve_policy.py \
  --config pi0_libero \
  --base-triton-path ../models/realtime-vla-flash/triton/pi0_libero_base \
  --draft-triton-path ../models/realtime-vla-flash/triton/draft_libero_goal \
  --task-suite-name libero_goal \
  --backend triton \
  --pytorch-device cuda:0 \
  --max-exec-steps 12 \
  --port 8000 \
  --rtcache-draft \
  --rtcache-index-dir ../database/flash_index \
  --rtcache-top-k 1
```

第一轮仍会走 full path，因为 verifier 需要先建立 full cache snapshot；后续 round 才会走 retrieval draft + verify。不要同时开 `--force-full-each-round`，否则 retrieval draft 不会被使用。

client smoke test 仍然用原来的 LIBERO client：

```bash
cd /path/to/MMRebuttal/realtime-vla-flash

source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD:$PWD/src:$PWD/third_party/libero
export MUJOCO_GL=egl

python scripts/spec/spec_client_libero.py \
  --host 127.0.0.1 \
  --port 8000 \
  --task-suite-name libero_goal \
  --task 0 \
  --num-trials-per-task 1 \
  --replan-steps 12 \
  --video-out-path ../outputs/flash_retrieval_smoke \
  --run-name retrieval_goal_task0
```
