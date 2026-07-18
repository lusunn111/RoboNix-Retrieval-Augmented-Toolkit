# RT-Cache Mix View 检索链路

> Superseded: 当前实际使用的数据库链路已经切到 `docs/flash_memory_retrieval.md`，不再使用旧 RT-Cache/Qdrant mix 库。

本文档记录 `/path/to/rtcache/scripts` 中与 mix view 数据库检索有关的实现，目标是为后续把 FLASH draft model 替换成数据库检索提供依据。

当前新的数据入口改为 FLASH 在线采集得到的 action chunk 数据：

- 采集目录：`/path/to/MMRebuttal/dataset/flash_episodes`
- 采集开关：`scripts/spec/spec_client_libero.py --record-flash-dataset`
- 默认关闭；不开启时不写 dataset。
- 样本前两级目录按 suite/task 组织：`{suite}/task_{task_id:02d}/...`

旧 RT-Cache LIBERO 数据集已不作为当前数据源使用。后续数据库应从 `dataset/flash_episodes` 构建，而不是从原 modified LIBERO RLDS 直接构建。

曾经把运行数据迁移到 MMRebuttal：

- Qdrant 持久化目录：`/path/to/MMRebuttal/database/qdrant`
- Qdrant 二进制：`/path/to/MMRebuttal/database/bin/qdrant`
- Qdrant 配置：`/path/to/MMRebuttal/database/qdrant_config.yaml`

注意：`dataset/libero` 这份旧数据已经按当前计划删除。

## 当前数据库

线上检索目标是本机 Qdrant：

- Host/port：`localhost:6333`
- 迁移后配置：`/path/to/MMRebuttal/database/qdrant_config.yaml`
- 本地持久化目录：`/path/to/MMRebuttal/database/qdrant`
- 原配置来源：`/path/to/rtcache/.env` 和 `config/rt_cache_config.py`

启动迁移后的 Qdrant：

```bash
cd /path/to/MMRebuttal
database/bin/qdrant --config-path database/qdrant_config.yaml
```

mix view collections 使用下面的命名：

```text
libero_{goal|10|object|spatial}_mix_task_{hash_id}
```

迁移后已验证的 mix collection 覆盖情况：

- `libero_goal_mix_task_*`：10 个
- `libero_10_mix_task_*`：10 个
- `libero_object_mix_task_*`：9 个
- `libero_spatial_mix_task_*`：10 个

这些 collection 的向量配置是 4352 维、Cosine 距离。注意 `{hash_id}` 不是 LIBERO 官方 0-9 task id，而是语言指令的 hash id。

metadata smoke test 已通过：

```bash
cd /path/to/MMRebuttal

uv run --with qdrant-client --with pillow --with numpy --with socksio \
  python database/scripts/test_rtcache_retriever.py --dataset-type goal
```

样例结果：`libero_goal_mix_task_128` 为 `green`，`points_count=7157`，样例 instruction 为 `open the top drawer and put the bowl inside`。

## 向量表征

mix view 版本对两张图像分别编码：

- third-person image：第三人称视角，对应 LIBERO client 中的 `agentview_image`
- wrist image：肘部/手腕视角，对应 LIBERO client 中的 `robot0_eye_in_hand_image`

每个视角使用 OpenVLA 的 vision backbone 抽取两类特征：

| 组件 | 维度 |
| --- | ---: |
| Third-person DINOv2 | 1024 |
| Third-person SigLIP | 1152 |
| Wrist DINOv2 | 1024 |
| Wrist SigLIP | 1152 |
| Total | 4352 |

实现位置：

- `scripts/embedding/embedding_server_mix.py`
- API：`POST /predict`
- 默认端口：`9021`
- 输出字段：`mix_features`

单视角 embedding 是 `DINOv2 + SigLIP = 2176` 维；mix embedding 是两个单视角特征拼接得到 4352 维。拼接顺序是：

```text
[third_person_dino, third_person_siglip, wrist_dino, wrist_siglip]
```

## FLASH Episode 数据集采集

当前先不构建新 Qdrant 数据库，而是先把 FLASH 实际输入输出存成离线数据集。采集逻辑只加在 LIBERO client 中：

```text
/path/to/MMRebuttal/realtime-vla-flash/scripts/spec/spec_client_libero.py
```

新增开关：

```bash
--record-flash-dataset
--record-flash-dataset-format episode_npz
--flash-dataset-out-path ../dataset/flash_episodes
--record-max-episodes-per-task 40
--record-successful-episodes-only
--initial-state-jitter-std 0.0005
--record-stop-task-when-full
```

不开 `--record-flash-dataset` 时，不会写任何 dataset 文件。默认只写 `success=True` 的正确轨迹；失败 rollout 会保留在 run log 里，但不会进入 dataset。

目录结构：

```text
dataset/flash_episodes/
  manifest.json
  libero_goal/
    task_00/
      index.jsonl
      <run_id>_episode_0000.npz
      <run_id>_episode_0001.npz
```

每个 `.npz` 是一个完整 episode。文件里按 inference call 存数组，`action_chunks` 是 FLASH client 收到的完整 chunk：

```text
third_person_images  uint8   [N, 224, 224, 3]
wrist_images         uint8   [N, 224, 224, 3]
states               float32 [N, 8]
initial_state        float32 [S]
action_chunks        float32 [N, 50, 7]
infer_ids            int32   [N]
frame_idxs           int32   [N]
env_steps            int32   [N]
route_types          str     [N]
accepted_prefix_lens int32   [N]
chunk_exec_lens      int32   [N]
executed_actions     float32 [M, 7]
executed_infer_ids   int32   [M]
executed_action_offsets int32 [M]
executed_frame_idxs  int32   [M]
executed_env_steps   int32   [M]
executed_rewards     float32 [M]
executed_done_after_step bool [M]
executed_route_types str     [M]
infer_payloads_json  str     [N]
infer_records_json   str     scalar JSON list
trace_records_json   str     scalar JSON list
metadata_json        str     scalar JSON object
action_space = flash_client_output_env_action_chunk_7
```

这不是单步动作，而是每个 inference call 对应一个 50 步动作序列。client 执行时仍然只取前 `replan_steps` / `accepted_prefix_len` 步，当前 smoke test 中通常是前 12 步。

样本 payload 语义：

- `third_person_images`：client 侧已旋转 180 度、resize/pad 后的第三人称图序列。
- `wrist_images`：client 侧已旋转 180 度、resize/pad 后的腕部图序列。
- `states`：发送给 FLASH server 的 8 维 LIBERO state 序列。
- `action_chunks`：FLASH server 返回并经过 output transform 后的 `[N, 50, 7]` action chunk。
- `executed_actions`：环境中实际执行过的动作轨迹，只来自成功 episode。
- `executed_*`：把执行动作映射回来源 `action_chunks[infer_id, action_offset]`，用于检查轨迹是否和 chunk proposal 对齐。
- `infer_payloads_json`：每次 inference 的 route type、accepted prefix、timing 等 payload；为避免重复大数组，里面用 `action_chunk_ref` 指向 `action_chunks[i]`。
- `trace_records_json`：episode 内实际执行动作、reward、done 等逐步 trace。
- `metadata_json`：suite、task、episode、success、初始状态扰动参数、数组 schema 等 episode 级元数据。
- `index.jsonl`：每个 task 一个索引，方便后续建库脚本顺序扫描。

采集命令示例：

```bash
python scripts/spec/spec_client_libero.py \
  --host 127.0.0.1 \
  --port 8000 \
  --task-suite-name libero_goal \
  --task 0 \
  --num-trials-per-task 5 \
  --replan-steps 12 \
  --video-out-path data/flash_dataset_collect \
  --run-name collect_goal_task0 \
  --record-flash-dataset \
  --record-flash-dataset-format episode_npz \
  --flash-dataset-out-path ../dataset/flash_episodes \
  --record-max-episodes-per-task 40 \
  --record-successful-episodes-only \
  --initial-state-jitter-std 0.0005
```

也可以直接使用一键脚本采四个 suite：

```bash
cd /path/to/MMRebuttal/realtime-vla-flash

TRIALS_PER_TASK=80 \
MAX_EPISODES_PER_TASK=40 \
DATASET_OUT=../dataset/flash_episodes \
VIDEO_OUT=data/flash_episode_collect \
scripts/spec/collect_flash_chunks.sh
```

脚本默认最多尝试 80 个 rollout，但只保存成功 episode，并在每个 task 达到 40 个成功 episode 后停止。如果只做单任务 smoke，可以加 `TASK_SPEC=0`；不设置时默认跑 suite 内 10 个 task。

这个脚本默认启动 FLASH server 的 full-teacher 路径：

```text
--force-full-each-round
```

这样采集的 50-step chunks 来自 full policy，而不是当前 draft head。

后续新数据库的 vector 建议从 `.npz` 中的 `third_person_images[i] + wrist_images[i]` 通过 FLASH/PaliGemma 视觉编码器生成；payload 则读取同一个 episode 文件里的 `action_chunks[i]`、`infer_payloads_json[i]` 和 episode 级 `metadata_json`，保留完整 50 步 chunk。做轨迹级检查时使用 `executed_actions` 和 `executed_*` 字段。

当前已完成一个最小采集 smoke：

```text
dataset/flash_episodes/libero_goal/task_00/<run_id>_episode_0000.npz
```

验证目标：`action_chunks` 为 `float32`，shape 是 `[N, 50, 7]`，并且 payload 中 `route_type=full`、`accepted_prefix_len` 可追踪。

## MMRebuttal 本地脚本

新代码放在：

```text
/path/to/MMRebuttal/database/scripts
```

文件职责：

- `mix_embedder.py`：本地加载 OpenVLA vision backbone，不走 HTTP，直接输出 4352 维 mix embedding。
- `rtcache_retriever.py`：提供 `RTCacheRetriever` 类，FLASH 后续应直接 import 这个类替换 draft proposal。
- `build_libero_mix_db.py`：从 migrated modified LIBERO RLDS 构建或增量更新 Qdrant collections。
- `test_rtcache_retriever.py`：metadata smoke test；加 `--run-embedding` 后加载 OpenVLA 做端到端检索。

端到端检索测试命令：

```bash
cd /path/to/MMRebuttal

uv run --with qdrant-client --with pillow --with numpy --with socksio \
  python database/scripts/test_rtcache_retriever.py \
  --dataset-type goal \
  --run-embedding \
  --device cuda:0
```

建库小样本命令：

```bash
cd /path/to/MMRebuttal

uv run --with qdrant-client --with tensorflow-datasets --with tensorflow --with pillow --with numpy --with socksio \
  python database/scripts/build_libero_mix_db.py \
  --dataset-type goal \
  --max-episodes 2
```

默认不清空已有库；只有显式加 `--clear` 才会删除并重建对应 collection。

## 数据库如何构建

主脚本是：

```text
/path/to/rtcache/scripts/data_processing/process_libero_goal_mix.py
```

它读取 modified LIBERO RLDS 数据集，原默认根路径是：

```text
/path/to/rtcache/libero/datasets--openvla--modified_libero_rlds/snapshots/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551
```

迁移后对应根路径是：

```text
/path/to/MMRebuttal/dataset/libero/datasets--openvla--modified_libero_rlds/snapshots/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551
```

支持的数据集类型：

- `goal` -> `libero_goal_no_noops`
- `10` -> `libero_10_no_noops`
- `object` -> `libero_object_no_noops`
- `spatial` -> `libero_spatial_no_noops`

构建流程：

1. 启动 Qdrant。
2. 启动 mix embedding server。
3. `process_libero_goal_mix.py` 逐 episode 读取 `steps`。
4. 每一步读取 `step["observation"]["image"]` 和 `step["observation"]["wrist_image"]`。
5. 请求 `http://127.0.0.1:9021/predict` 生成 4352 维向量。MMRebuttal 新脚本改成本地类调用，不再依赖 embedding server。
6. 用 `md5(language_instruction) % 1001` 得到 task id。
7. 写入 `libero_{dataset}_mix_task_{hash_id}` collection。

典型命令：

```bash
cd /path/to/rtcache

python scripts/embedding/embedding_server_mix.py \
  --port 9021 \
  --device cuda:0
```

```bash
cd /path/to/rtcache

python scripts/data_processing/process_libero_goal_mix.py \
  --process_all \
  --clear_all \
  --backup \
  --backup_name mix_base
```

原单个 Qdrant point 的 payload 包含：

- `dataset_name`
- `episode_idx`
- `step_idx`
- `current_action`
- `next_actions`
- `language_instruction`

`current_action` 是当前 7 维动作，`next_actions` 默认保存后续 3 步动作；末尾不足时用 `[0.0] * 7` 补齐。

MMRebuttal 新建库脚本会保留这些字段，并额外写入：

- `action_chunk`
- `source`
- `suite`

## 数据如何采集

当前 mix 版数据库不是由在线 robot server 逐步采集得到，而是由 `process_libero_goal_mix.py` 从 modified LIBERO RLDS 离线数据集中抽取 observation/action 建库。

相关但不是当前主链路的脚本：

- `scripts/data_acquisition/data_collection_server.py`
- `scripts/data_acquisition/action_generators.py`

`data_collection_server.py` 是早期单视图/调试服务：

- `MODE=debug` 时保存上传图像并返回脚本化动作。
- `MODE=test` 时对单张图像生成 embedding，并在单视图 Qdrant collection 中检索。
- 默认使用 `file` 单图输入，不处理 third-person + wrist 双图。

因此后续接 FLASH 时，应以 mix 版 `embedding_server_mix.py`、`process_libero_goal_mix.py`、`retrieval_libero_goal_mix.py` 为准。

## 如何检索

原主脚本是：

```text
/path/to/rtcache/scripts/retrieval/retrieval_libero_goal_mix.py
```

推荐启动脚本：

```text
/path/to/rtcache/scripts/retrieval/start_libero_goal_retrieval_mix.sh
```

默认服务：

- retrieval server：`0.0.0.0:5003`
- embedding URL：`http://127.0.0.1:9021/predict`
- Qdrant：`localhost:6333`

启动示例：

```bash
cd /path/to/rtcache/scripts/retrieval

./start_libero_goal_retrieval_mix.sh \
  --dataset-types goal \
  --skip-restore
```

如果要服务所有数据集：

```bash
./start_libero_goal_retrieval_mix.sh \
  --dataset-types all \
  --skip-restore
```

检索 API：

```text
POST /pipeline
```

multipart/form-data 字段：

- `third_person_image`：第三人称图像，必填
- `wrist_image`：肘部/手腕图像，必填
- `instruction`：任务语言指令，必填
- `dataset_type`：`goal`、`10`、`object`、`spatial`，可选

Python 请求示例：

```python
import requests

url = "http://127.0.0.1:5003/pipeline"

files = {
    "third_person_image": ("third_person.png", open("third_person.png", "rb"), "image/png"),
    "wrist_image": ("wrist.png", open("wrist.png", "rb"), "image/png"),
}
data = {
    "instruction": "open the middle drawer of the cabinet",
    "dataset_type": "goal",
}

response = requests.post(url, files=files, data=data, timeout=60)
result = response.json()
```

检索流程：

1. `retrieval_libero_goal_mix.py` 接收两张图和 instruction。
2. 调用 mix embedding server 生成 query vector。
3. 用同样的 `md5(instruction) % 1001` 得到 collection 名。
4. 在对应 Qdrant collection 中 topK 检索。
5. 从 payload 中取 top result 的 `current_action + next_actions`。
6. 返回 `rtcache_trajectory`、`averaged_trajectory` 和 metadata。

当前实现细节：

- `TOP_K=10`。
- `NUM_ACTIONS=3`，即返回当前动作加后续 3 步动作。
- `averaged_trajectory` 目前实际只平均 top 1，因为代码使用 `results[:min(1, len(results))]`。
- retrieval server 启动时会预加载已发现 collections 的 payload 到内存，向量仍在 Qdrant 中检索。

## 后续替换 FLASH Draft Model 的接入点

目标不是替换 pi0 full/verifier，而是替换 draft proposal 来源：

1. 保留 FLASH 的 base Triton runtime、full round、verify 和 prefix acceptance。
2. 将原 `DraftChunkHead` 生成 action chunk 的步骤替换为 RT-Cache retrieval。
3. 从 LIBERO observation 中取两张图：
   - `agentview_image` -> `third_person_image`
   - `robot0_eye_in_hand_image` -> `wrist_image`
4. 调用 `database/scripts/rtcache_retriever.py` 里的 `RTCacheRetriever.retrieve()`，将 `rtcache_trajectory` 作为 draft action proposal。
5. 继续交给现有 verifier 检查 prefix，成功则执行 accepted prefix，失败则 fallback full pipeline。

建议优先新增开关，例如 `--rtcache-draft`。关闭时保持原 FLASH draft model；开启时只替换 draft proposal 生成，不改 Triton verifier kernel，不破坏 full/verifier/fallback 路径。增量采集另加独立开关，例如 `--rtcache-record-rollouts`，默认关闭。

## 关键文件

- `/path/to/rtcache/scripts/README_MIX.md`
- `/path/to/rtcache/scripts/embedding/embedding_server_mix.py`
- `/path/to/rtcache/scripts/data_processing/process_libero_goal_mix.py`
- `/path/to/rtcache/scripts/retrieval/retrieval_libero_goal_mix.py`
- `/path/to/rtcache/scripts/retrieval/start_libero_goal_retrieval_mix.sh`
- `/path/to/rtcache/scripts/data_acquisition/data_collection_server.py`
- `/path/to/MMRebuttal/database/scripts/mix_embedder.py`
- `/path/to/MMRebuttal/database/scripts/rtcache_retriever.py`
- `/path/to/MMRebuttal/database/scripts/build_libero_mix_db.py`
- `/path/to/MMRebuttal/database/scripts/test_rtcache_retriever.py`
