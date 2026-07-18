# RT-Cache Mix View 使用说明

## 概述

Mix View 版本使用 **第三人称视角 + 肘部/手腕视角** 两个摄像头的图像来构建向量数据库。

### 嵌入向量结构 (4352 维)

| 组件 | 维度 | 来源 |
|------|------|------|
| Third-person DINOv2 | 1024 | 第三人称视角 |
| Third-person SigLIP | 1152 | 第三人称视角 |
| Wrist DINOv2 | 1024 | 肘部/手腕视角 |
| Wrist SigLIP | 1152 | 肘部/手腕视角 |
| **总计** | **4352** | |

## 文件结构

```
scripts/
├── embedding/
│   └── embedding_server_mix.py    # Mix 嵌入服务器 (端口 9021)
├── data_processing/
│   └── process_libero_goal_mix.py # 数据集处理脚本
├── retrieval/
│   ├── retrieval_libero_goal_mix.py      # Mix 检索服务器
│   └── start_libero_goal_retrieval_mix.sh # 启动脚本
└── README_MIX.md                  # 本文档
```

## 快速开始

### Step 1: 启动 Mix 嵌入服务器

```bash
# 激活环境
conda activate rt-mzh

# 启动 Mix 嵌入服务器 (端口 9021)
cd /path/to/rtcache
python scripts/embedding/embedding_server_mix.py --port 9021 --device cuda:0
```

服务器启动后，可以访问:
- API 文档: http://localhost:9021/docs
- 健康检查: http://localhost:9021/health

### Step 2: 处理数据集并构建向量数据库

#### 处理单个数据集

```bash
# 处理 LIBERO-Goal
python scripts/data_processing/process_libero_goal_mix.py \
    --dataset_type goal \
    --clear_db \
    --backup \
    --backup_name mix_base
```

#### 处理所有四个数据集

```bash
# 清空数据库，处理所有数据集，并备份
python scripts/data_processing/process_libero_goal_mix.py \
    --process_all \
    --clear_all \
    --backup \
    --backup_name mix_base
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset_type` | 数据集类型: goal, 10, object, spatial | goal |
| `--process_all` | 处理所有四个数据集 | false |
| `--clear_db` | 清空当前数据集的 mix collections | false |
| `--clear_all` | 清空所有 mix collections | false |
| `--backup` | 处理完成后备份数据库 | false |
| `--backup_name` | 备份文件夹名称 | mix_base |
| `--max_episodes` | 最大处理 episode 数 (-1=全部) | -1 |
| `--batch_size` | 批量插入大小 | 50 |
| `--embedding_server_url` | Mix 嵌入服务器 URL | http://127.0.0.1:9021/predict |

### Step 3: 启动 Mix 检索服务器

```bash
# 方式 1: 使用启动脚本 (推荐)
cd /path/to/rtcache/scripts/retrieval
./start_libero_goal_retrieval_mix.sh --dataset-types goal

# 方式 2: 直接运行 Python
python retrieval_libero_goal_mix.py \
    --port 5003 \
    --embedding-url http://127.0.0.1:9021/predict \
    --dataset-types goal
```

#### 启动参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host` | 服务器地址 | 0.0.0.0 |
| `--port` | 服务器端口 | 5003 |
| `--embedding-url` | Mix 嵌入服务器 URL | http://127.0.0.1:9021/predict |
| `--dataset-types` | 数据集类型 (逗号分隔或 "all") | goal |
| `--skip-restore` | 跳过数据库恢复 | false |

## API 使用

### 检索接口

**端点**: `POST /pipeline`

**请求参数** (multipart/form-data):

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `third_person_image` | file | ✓ | 第三人称视角图像 |
| `wrist_image` | file | ✓ | 肘部/手腕视角图像 |
| `instruction` | string | ✓ | 任务描述文本 |
| `dataset_type` | string | 可选 | 数据集类型 (goal/10/object/spatial) |

**示例 (Python)**:

```python
import requests

url = "http://localhost:5003/pipeline"

# 准备两张图片
files = {
    'third_person_image': ('third_person.png', open('third_person.png', 'rb'), 'image/png'),
    'wrist_image': ('wrist.png', open('wrist.png', 'rb'), 'image/png')
}

data = {
    'instruction': 'put the butter on the bowl',
    'dataset_type': 'goal'
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Success: {result['success']}")
print(f"Trajectory: {result['rtcache_trajectory']}")
```

**示例 (cURL)**:

```bash
curl -X POST http://localhost:5003/pipeline \
    -F "third_person_image=@/path/to/third_person.png" \
    -F "wrist_image=@/path/to/wrist.png" \
    -F "instruction=put the butter on the bowl" \
    -F "dataset_type=goal"
```

**响应示例**:

```json
{
    "success": true,
    "task_id": 123,
    "collection_name": "libero_goal_mix_task_123",
    "top_score": 0.95,
    "num_results": 10,
    "rtcache_trajectory": [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  // current action
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  // next action 1
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  // next action 2
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]   // next action 3
    ],
    "averaged_trajectory": [...],
    "metadata": {
        "episode_idx": 42,
        "step_idx": 15,
        "dataset_name": "libero_goal_no_noops",
        "language_instruction": "put the butter on the bowl"
    }
}
```

### 健康检查

```bash
curl http://localhost:5003/health
```

响应:
```json
{
    "status": "healthy",
    "view_type": "mix",
    "embedding_dim": 4352,
    "collections": 10,
    "total_points": 50000
}
```

## 完整工作流程

### 1. 首次部署 (构建数据库)

```bash
# Terminal 1: 启动 Qdrant 数据库
docker run -p 6333:6333 -v /path/to/qdrant_storage:/qdrant/storage qdrant/qdrant

# Terminal 2: 启动 Mix 嵌入服务器
conda activate rt-mzh
cd /path/to/rtcache
python scripts/embedding/embedding_server_mix.py --device cuda:0

# Terminal 3: 处理数据集
conda activate rt-mzh
cd /path/to/rtcache
python scripts/data_processing/process_libero_goal_mix.py \
    --process_all \
    --clear_all \
    --backup \
    --backup_name mix_base
```

### 2. 日常使用 (从备份恢复)

```bash
# Terminal 1: 确保 Qdrant 运行中

# Terminal 2: 启动 Mix 嵌入服务器
conda activate rt-mzh
python scripts/embedding/embedding_server_mix.py --device cuda:0

# Terminal 3: 启动 Mix 检索服务器 (自动恢复备份)
cd /path/to/rtcache/scripts/retrieval
./start_libero_goal_retrieval_mix.sh --dataset-types all
```

### 3. 不恢复备份启动

```bash
# 如果数据库已有数据，跳过恢复
./start_libero_goal_retrieval_mix.sh --skip-restore --dataset-types goal
```

## 端口规划

| 服务 | 端口 | 说明 |
|------|------|------|
| Qdrant | 6333 | 向量数据库 |
| 单视图嵌入服务器 | 9020 | 原版 (仅第三人称) |
| **Mix 嵌入服务器** | **9021** | **Mix 版本** |
| 单视图检索服务器 | 5002 | 原版 |
| **Mix 检索服务器** | **5003** | **Mix 版本** |

## Collection 命名规则

- 单视图: `libero_{dataset}_task_{id}`
- **Mix 视图**: `libero_{dataset}_mix_task_{id}`

例如:
- `libero_goal_task_123` (单视图)
- `libero_goal_mix_task_123` (Mix 视图)

## 备份位置

Mix 版本备份存储在:
```
/path/to/rtcache/scripts/retrieval/qdrant_backups/
├── mix_base/                    # 默认备份目录
│   ├── libero_goal_mix_task_xxx.snapshot
│   ├── libero_10_mix_task_xxx.snapshot
│   └── ...
└── latest_mix -> mix_base       # 符号链接
```

## 注意事项

1. **嵌入服务器必须先启动**: 数据处理和检索都依赖 Mix 嵌入服务器
2. **显存需求**: OpenVLA 模型约需 15GB 显存
3. **数据集路径**: 确保 LIBERO 数据集位于正确位置
4. **两张图片必须同时提供**: Mix 检索需要第三人称和肘部两张图片

## 故障排除

### 嵌入服务器无法启动

```bash
# 检查 GPU 可用性
nvidia-smi

# 尝试使用 CPU (不推荐，很慢)
python embedding_server_mix.py --device cpu
```

### 检索失败

```bash
# 检查 Qdrant 是否运行
curl http://localhost:6333/health

# 检查 collection 是否存在
curl http://localhost:6333/collections | python -m json.tool
```

### 找不到 mix collections

确保数据处理时使用了 `process_libero_goal_mix.py` 而不是原版 `process_libero_goal.py`。
