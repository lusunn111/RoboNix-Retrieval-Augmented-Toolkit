# RT-Cache Retrieval Server for LIBERO-Goal

纯向量数据库检索服务，专门为LIBERO-Goal数据集设计。

## 特点

- **纯向量数据库**: 仅使用Qdrant，不依赖MongoDB
- **内存优化**: 所有payload预加载到内存，快速检索
- **任务匹配**: 智能匹配语言指令到对应的task collection
- **动作序列**: 返回当前动作和接下来3步的动作

## 数据库结构

数据存储在Qdrant中，每个LIBERO-Goal任务有一个独立的collection：

```
libero_goal_task_0  - "turn on the stove"
libero_goal_task_1  - "put the black bowl on top of the cabinet"
libero_goal_task_2  - "put the cream cheese in the bowl"
libero_goal_task_3  - "stack the white bowl on the plate"
libero_goal_task_4  - "put the wine bottle on top of the cabinet"
libero_goal_task_5  - "put the white bowl on the plate"
libero_goal_task_6  - "turn off the stove"
libero_goal_task_7  - "put the bowl on the stove"
libero_goal_task_8  - "put the chocolate pudding on the plate"
libero_goal_task_9  - "put the butter in the bowl"
```

每个点的payload包含：
- `dataset_name`: 数据集名称
- `episode_idx`: episode索引
- `step_idx`: step索引
- `current_action`: 当前动作 (7维向量)
- `next_actions`: 接下来3步的动作 (3x7维向量列表)
- `language_instruction`: 语言指令

## 安装

确保已安装以下依赖：

```bash
pip install flask qdrant-client torch pillow numpy requests
```

## 使用方法

### 1. 启动Qdrant数据库

```bash
cd /path/to/rtcache
./start_db.sh
```

### 2. 启动Embedding服务器

确保OpenVLA embedding服务器在运行（默认端口9020）。

### 3. 启动检索服务器

```bash
cd /path/to/rtcache/scripts/retrieval
./start_libero_goal_retrieval.sh
```

或使用自定义参数：

```bash
python retrieval_libero_goal.py \
    --host 0.0.0.0 \
    --port 5002 \
    --embedding-url http://127.0.0.1:9020/predict \
    --qdrant-host localhost \
    --qdrant-port 6333 \
    --log-level INFO
```

### 4. 测试服务

健康检查：
```bash
curl http://localhost:5002/health
```

查看统计信息：
```bash
curl http://localhost:5002/stats
```

## API接口

### POST /pipeline

主要检索接口。

**请求参数**:
- `file`: 图像文件 (multipart/form-data)
- `instruction`: 语言指令 (form field)

**返回**:
```json
{
    "success": true,
    "task_id": 0,
    "collection_name": "libero_goal_task_0",
    "top_score": 0.95,
    "num_results": 10,
    "rtcache_trajectory": [
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
    ],
    "averaged_trajectory": [...],
    "metadata": {
        "episode_idx": 0,
        "step_idx": 10,
        "dataset_name": "libero_goal_no_noops",
        "language_instruction": "turn on the stove"
    }
}
```

**Python示例**:
```python
import requests
from PIL import Image
from io import BytesIO

# 准备图像
image = Image.open("test.png")
buf = BytesIO()
image.save(buf, format='PNG')
buf.seek(0)

# 发送请求
files = {"file": ("image.png", buf, "image/png")}
data = {"instruction": "turn on the stove"}
response = requests.post("http://localhost:5002/pipeline", files=files, data=data)

# 解析结果
result = response.json()
if result['success']:
    trajectory = result['rtcache_trajectory']
    print(f"Retrieved {len(trajectory)} actions")
    print(f"First action: {trajectory[0]}")
```

### GET /health

健康检查接口。

**返回**:
```json
{
    "status": "healthy",
    "collections": 10,
    "total_points": 5000
}
```

### GET /stats

统计信息接口。

**返回**:
```json
{
    "total_points": 5000,
    "collections": {
        "libero_goal_task_0": 500,
        "libero_goal_task_1": 500,
        ...
    }
}
```

## 与SpecVLA集成

在 `run_libero_goal_AR_DB.py` 中使用：

```python
import requests
from io import BytesIO
from PIL import Image
import numpy as np

RETRIEVAL_URL = "http://127.0.0.1:5002/pipeline"

# 在推理循环中
pil_img = Image.fromarray(img)
buf = BytesIO()
pil_img.save(buf, format='PNG')
buf.seek(0)

files = {"file": ("image.png", buf, "image/png")}
data = {"instruction": task_description}

response = requests.post(RETRIEVAL_URL, files=files, data=data, timeout=30)

if response.status_code == 200:
    result = response.json()
    if result['success']:
        trajectory = np.array(result['rtcache_trajectory'])
        action = trajectory[0]  # 第一个动作
```

## 任务匹配机制

服务器使用以下策略匹配语言指令到任务：

1. **LIBERO Benchmark**: 如果可用，使用官方任务描述
2. **直接匹配**: 精确匹配指令文本
3. **模糊匹配**: 检查子串包含关系
4. **Hash Fallback**: 使用MD5 hash确保一致性（当以上都失败时）

这解决了 `WARNING - Could not match instruction` 的问题。

## 性能优化

- **Payload缓存**: 启动时加载所有payload到内存
- **批量检索**: 支持并行处理多个请求
- **向量搜索**: 仅在Qdrant中搜索向量，不需要额外的数据库查询

## 故障排除

### 问题: "Collection not found"

确保已运行 `process_libero_goal.py` 创建所有collections。

### 问题: "Could not match instruction to task"

检查日志查看使用的task_id。如果是hash-based，考虑更新TaskMatcher中的默认任务列表。

### 问题: "Failed to generate embedding"

确保embedding服务器正在运行：
```bash
curl http://127.0.0.1:9020/health
```

## 目录结构

```
rtcache/scripts/retrieval/
├── retrieval_libero_goal.py          # 主检索服务器
├── start_libero_goal_retrieval.sh    # 启动脚本
└── README_LIBERO_GOAL.md             # 本文档
```

## 配置说明

主要配置项在 `RetrievalConfig` 类中：

```python
class RetrievalConfig:
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 5002
    EMBEDDING_URL = "http://127.0.0.1:9020/predict"
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    TASK_SUITE_NAME = "libero_goal"
    NUM_TASKS = 10
    TOP_K = 10
    SIMILARITY_THRESHOLD = 0.5
```

可以通过命令行参数覆盖这些配置。
