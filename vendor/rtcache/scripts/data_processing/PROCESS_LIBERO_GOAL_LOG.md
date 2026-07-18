# LIBERO-Goal 数据集处理脚本开发日志

## 项目概述

开发了一个专门用于处理 LIBERO-Goal 数据集的脚本，将数据集中的轨迹数据转换为向量数据库格式，支持基于图像和指令的检索。

**脚本路径**: `rtcache/scripts/data_processing/process_libero_goal.py`

**开发日期**: 2024-12-23

## 功能特性

### 1. 数据集处理
- 支持 LIBERO-Goal 数据集（RLDS 格式）
- 自动提取图像、动作序列和语言指令
- 处理 428 个 episodes，每个 episode 包含多个 steps

### 2. 任务识别与集合管理
- **任务识别策略**:
  - 优先使用 LIBERO benchmark 进行精确匹配（如果可用）
  - Fallback: 使用 MD5 hash 对 instruction 进行哈希，取模 1001 作为 task_id
  - 确保相同 instruction 始终分配到相同的 task_id

- **集合命名**:
  - 格式: `libero_goal_task_{task_id}`
  - task_id 范围: 0-1000（使用 hash % 1001）
  - 每个 instruction 获得唯一的 task_id，避免冲突

### 3. 数据存储
- **向量存储** (Qdrant):
  - 图像特征向量: OpenVLA embeddings (2176 维)
  - 使用 Cosine 距离进行相似度搜索

- **Metadata 存储**:
  - `dataset_name`: 数据集名称（libero_goal_no_noops）
  - `episode_idx`: 当前轨迹号
  - `step_idx`: 当前帧号
  - `current_action`: 当前帧的 action（7 维数组）
  - `next_actions`: 下 3 帧的 action 列表（每个 7 维，共 3 个）
  - `language_instruction`: 任务描述文本

### 4. 批量处理
- 批量插入到 Qdrant（默认 batch_size=50）
- 自动管理内存，避免内存溢出
- 支持断点续传（通过 max_episodes 参数）

## 技术实现

### 核心组件

1. **LiberoGoalProcessor 类**
   - 主处理类，负责整个数据处理流程
   - 初始化存储连接、任务映射
   - 处理 episodes 和 steps

2. **任务匹配机制**
   ```python
   # 优先使用 LIBERO benchmark
   task_id = self._match_task_id(language_instruction)
   
   # Fallback: hash-based assignment
   if task_id is None:
       instruction_hash = int(hashlib.md5(instruction.encode('utf-8')).hexdigest(), 16)
       task_id = instruction_hash % 1001
   ```

3. **Embedding 生成**
   - 通过 Embedding Server API 生成图像特征
   - 支持 OpenVLA 模型（2176 维）
   - 自动处理图像格式转换

### 数据流程

```
Dataset (RLDS) 
  → Extract Episodes 
    → Extract Steps 
      → Extract Image + Action + Instruction
        → Generate Embedding (via API)
          → Create Qdrant Point
            → Batch Insert to Qdrant
```

## 使用方法

### 基本运行

```bash
cd /path/to/rtcache
conda activate rt-mzh
CUDA_VISIBLE_DEVICES=1 python scripts/data_processing/process_libero_goal.py
```

### 完整命令（推荐）

```bash
cd /path/to/rtcache
conda activate rt-mzh
CUDA_VISIBLE_DEVICES=1 python scripts/data_processing/process_libero_goal.py \
    --log_level INFO
```

### 自定义参数

```bash
python scripts/data_processing/process_libero_goal.py \
    --dataset_path /path/to/dataset \
    --embedding_server_url http://127.0.0.1:9020/predict \
    --qdrant_host localhost \
    --qdrant_port 6333 \
    --batch_size 50 \
    --max_episodes -1 \
    --log_level INFO
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset_path` | str | (已设置) | LIBERO-Goal 数据集路径 |
| `--embedding_server_url` | str | http://127.0.0.1:9020/predict | Embedding 服务器 URL |
| `--qdrant_host` | str | localhost | Qdrant 主机地址 |
| `--qdrant_port` | int | 6333 | Qdrant 端口 |
| `--batch_size` | int | 50 | 批量插入大小 |
| `--max_episodes` | int | -1 | 最大处理 episodes 数（-1=全部） |
| `--log_level` | str | INFO | 日志级别 (DEBUG/INFO/WARNING/ERROR) |

### 后台运行

```bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/data_processing/process_libero_goal.py \
    --log_level INFO > logs/process_libero_goal.log 2>&1 &
```

## 开发历史

### 2024-12-23: 初始开发

#### 问题 1: 任务 ID 分配不一致
- **问题**: 使用 `episode_idx % 10` 导致相同 instruction 的不同 episode 被分配到不同 task_id
- **解决**: 改为使用 instruction 的 MD5 hash 值，确保相同 instruction 始终分配到相同 task_id

#### 问题 2: Hash 冲突导致集合缺失
- **问题**: 使用 `hash % 10` 后，10 个 instruction 只创建了 7 个集合（task_id=5,7,8 未被使用）
- **原因**: Hash 分布不均匀，多个 instruction 映射到同一 task_id
- **解决**: 改为使用 `hash % 1001`，每个 instruction 获得唯一的 task_id（0-1000 范围）

#### 问题 3: 日志噪音
- **问题**: 每个 episode 都输出 warning，日志过于冗长
- **解决**: 
  - 将 warning 改为 INFO 级别（这是预期的 fallback 行为）
  - 每个 instruction 只记录一次日志

#### 问题 4: LIBERO benchmark 不可用
- **问题**: LIBERO benchmark 导入失败，无法进行精确任务匹配
- **解决**: 实现 hash-based fallback 机制，确保功能正常

### 最终实现

- ✅ 使用 `hash % 1001` 确保每个 instruction 有唯一的 task_id
- ✅ 相同 instruction 始终分配到相同 task_id（一致性保证）
- ✅ 减少日志噪音，每个 instruction 只记录一次
- ✅ 支持批量处理，自动管理内存
- ✅ 完整的错误处理和统计信息

## 数据集信息

### LIBERO-Goal 数据集
- **路径**: `/path/to/rtcache/libero/datasets--openvla--modified_libero_rlds/snapshots/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551/libero_goal_no_noops/1.0.0`
- **总 Episodes**: 428
- **唯一 Instructions**: 10 个
- **格式**: RLDS (Reinforcement Learning Dataset)

### 10 个任务指令

1. `put the bowl on the plate` → task_id=548
2. `put the wine bottle on the rack` → task_id=775
3. `open the top drawer and put the bowl inside` → task_id=128
4. `put the cream cheese in the bowl` → task_id=237
5. `put the wine bottle on top of the cabinet` → task_id=319
6. `push the plate to the front of the stove` → task_id=362
7. `put the bowl on the stove` → task_id=93
8. `put the bowl on top of the cabinet` → task_id=847
9. `open the middle drawer of the cabinet` → task_id=632
10. `turn on the stove` → task_id=806

## 输出结果

### Qdrant 集合
- 每个 instruction 对应一个集合: `libero_goal_task_{task_id}`
- 集合包含该 instruction 的所有 episodes 和 steps
- 每个 point 包含:
  - 向量: OpenVLA 图像特征 (2176 维)
  - Payload: 完整的 metadata

### 统计信息
脚本完成后会输出:
- 总 episodes 数
- 总 steps 数
- 跳过的 episodes 数
- 失败的 embeddings 数
- 每个 task_id 的 episode 分布

## 依赖要求

### Python 环境
- Python 3.10+
- Conda 环境: `rt-mzh`

### 必需服务
1. **Embedding Server** (端口 9020)
   - 提供图像 embedding 生成服务
   - 支持 OpenVLA 模型

2. **Qdrant** (端口 6333)
   - 向量数据库服务
   - 存储图像特征和 metadata

### Python 包
- `tensorflow` / `tensorflow_datasets`: 数据集加载
- `qdrant_client`: Qdrant 客户端
- `requests`: HTTP 请求
- `PIL`: 图像处理
- `numpy`: 数值计算
- `torch`: 张量处理

## 性能指标

### 处理速度
- 每个 episode 约 5-6 秒
- 总处理时间: 约 35-40 分钟（428 episodes）

### 资源使用
- GPU: 使用 device 1 (通过 CUDA_VISIBLE_DEVICES=1)
- 内存: 批量处理，自动管理
- 网络: 需要与 Embedding Server 和 Qdrant 通信

## 注意事项

1. **GPU 设备**: 必须使用 `CUDA_VISIBLE_DEVICES=1` 指定 GPU 1
2. **Conda 环境**: 必须激活 `rt-mzh` 环境
3. **服务依赖**: Embedding Server 和 Qdrant 必须提前启动
4. **数据集路径**: 确保数据集路径正确且可访问
5. **Hash 一致性**: 相同 instruction 的 hash 值在不同运行中保持一致

## 故障排查

### 问题: Embedding Server 连接失败
- **检查**: `curl http://127.0.0.1:9020/health`
- **解决**: 启动 Embedding Server

### 问题: Qdrant 连接失败
- **检查**: `curl http://localhost:6333/collections`
- **解决**: 启动 Qdrant 服务

### 问题: LIBERO benchmark 不可用
- **影响**: 无法进行精确任务匹配
- **解决**: 使用 hash-based fallback（已自动处理）

### 问题: GPU 内存不足
- **检查**: `nvidia-smi`
- **解决**: 确保 GPU 1 有足够内存，或使用其他 GPU

## 未来改进

1. **LIBERO Benchmark 集成**: 如果 LIBERO benchmark 可用，优先使用精确匹配
2. **进度保存**: 支持断点续传，避免重复处理
3. **并行处理**: 支持多进程/多线程加速
4. **数据验证**: 添加数据完整性检查
5. **性能优化**: 优化批量插入策略

## 相关文件

- **主脚本**: `rtcache/scripts/data_processing/process_libero_goal.py`
- **配置文件**: `rtcache/config/rt_cache_config.py`
- **Embedding Server**: `rtcache/scripts/embedding/embedding_server.py`
- **测试脚本**: `rtcache/Test.ipynb`

## 作者

RT-Cache Team  
开发日期: 2024-12-23

## 更新日志

### v1.0.0 (2024-12-23)
- ✅ 初始版本
- ✅ 支持 LIBERO-Goal 数据集处理
- ✅ Hash-based 任务分配（% 1001）
- ✅ 批量插入优化
- ✅ 完整的错误处理和日志

