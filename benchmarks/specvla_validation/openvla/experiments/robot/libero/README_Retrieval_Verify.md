# 检索验证实验说明

## 实验目的

本实验用于验证从数据库检索到的action tokens的接受长度（accept length）。在每次调用`get_action`时：

1. 从Qdrant数据库检索相似的action slice
2. 将检索到的action转换为tokens（使用与predict_action相同的离散化过程）
3. 将这些tokens作为draft tokens送入speculative decoding进行验证
4. 记录验证的接受长度以及各种时间统计数据

## 新增/修改的文件

### 1. 实验脚本
- `openvla/experiments/robot/libero/run_libero_goal_Retrieval_Verify.py` - 主实验脚本
- `openvla/experiments/robot/libero/run_libero_goal_Retrieval_Verify.sh` - 启动脚本

### 2. 核心修改

#### modeling_speculation.py
- 新增 `verify_retrieved_tokens()` 方法：用于验证外部提供的tokens并返回接受长度
- 修改 `eagenerate()` 方法：支持`retrieved_tokens`和`verify_only`参数
- 修改 `predict_action()` 方法：支持`retrieved_tokens`和`return_accept_length`参数

#### openvla_utils.py
- 修改 `get_vla_action()` 函数：支持传入`retrieved_tokens`并返回`accept_length`

#### robot_utils.py
- 修改 `get_action()` 函数：支持传入`retrieved_tokens`并返回`accept_length`

#### run_libero_goal_Retrieval_Verify.py
- 实现 `action_to_tokens()` 函数：将连续action转换为token IDs

## 使用方法

### 前置条件

1. **启动Qdrant数据库**
```bash
# 确保Qdrant正在运行在localhost:6333
```

2. **启动Embedding服务**
```bash
# 确保embedding服务运行在http://127.0.0.1:9020/predict
```

3. **启动Retrieval API服务**
```bash
cd /path/to/rtcache/scripts/retrieval
bash start_libero_goal_retrieval.sh --skip-restore
```

### 运行实验

```bash
cd /path/to/SpecVLA/openvla

# 方法1: 使用shell脚本（推荐）
bash experiments/robot/libero/run_libero_goal_Retrieval_Verify.sh

# 方法2: 直接运行Python脚本
python experiments/robot/libero/run_libero_goal_Retrieval_Verify.py \
    --model_family openvla \
    --pretrained_checkpoint /path/to/checkpoint \
    --spec_checkpoint /path/to/spec_checkpoint \
    --task_suite_name libero_goal \
    --center_crop True \
    --accept_threshold 9 \
    --num_trials_per_task 10 \
    --run_id_note "Retrieval_Verify"
```

## 数据库恢复

脚本会自动将数据库恢复到base状态：

1. 将`latest`软链接指向`backup_base`
2. 调用`restore_qdrant.py`恢复数据库

如果需要手动恢复：
```bash
cd /path/to/rtcache/scripts/retrieval

# 更新软链接
rm -f qdrant_backups/latest
ln -s qdrant_backups/backup_base qdrant_backups/latest

# 恢复数据库
conda activate rt-mzh
python restore_qdrant.py \
    --backup-dir qdrant_backups/latest \
    --qdrant-host localhost \
    --qdrant-port 6333 \
    --force
```

## 输出数据

### 日志文件

保存在 `specdecoding/test-speed/libero_goal_Retrieval_Verify/` 目录下：

- `EVAL-{task_suite}-{model_family}-{timestamp}.txt` - 文本日志
- `EVAL-{task_suite}-{model_family}-{timestamp}_retrieval_verify.json` - 详细数据

### JSON数据格式

```json
[
  {
    "task_id": 0,
    "task_description": "put the alphabet soup in the basket",
    "episode_idx": 0,
    "success": true,
    "steps": [
      {
        "episode": 0,
        "step": 0,
        "retrieval_success": true,
        "retrieval_time": 0.123456,
        "tokenization_time": 0.001234,
        "generation_time": 0.234567,
        "accept_length": 5,
        "has_retrieved_tokens": true
      },
      ...
    ]
  },
  ...
]
```

### 统计数据

每个任务的统计包括：
- 总检索尝试次数
- 成功检索次数和成功率
- 平均检索时间
- 平均生成时间
- 接受长度分布

## action_to_tokens 转换逻辑

将连续action转换为tokens的过程：

1. **归一化** (action → normalized_action)
   ```
   normalized = 2 * (action - q01) / (q99 - q01) - 1
   ```

2. **离散化** (normalized_action → bin_index)
   ```
   bin_index = argmin(|normalized - bin_centers|)
   ```

3. **转换为token** (bin_index → token_id)
   ```
   token_id = vocab_size - bin_index - 1
   ```

这个过程与`predict_action`中的逆过程完全对应。

## verify_retrieved_tokens 验证流程

1. 初始化树结构和KV缓存
2. 将retrieved_tokens转换为candidates格式
3. 执行tree_decoding获取logits
4. 调用evaluate_posterior计算接受长度
5. 返回accept_length

## 注意事项

1. **数据库状态**：每次运行前会自动恢复到base状态，确保实验可重复
2. **服务依赖**：必须确保Qdrant、Embedding和Retrieval三个服务都在运行
3. **GPU显存**：需要足够的显存加载模型（约14GB）
4. **检索超时**：默认检索超时时间为30秒，可根据需要调整

## 调试

如果遇到问题：

1. **检查服务状态**
   ```bash
   # Qdrant
   curl http://localhost:6333/collections
   
   # Retrieval API
   curl http://127.0.0.1:5002
   
   # Embedding服务
   curl http://127.0.0.1:9020
   ```

2. **查看日志**
   - 脚本日志：`specdecoding/test-speed/libero_goal_Retrieval_Verify/*.txt`
   - 检索服务日志：检查启动Retrieval API的终端输出

3. **常见错误**
   - `Connection refused`: 服务未启动
   - `Timeout`: 检索超时，可能是数据库问题或网络问题
   - `CUDA out of memory`: GPU显存不足

## 后续分析

使用保存的JSON文件可以进行：
- 接受长度分布统计
- 检索成功率分析
- 时间开销分析（检索 vs 生成）
- 不同任务的表现对比

示例分析代码：
```python
import json
import numpy as np

# 读取数据
with open('EVAL-..._retrieval_verify.json', 'r') as f:
    data = json.load(f)

# 统计接受长度
accept_lengths = []
for task in data:
    for step in task['steps']:
        if step['has_retrieved_tokens']:
            accept_lengths.append(step['accept_length'])

# 分析
print(f"平均接受长度: {np.mean(accept_lengths):.2f}")
print(f"接受长度中位数: {np.median(accept_lengths):.2f}")
print(f"接受长度标准差: {np.std(accept_lengths):.2f}")
```
