# 检索验证实验 - 实现总结

## 完成的工作

### 1. 创建新的实验脚本 ✅

- **run_libero_goal_Retrieval_Verify.py** - 主实验脚本
  - 去掉了数据库插入功能
  - 去掉了1:1交替执行模式
  - 每次get_action时执行检索、转换、验证流程
  - 记录详细的统计数据

- **run_libero_goal_Retrieval_Verify.sh** - 启动脚本
  - 自动恢复数据库到base状态
  - 检查服务可用性
  - 设置实验参数

### 2. 实现action到tokens的转换 ✅

在`run_libero_goal_Retrieval_Verify.py`中实现了`action_to_tokens()`函数：

```python
def action_to_tokens(action, model, unnorm_key):
    """
    将连续的action转换为token IDs，使用与predict_action相同的离散化过程。
    
    流程：
    1. 归一化action到[-1, 1]
    2. 离散化到bin索引  
    3. 转换为vocab token IDs
    """
```

这个实现完全对应modeling_speculation.py第794-799行的逆过程。

### 3. 修改speculative decoding支持外部draft tokens ✅

#### modeling_speculation.py 修改：

**新增方法：**
```python
def verify_retrieved_tokens(self, input_ids, retrieved_tokens, accept_threshold, **kwargs):
    """
    验证检索到的tokens的接受长度
    
    流程：
    1. 初始化树结构
    2. 将retrieved_tokens转为candidates格式
    3. 执行tree_decoding获取logits
    4. 调用evaluate_posterior计算accept_length
    """
```

**修改方法：**
- `eagenerate()`: 添加`retrieved_tokens`和`verify_only`参数
- `predict_action()`: 添加`retrieved_tokens`和`return_accept_length`参数

### 4. 更新接口层 ✅

#### openvla_utils.py:
```python
def get_vla_action(..., retrieved_tokens=None, return_accept_length=False):
    # 支持传入retrieved_tokens并返回accept_length
    if return_accept_length and retrieved_tokens is not None:
        action, accept_length = vla.predict_action(
            retrieved_tokens=retrieved_tokens,
            return_accept_length=True
        )
```

#### robot_utils.py:
```python
def get_action(..., retrieved_tokens=None, return_accept_length=False):
    # 转发到get_vla_action
    if return_accept_length and retrieved_tokens is not None:
        return action, accept_length
```

### 5. 实现数据库base恢复逻辑 ✅

在`run_libero_goal_Retrieval_Verify.sh`中：
```bash
# 更新latest软链接指向backup_base
rm -f "$BACKUP_LATEST_LINK"
ln -s "$BACKUP_BASE_DIR" "$BACKUP_LATEST_LINK"

# 调用恢复脚本
python3 "$RESTORE_SCRIPT" \
    --backup-dir "$BACKUP_LATEST_LINK" \
    --qdrant-host "$QDRANT_HOST" \
    --qdrant-port "$QDRANT_PORT" \
    --force
```

### 6. 完善日志记录 ✅

记录的数据包括：
- 每步的检索成功状态
- 检索时间
- token化时间  
- 生成时间
- **接受长度** (accept_length)
- 是否有retrieved tokens

输出格式：JSON文件，便于后续分析

## 核心实现细节

### action → tokens 转换

```python
# 1. 归一化
normalized = 2.0 * (action - q01) / (q99 - q01) - 1.0

# 2. 离散化
distances = |bin_centers - normalized|
bin_index = argmin(distances)

# 3. 转token
token_id = vocab_size - bin_index - 1
```

### 验证流程

```python
# 1. 检索action
retrieved_action = retrieval_api.get_action(image, instruction)

# 2. 转换为tokens
retrieved_tokens = action_to_tokens(retrieved_action, model, unnorm_key)

# 3. 调用get_action验证
action, accept_length = get_action(
    cfg, model, obs, task_label,
    retrieved_tokens=retrieved_tokens,
    return_accept_length=True
)

# 4. 记录accept_length
step_data['accept_length'] = accept_length
```

### verify_retrieved_tokens 内部实现

```python
def verify_retrieved_tokens(self, input_ids, retrieved_tokens, accept_threshold, **kwargs):
    # 1. 初始化
    draft_tokens, retrieve_indices, tree_mask, ... = initialize_tree(...)
    
    # 2. 构造candidates
    candidates = torch.cat([retrieved_tokens_tensor, padding], dim=1)
    
    # 3. Tree decoding
    logits, ... = tree_decoding(...)
    
    # 4. 评估后验
    best_candidate, accept_length, sample_p = evaluate_posterior(
        logits, candidates, logits_processor, accept_threshold
    )
    
    return int(accept_length)
```

## 使用流程

1. **启动服务**
   - Qdrant数据库
   - Embedding服务
   - Retrieval API

2. **运行实验**
   ```bash
   bash run_libero_goal_Retrieval_Verify.sh
   ```

3. **查看结果**
   - 文本日志: `*.txt`
   - 详细数据: `*_retrieval_verify.json`

4. **分析数据**
   - 统计accept_length分布
   - 分析检索成功率
   - 对比时间开销

## 关键特性

### ✅ 已实现

1. 每次get_action都执行检索和验证
2. 使用与predict_action完全相同的离散化过程
3. 在speculative decoding中验证retrieved tokens
4. 记录详细的accept_length和时间数据
5. 数据库默认载入base备份
6. 去掉了数据库插入功能
7. 去掉了1:1交替执行模式

### 📊 数据输出

- 每步的检索成功状态
- 检索时间 (retrieval_time)
- Token化时间 (tokenization_time)
- 生成时间 (generation_time)
- **接受长度 (accept_length)** ⭐
- 是否有retrieved tokens

### 🎯 验证逻辑

```
观察图像
    ↓
检索相似action
    ↓
转换为tokens (与predict_action相同的离散化)
    ↓
送入speculative decoding验证
    ↓
计算accept_length
    ↓
记录到日志
```

## 文件清单

### 新文件
1. `openvla/experiments/robot/libero/run_libero_goal_Retrieval_Verify.py`
2. `openvla/experiments/robot/libero/run_libero_goal_Retrieval_Verify.sh`
3. `openvla/experiments/robot/libero/README_Retrieval_Verify.md`

### 修改的文件
1. `openvla/prismatic/extern/hf/modeling_speculation.py`
   - 新增 `verify_retrieved_tokens()`
   - 修改 `eagenerate()`
   - 修改 `predict_action()`

2. `openvla/experiments/robot/openvla_utils.py`
   - 修改 `get_vla_action()`

3. `openvla/experiments/robot/robot_utils.py`
   - 修改 `get_action()`

## 测试建议

1. **功能测试**
   ```bash
   # 运行1个episode测试
   python run_libero_goal_Retrieval_Verify.py \
       --num_trials_per_task 1 \
       --run_id_note "test"
   ```

2. **检查输出**
   - 确认JSON文件包含accept_length字段
   - 确认时间统计正确记录
   - 确认检索成功率合理

3. **分析结果**
   - 计算平均accept_length
   - 统计检索成功率
   - 对比不同任务的表现

## 注意事项

1. **GPU显存**: 需要约14GB显存
2. **服务依赖**: 必须启动3个服务（Qdrant、Embedding、Retrieval）
3. **数据库状态**: 每次运行前自动恢复到base
4. **超时设置**: 默认30秒，可根据需要调整

## 下一步工作

如果需要进一步优化：

1. **性能优化**
   - 批量检索以减少API调用
   - 缓存retrieved_tokens

2. **分析工具**
   - 创建可视化脚本
   - 统计分析工具

3. **实验扩展**
   - 测试不同的accept_threshold
   - 测试不同的任务套件
   - 对比不同的检索策略

祝实验顺利！🎉
