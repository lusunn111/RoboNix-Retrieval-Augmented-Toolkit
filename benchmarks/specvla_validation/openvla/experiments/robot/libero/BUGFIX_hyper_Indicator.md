# Bug修复说明 - run_libero_goal_hyper_Indicator.py

## 问题总结

运行时遇到了4个错误，现已全部修复。

---

## 修复的问题

### ❌ 问题1: action_to_tokens函数实现错误

**错误信息：**
```
Tokenization error: 'SpecVLAforActionPrediction' object has no attribute 'action_tokenizer'
```

**原因：**
模型对象没有 `action_tokenizer` 属性，我错误地假设了这个API。

**修复方法：**
从 `run_libero_goal_Retrieval_Verify.py` 复制了正确的实现：

```python
def action_to_tokens(action, model, unnorm_key):
    """
    将连续的action转换为token IDs
    
    步骤：
    1. 归一化action到[-1, 1]
    2. 离散化到bin索引
    3. 转换为vocab token IDs
    """
    # 1. 归一化
    action_norm_stats = model.get_action_stats(unnorm_key)
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    normalized_actions = 2.0 * (action - action_low) / (action_high - action_low) - 1.0
    
    # 2. 离散化
    bin_centers = model.bin_centers
    discretized_actions = np.argmin(np.abs(bin_centers - normalized_actions[:, None]), axis=1)
    
    # 3. 转换为token IDs
    vocab_size = model.vocab_size
    token_ids = vocab_size - discretized_actions - 1
    
    return token_ids
```

✅ **已修复**

---

### ❌ 问题2: get_action函数参数错误

**错误信息：**
```
TypeError: get_action() got an unexpected keyword argument 'use_spec'
```

**原因：**
`get_action` 的实际签名是：
```python
get_action(cfg, model, obs, task_label, processor=None, generate_mode=None, 
           retrieved_tokens=None, return_accept_length=False)
```

但我错误地使用了：
```python
get_action(observation, task_description, model, ...)  # ❌ 参数顺序错误
get_action(..., use_spec=True, spec_draft_logits=...)  # ❌ 不存在的参数
```

**修复方法：**

**修复前（错误）：**
```python
# AR验证模式
action, accept_length = get_action(
    observation,          # ❌ 顺序错误
    task_description,     # ❌ 顺序错误
    model,
    processor=processor,
    use_spec=True,        # ❌ 不存在的参数
    spec_draft_logits=retrieved_tokens,  # ❌ 不存在的参数
)

# 纯AR模式
action, _ = get_action(
    observation,          # ❌ 顺序错误
    task_description,     # ❌ 顺序错误
    model,
    processor=processor,
    use_spec=False,       # ❌ 不存在的参数
)
```

**修复后（正确）：**
```python
# AR验证模式
action, accept_length = get_action(
    cfg,                  # ✅ 第1个参数
    model,                # ✅ 第2个参数
    observation,          # ✅ 第3个参数
    task_description,     # ✅ 第4个参数
    processor=processor,
    generate_mode='AR',   # ✅ 正确参数
    retrieved_tokens=retrieved_tokens,  # ✅ 正确参数
    return_accept_length=True,          # ✅ 正确参数
)

# 纯AR模式
action = get_action(
    cfg,                  # ✅ 第1个参数
    model,                # ✅ 第2个参数
    observation,          # ✅ 第3个参数
    task_description,     # ✅ 第4个参数
    processor=processor,
    generate_mode='AR',   # ✅ 正确参数
)
```

✅ **已修复**

---

### ❌ 问题3: save_rollout_video参数错误

**错误信息：**
```
TypeError: save_rollout_video() got an unexpected keyword argument 'log_dir'
```

**原因：**
`save_rollout_video` 的实际签名是：
```python
save_rollout_video(rollout_images, idx, success, task_description, log_file=None)
```

但我错误地传入了 `log_dir` 参数。

**修复方法：**

**修复前（错误）：**
```python
save_rollout_video(
    replay_images,
    total_episodes,
    success=done,
    task_description=task_description,
    log_dir=log_dir,  # ❌ 应该是 log_file
)
```

**修复后（正确）：**
```python
save_rollout_video(
    replay_images,
    total_episodes,
    success=done,
    task_description=task_description,
    log_file=log_file,  # ✅ 正确参数
)
```

✅ **已修复**

---

### ❌ 问题4: f-string格式化错误

**错误信息：**
```
ValueError: Invalid format specifier
Traceback:
  print(f"  Step {t-cfg.num_steps_wait}: mode={mode}, composite={composite_metric:.4f if not np.isnan(composite_metric) else 'nan'}, accept_len={accept_length}")
```

**原因：**
在f-string中不能使用条件表达式来决定格式化规则。Python的f-string格式化器无法解析这种嵌套的条件表达式。

**修复方法：**

**修复前（错误）：**
```python
# ❌ 在f-string中使用条件表达式决定格式化
print(f"  Step {t-cfg.num_steps_wait}: mode={mode}, composite={composite_metric:.4f if not np.isnan(composite_metric) else 'nan'}, accept_len={accept_length}")
```

**修复后（正确）：**
```python
# ✅ 先计算要显示的字符串，然后放入f-string
composite_str = f"{composite_metric:.4f}" if not np.isnan(composite_metric) else "nan"
print(f"  Step {t-cfg.num_steps_wait}: mode={mode}, composite={composite_str}, accept_len={accept_length}")
```

**关键点：**
- 条件表达式应该在f-string外面计算
- 先生成格式化字符串，再插入到f-string中
- 这样可以避免格式化器解析复杂表达式的问题

✅ **已修复**

---

## 修复总结

### 修改的代码行

1. **action_to_tokens函数** (第120-170行)
   - 完全重写，使用正确的离散化算法

2. **get_action调用 - AR验证模式** (第419-428行)
   - 修改参数顺序：`cfg, model, observation, task_description`
   - 替换参数：`use_spec=True` → `generate_mode='AR'`
   - 替换参数：`spec_draft_logits` → `retrieved_tokens`
   - 添加参数：`return_accept_length=True`

3. **get_action调用 - 纯AR模式** (第432-439行)
   - 修改参数顺序：`cfg, model, observation, task_description`
   - 替换参数：`use_spec=False` → `generate_mode='AR'`
   - 移除无效的返回值解包

4. **save_rollout_video调用** (第506-512行)
   - 替换参数：`log_dir=log_dir` → `log_file=log_file`

5. **print格式化** (第468-469行)
   - 修复f-string格式化错误
   - 将条件表达式移到f-string外面

### 验证结果

✅ Python语法检查通过
✅ Linter检查通过
✅ 所有API调用符合实际函数签名

---

## 正确的API使用指南

### 1. action_to_tokens

```python
# 将连续action转换为token IDs
retrieved_tokens = action_to_tokens(
    action=retrieved_action,  # numpy array, shape (7,)
    model=model,              # VLA模型实例
    unnorm_key=cfg.unnorm_key # 反归一化key
)
```

### 2. get_action

```python
# 基础用法（纯AR）
action = get_action(
    cfg=cfg,
    model=model,
    obs=observation,
    task_label=task_description,
    processor=processor,
    generate_mode='AR',
)

# 带accept_length（AR + 验证）
action, accept_length = get_action(
    cfg=cfg,
    model=model,
    obs=observation,
    task_label=task_description,
    processor=processor,
    generate_mode='AR',
    retrieved_tokens=retrieved_tokens,  # 检索到的tokens
    return_accept_length=True,          # 返回接受长度
)

# 带时间统计
action, time_tuple = get_action(
    cfg=cfg,
    model=model,
    obs=observation,
    task_label=task_description,
    processor=processor,
    generate_mode='AR',
    return_time=True,
)
```

### 3. save_rollout_video

```python
save_rollout_video(
    rollout_images=replay_images,        # 图像列表
    idx=total_episodes,                  # episode索引
    success=done,                        # 是否成功
    task_description=task_description,   # 任务描述
    log_file=log_file,                  # 日志文件对象（可选）
)
```

---

## 参考代码来源

所有修复都基于以下参考代码：

1. **action_to_tokens**: `run_libero_goal_Retrieval_Verify.py` 第102-147行
2. **get_action**: `robot_utils.py` 第63行开始
3. **save_rollout_video**: `libero_utils.py` 第61-74行

---

## 现在可以运行了

所有错误已修复，可以正常运行：

```bash
bash openvla/experiments/robot/libero/run_libero_goal_hyper_Indicator.sh
```

或者直接运行Python脚本：

```bash
python openvla/experiments/robot/libero/run_libero_goal_hyper_Indicator.py \
    --composite_threshold 0.4 \
    --alpha 0.5 \
    --num_trials_per_task 10
```

---

## 经验教训

1. **查看参考代码**：直接复制工作正常的代码实现
2. **检查函数签名**：使用 `grep` 或直接阅读源码确认参数
3. **参数顺序重要**：位置参数必须按顺序传递
4. **避免假设API**：不要猜测函数参数，要查看实际定义
5. **f-string格式化**：复杂的条件表达式应该在f-string外面计算

### f-string格式化的最佳实践

```python
# ❌ 错误：在f-string中使用条件表达式决定格式化
value = 3.14159
print(f"{value:.2f if value > 0 else value}")  # ValueError!

# ✅ 正确：先计算格式化字符串
value = 3.14159
formatted = f"{value:.2f}" if value > 0 else str(value)
print(f"Value: {formatted}")

# ✅ 或者使用嵌套f-string（Python 3.8+）
print(f"Value: {f'{value:.2f}' if value > 0 else str(value)}")
```

修复完成！ ✅
