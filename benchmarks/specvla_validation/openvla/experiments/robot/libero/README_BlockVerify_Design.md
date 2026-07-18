# SpecVLA Block-wise Verification 策略设计文档

## 1. 背景与动机

### 1.1 现有方案

**标准 SD (Speculative Decoding)** 的核心流程：
1. Draft model 生成多个候选 tokens
2. Target model 一次 forward 验证所有候选
3. **逐 token 比较**：draft token vs target model 预测的 token
4. 找到第一个不匹配的位置，接受之前的所有 tokens

**核心问题**：SD 是 token 级别的验证，验证粒度太细，一旦某个 token 不匹配就需要重新生成。

### 1.2 新需求：Block-wise SD

将 SD 的验证粒度从 **token 级别** 改为 **block 级别**：

1. **Draft 来源**：检索 top-5 候选（而非 draft model 生成）
2. **验证粒度**：以 block 为单位验证，而非单个 token
3. **Fallback 机制**：block 验证失败时，AR 生成该 block，然后继续验证下一个 block

```
标准 SD:    [t0] → [t1] → [t2] → [t3] → [t4] → [t5] → [t6]
             ↓      ↓      ↓      ↓      ↓      ↓      ↓
           verify verify verify verify verify verify verify

Block SD:  [---Block1---] → [---Block2---] → [Block3]
           [t0, t1, t2]     [t3, t4, t5]       [t6]
                ↓                ↓               ↓
           block_verify    block_verify    block_verify
```

---

## 2. 设计概述

### 2.1 Block 划分原理

OpenVLA 的 action 由 7 个 tokens 组成，对应：
```
[x, y, z, roll, pitch, yaw, gripper]
  └─Block1─┘  └───Block2───┘   └B3┘
    位置        姿态              夹爪
```

| Block | Token 索引 | 语义含义 | 大小 |
|-------|-----------|---------|------|
| Block 1 | [0, 1, 2] | 末端位置 (x, y, z) | 3 tokens |
| Block 2 | [3, 4, 5] | 末端姿态 (roll, pitch, yaw) | 3 tokens |
| Block 3 | [6] | 夹爪状态 (open/close) | 1 token |

### 2.2 整体流程

```
Step 1: 检索 Top-5 候选 Actions
        ↓
Step 2: 将 Actions 转换为 Tokens，构建 5 条候选链
        ↓
Step 3: Block-wise SD 验证
        ┌─────────────────────────────────────────────────┐
        │  For each Block:                                │
        │    1. 将 5 条链的该 block tokens 送入 Target LM  │
        │    2. Target LM 输出 logits                     │
        │    3. 比较 logits argmax vs draft tokens       │
        │    4. 选择最佳候选，判断是否 block 级别通过     │
        │    5. 通过 → 接受该 block                       │
        │       失败 → AR 生成该 block                    │
        │    6. 更新 context，继续下一个 block           │
        └─────────────────────────────────────────────────┘
        ↓
Step 4: 拼接所有 block tokens，转换回 Action
```

---

## 3. Block-wise SD 验证的核心设计

### 3.1 与标准 SD 的对比

| 方面 | 标准 SD | Block-wise SD |
|------|---------|---------------|
| Draft 来源 | Draft Model 生成 | 检索 Top-K 候选 |
| 验证粒度 | 单个 token | Block (3/3/1 tokens) |
| 失败处理 | 从失败位置开始 AR | 从失败 block 开始 AR |
| Tree 结构 | Draft model 生成的树 | 5 条平行候选链 |

### 3.2 Block-wise 验证的核心逻辑

类似标准 SD 的 `evaluate_posterior`，但改为 block 级别：

```python
def evaluate_posterior_block(logits, candidates, block_indices, accept_threshold=None):
    """
    Block 级别的后验评估 - 类似 SD 的 evaluate_posterior
    
    核心思想：
    - 标准 SD: 逐 token 比较，第一个不匹配就停止
    - Block SD: 整个 block 内的 tokens 都要满足条件才算通过
    
    Args:
        logits: Target model 的输出 [K, block_size, vocab_size]
                K = 候选数量, block_size = 当前 block 的 token 数
        candidates: Draft tokens [K, block_size]
        block_indices: 当前 block 在完整 action 中的位置 [0,1,2] 或 [3,4,5] 或 [6]
        accept_threshold: token 差异容忍阈值，None 表示精确匹配
    
    Returns:
        best_candidate: 最佳候选索引
        accept_length: 在该 block 内接受的 token 数
        block_passed: 整个 block 是否通过 (accept_length == block_size)
    """
    K, block_size = candidates.shape
    
    # 对每个候选计算 block 内的匹配情况
    if accept_threshold is None:
        # 精确匹配
        posterior_mask = (candidates == torch.argmax(logits, dim=-1)).int()
    else:
        # 允许一定差异
        posterior_mask = (
            torch.abs(candidates - torch.argmax(logits, dim=-1)) <= accept_threshold
        ).int()
    
    # 计算每个候选的连续接受长度 (block 内)
    # 使用 cumprod 找到第一个不匹配的位置
    candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
    
    # 选择最佳候选
    best_accept_length = candidates_accept_length.max()
    best_candidate = torch.argmax(candidates_accept_length).item()
    
    # 判断 block 是否完全通过
    block_passed = (best_accept_length == block_size)
    
    return best_candidate, best_accept_length.item(), block_passed
```

### 3.3 Block 级别的"通过"标准

**标准 1: 严格模式 - 整个 block 都要通过**
```python
# Block 内所有 token 都必须满足条件
block_passed = (accept_length == block_size)
```

**标准 2: 宽松模式 - 允许部分通过**
```python
# Block 内允许部分 token 通过，设置比例阈值
block_passed = (accept_length / block_size) >= block_pass_ratio  # 如 0.67 (2/3)
```

**标准 3: 带 token 阈值的模式**
```python
# 使用 accept_threshold 允许 token 差异
# 例如 accept_threshold=5 表示 draft token 和 target 预测相差 ≤5 就算匹配
block_passed = all(|draft_token[i] - argmax(logits[i])| <= accept_threshold for i in block)
```

**推荐**: 使用 **标准 3**，与原 SD 的 `accept_threshold` 参数保持一致

---

### 3.4 完整 Block-wise SD 流程

```python
def blockwise_sd_verify(model, observation, top_k_actions, processor, task_description, cfg):
    """
    Block-wise Speculative Decoding 验证的完整流程
    
    核心流程：
    1. Prefill: 处理 image + prompt，得到 context
    2. 将检索的 top-k actions 转换为 tokens (draft)
    3. 逐 Block 验证:
       - 构建 tree mask，让 K 个候选的当前 block 并行 forward
       - Target model 输出 logits
       - 类似 SD 的 evaluate_posterior，但以 block 为单位
       - 通过 → 接受该 block
       - 失败 → AR 生成该 block
    4. 拼接所有 block，转换回 action
    
    Returns:
        final_action: shape [7] 最终 action
        stats: 统计信息
    """
    K = len(top_k_actions)  # 候选数量，如 5
    
    # ============================================
    # Step 1: Prefill - 处理 image + prompt
    # ============================================
    model_inputs = model.base_model.prepare_inputs_for_generation(input_ids, **kwargs)
    
    outputs, orig, hidden_states, input_embeds = model(
        **model_inputs,
        return_dict=True,
        output_hidden_states=True,
        output_orig=True,
    )
    
    # 获取第一个 token (类似原 SD 的 initialize_tree)
    first_token = torch.argmax(orig[:, -1])
    past_key_values = outputs.past_key_values
    
    # ============================================
    # Step 2: 将 actions 转换为 tokens (draft)
    # ============================================
    top_k_tokens = torch.stack([
        torch.tensor(action_to_tokens(a, model, cfg.unnorm_key))
        for a in top_k_actions
    ]).to(input_embeds.device)  # [K, 7]
    
    # ============================================
    # Step 3: 定义 Block 配置
    # ============================================
    BLOCKS = [
        {'name': 'position', 'indices': [0, 1, 2]},
        {'name': 'orientation', 'indices': [3, 4, 5]},
        {'name': 'gripper', 'indices': [6]},
    ]
    
    final_tokens = []
    stats = {'blocks': []}
    current_past_kv = past_key_values
    
    # ============================================
    # Step 4: 逐 Block 验证
    # ============================================
    for block_idx, block in enumerate(BLOCKS):
        block_stat = {'name': block['name'], 'indices': block['indices']}
        block_size = len(block['indices'])
        
        # 4.1 提取当前 block 的 draft tokens
        draft_block = top_k_tokens[:, block['indices']]  # [K, block_size]
        
        # 4.2 构建 tree mask 用于并行验证
        tree_mask = build_block_tree_mask(K, block_size, len(final_tokens))
        model.base_model.language_model.tree_mask = tree_mask
        
        # 4.3 准备输入 embeddings
        # 如果有已验证/生成的 tokens，需要包含它们
        if len(final_tokens) > 0:
            prefix_embeds = model.ea_layer.embed_tokens(
                torch.tensor([final_tokens]).to(input_embeds.device)
            )
        else:
            prefix_embeds = model.ea_layer.embed_tokens(first_token.unsqueeze(0).unsqueeze(0))
        
        # 将 K 个候选的 block tokens 展平
        draft_block_flat = draft_block.flatten()  # [K * block_size]
        draft_embeds = model.ea_layer.embed_tokens(draft_block_flat.unsqueeze(0))
        
        # 4.4 Target model forward (并行验证 K 个候选)
        position_ids = build_block_position_ids(K, block_size, len(final_tokens))
        
        outputs = model.base_model.language_model(
            inputs_embeds=draft_embeds,
            past_key_values=current_past_kv,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True
        )
        
        logits = outputs.logits  # [1, K * block_size, vocab_size]
        logits = logits.view(K, block_size, -1)  # [K, block_size, vocab_size]
        
        # 4.5 Block 级别的 evaluate_posterior
        best_candidate, accept_length, block_passed = evaluate_posterior_block(
            logits, draft_block, block['indices'], 
            accept_threshold=cfg.accept_threshold
        )
        
        # 4.6 根据验证结果决定
        if block_passed:
            # Block 验证通过，接受该 block
            block_stat['mode'] = 'verified'
            block_stat['accept_length'] = block_size
            block_stat['candidate_idx'] = best_candidate
            
            accepted_tokens = draft_block[best_candidate].tolist()
            final_tokens.extend(accepted_tokens)
            
            # 更新 KV cache (只保留最佳候选的 KV)
            current_past_kv = select_kv_for_candidate(
                outputs.past_key_values, best_candidate, K, block_size
            )
            
        else:
            # Block 验证失败，AR 生成该 block
            block_stat['mode'] = 'AR_generated'
            block_stat['accept_length'] = accept_length
            
            # 清除 tree mask
            model.base_model.language_model.tree_mask = None
            
            # AR 生成当前 block
            ar_tokens = ar_generate_block(
                model, current_past_kv, final_tokens, block_size
            )
            
            final_tokens.extend(ar_tokens)
            current_past_kv = ar_past_kv  # 更新为 AR 生成后的 KV
        
        stats['blocks'].append(block_stat)
    
    # ============================================
    # Step 5: 将 tokens 转换回 action
    # ============================================
    final_action = tokens_to_action(np.array(final_tokens), model, cfg.unnorm_key)
    
    return final_action, stats


def ar_generate_block(model, past_kv, prefix_tokens, block_size):
    """
    AR 生成指定数量的 tokens
    
    这与原 ea_forward 类似，但只生成 block_size 个 tokens
    """
    generated = []
    current_past_kv = past_kv
    
    for i in range(block_size):
        # 如果是第一个 token，使用 prefix 的最后一个
        if i == 0 and len(prefix_tokens) > 0:
            input_token = torch.tensor([[prefix_tokens[-1]]]).to(model.device)
        elif i > 0:
            input_token = torch.tensor([[generated[-1]]]).to(model.device)
        else:
            # 需要从 first_token 开始
            input_token = first_token.unsqueeze(0).unsqueeze(0)
        
        input_embeds = model.ea_layer.embed_tokens(input_token)
        
        outputs = model.base_model.language_model(
            inputs_embeds=input_embeds,
            past_key_values=current_past_kv,
            use_cache=True,
            return_dict=True
        )
        
        next_token = outputs.logits[0, -1].argmax().item()
        generated.append(next_token)
        current_past_kv = outputs.past_key_values
    
    return generated, current_past_kv
```

---

## 4. Tree Mask 构建 (关键)

### 4.1 将 Top-5 视为 5 条平行链

与原 SD 的 tree 结构不同，这里的 5 条链是平行的（来自检索的 5 个候选），而非 draft model 生成的树形结构：

```
                        [Context: Image + Prompt]
                              │
                        [first_token]  ← Target model 预测的第一个 token
                              │
           ┌──────────┬───────┼───────┬──────────┐
           ▼          ▼       ▼       ▼          ▼
       [Chain 1] [Chain 2] [Chain 3] [Chain 4] [Chain 5]
       (top-1)    (top-2)  (top-3)  (top-4)   (top-5)
           │          │       │       │          │
           ▼          ▼       ▼       ▼          ▼
       [B1: pos] [B1: pos] [B1: pos] [B1: pos] [B1: pos]
           │          │       │       │          │
           ▼          ▼       ▼       ▼          ▼
       [B2: ori] [B2: ori] [B2: ori] [B2: ori] [B2: ori]
           │          │       │       │          │
           ▼          ▼       ▼       ▼          ▼
       [B3: grip][B3: grip][B3: grip][B3: grip][B3: grip]
```

### 4.2 Block 级别的 Tree Mask 构建

```python
def build_block_tree_mask(K, block_size, prefix_len):
    """
    为 Block 级验证构建 tree attention mask
    
    场景：验证 K 个候选的某个 block (含 block_size 个 tokens)
    
    核心规则：
    1. 所有 draft tokens 可以 attend to context (past_kv 中)
    2. 同一候选链内的 tokens 保持 causal attention
    3. 不同候选链之间不能相互 attend
    
    Args:
        K: 候选数量 (如 5)
        block_size: 当前 block 的 token 数 (如 3 for position, 1 for gripper)
        prefix_len: 已验证/生成的 tokens 数量
    
    Returns:
        tree_mask: [K * block_size, K * block_size]
    
    Example for K=5, block_size=3:
        展平后的 tokens: [c1_t0, c1_t1, c1_t2, c2_t0, c2_t1, c2_t2, ..., c5_t0, c5_t1, c5_t2]
        
        Mask 示意 (1=可见, 0=不可见):
        c1_t0: [1, 0, 0, | 0, 0, 0, | 0, 0, 0, | 0, 0, 0, | 0, 0, 0]
        c1_t1: [1, 1, 0, | 0, 0, 0, | 0, 0, 0, | 0, 0, 0, | 0, 0, 0]
        c1_t2: [1, 1, 1, | 0, 0, 0, | 0, 0, 0, | 0, 0, 0, | 0, 0, 0]
        c2_t0: [0, 0, 0, | 1, 0, 0, | 0, 0, 0, | 0, 0, 0, | 0, 0, 0]
        c2_t1: [0, 0, 0, | 1, 1, 0, | 0, 0, 0, | 0, 0, 0, | 0, 0, 0]
        ...
    """
    total_len = K * block_size
    
    # 初始化为全 0 (不可见)
    mask = torch.zeros(total_len, total_len)
    
    for k in range(K):
        start = k * block_size
        
        # 链内 causal attention
        for i in range(block_size):
            for j in range(i + 1):
                mask[start + i, start + j] = 1
    
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, total_len, total_len]


def build_block_position_ids(K, block_size, prefix_len):
    """
    构建 position ids
    
    所有 K 个候选的同一位置应该有相同的 position id
    因为它们是从同一个 prefix 出发的
    
    Args:
        K: 候选数量
        block_size: 当前 block 的 token 数
        prefix_len: 已有的 prefix 长度
    
    Returns:
        position_ids: [1, K * block_size]
    
    Example: K=5, block_size=3, prefix_len=10
        每个候选的位置: [10, 11, 12]
        展平后: [10, 11, 12, 10, 11, 12, 10, 11, 12, 10, 11, 12, 10, 11, 12]
    """
    base_positions = torch.arange(prefix_len, prefix_len + block_size)
    position_ids = base_positions.repeat(K)
    return position_ids.unsqueeze(0)
```

### 4.3 处理 Block 之间的依赖

当 Block 1 验证通过后，验证 Block 2 时需要考虑：

**方案 A: 每个 Block 独立验证（简单但有 gap）**
```python
# Block 1 通过后，所有候选的 Block 2 都基于同一个验证通过的 Block 1
# 问题：top-2 候选的 Block 2 可能与 top-1 候选的 Block 1 不太匹配
```

**方案 B: 保持链的一致性（推荐）**
```python
# Block 1 验证通过后，后续只验证同一候选链的 Block 2
# 即如果 top-3 的 Block 1 最佳，后续优先验证 top-3 的 Block 2/3
```

```python
def blockwise_sd_with_chain_consistency(model, top_k_tokens, ...):
    """
    保持链一致性的 Block-wise SD
    """
    current_chain_idx = None  # 当前使用的候选链
    
    for block in BLOCKS:
        if current_chain_idx is None:
            # 第一个 block: 验证所有候选
            best_idx, accept_len, passed = verify_block_all_candidates(...)
        else:
            # 后续 block: 优先验证当前链，失败再尝试其他
            best_idx, accept_len, passed = verify_block_with_priority(
                priority_chain=current_chain_idx, ...
            )
        
        if passed:
            current_chain_idx = best_idx  # 更新当前链
            accept_block(...)
        else:
            # AR fallback
            current_chain_idx = None  # 重置
            ar_generate_block(...)
```

---

## 5. 与原 SD 代码的对应关系

### 5.1 核心函数对照

| 原 SD 函数 (`utils.py`) | Block-wise SD 对应 | 说明 |
|-------------------------|-------------------|------|
| `initialize_tree()` | `initialize_block_verify()` | Prefill + 获取 first token |
| `tree_decoding()` | `block_tree_decoding()` | 并行 forward K 个候选的某个 block |
| `evaluate_posterior()` | `evaluate_posterior_block()` | Block 级别的验证判断 |
| `update_inference_inputs()` | `update_block_inputs()` | 更新 KV cache 和 prefix |
| `ea_layer.topK_genrate()` | 检索 Top-K | Draft 来源不同 |

### 5.2 关键代码复用

```python
# 原 evaluate_posterior 的核心逻辑 (utils.py)
posterior_mask = (
    candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
).int()
candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)

# Block-wise 版本：几乎相同，只是作用在 block 上
posterior_mask = (
    draft_block.to(logits.device) == torch.argmax(logits, dim=-1)
).int()
block_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
```

### 5.3 可直接复用的原 SD 参数

```python
# 这些参数可以直接复用
accept_threshold: int = 9  # token 差异容忍阈值，与原 SD 一致
```

---

## 6. 实现计划

### 6.1 文件结构

```
openvla/
├── experiments/robot/libero/
│   ├── run_libero_block_sd.py           # 主运行脚本
│   ├── run_libero_block_sd.sh           # Shell 脚本
│   └── README_BlockVerify_Design.md     # 本文档
│
└── prismatic/extern/hf/
    ├── modeling_speculation.py          # 修改: 添加 block-wise 验证方法
    └── utils.py                          # 修改: 添加 block-wise 辅助函数
```

### 6.2 开发步骤

| 阶段 | 任务 | 依赖 |
|------|------|------|
| 1 | 在 `utils.py` 中添加 `build_block_tree_mask()` | - |
| 2 | 在 `utils.py` 中添加 `evaluate_posterior_block()` | 1 |
| 3 | 在 `modeling_speculation.py` 中添加 `block_verify()` 方法 | 1, 2 |
| 4 | 在 `modeling_speculation.py` 中添加 `predict_action_block_sd()` | 3 |
| 5 | 创建 `run_libero_block_sd.py` (参考 `ambiguity_mix.py`) | 4 |
| 6 | 添加统计和日志 | 5 |
| 7 | 测试与调参 | 6 |

### 6.3 关键参数

```python
@dataclass
class BlockSDConfig:
    # Block 配置
    blocks: list = field(default_factory=lambda: [
        {'name': 'position', 'indices': [0, 1, 2]},
        {'name': 'orientation', 'indices': [3, 4, 5]},
        {'name': 'gripper', 'indices': [6]},
    ])
    
    # 检索参数
    top_k: int = 5  # 检索候选数量
    
    # 验证参数 (与原 SD 一致)
    accept_threshold: int = 9  # token 差异容忍阈值
    
    # Block 验证通过标准
    block_pass_mode: str = 'strict'  # 'strict' = 整个 block 通过, 'partial' = 允许部分通过
    block_pass_ratio: float = 1.0    # partial 模式下的通过比例
```

---

## 7. 预期效果与评估指标

### 7.1 与其他策略的对比

| 策略 | 每步时间 | 准确性 | 特点 |
|------|---------|--------|------|
| 纯 AR | 慢 (7 次 forward) | 最高 | 基准 |
| 原 SD | 快 (1-2 次 forward) | 高 | draft model 生成 |
| Retrieval noverify | 最快 (0 forward) | 低 | 直接用检索 |
| **Block SD** | 中等 (1-3 次 forward) | 中高 | 本方案 |

### 7.2 评估指标

```python
metrics = {
    # 任务成功率
    'success_rate': total_successes / total_episodes,
    
    # 时间效率
    'avg_time_per_step': weighted_avg_time,
    'speedup_vs_ar': pure_ar_time / actual_time,
    
    # Block 验证统计
    'block_pass_rates': {
        'position': position_pass_count / total_steps,
        'orientation': orientation_pass_count / total_steps,
        'gripper': gripper_pass_count / total_steps,
    },
    
    # Fallback 统计
    'ar_fallback_rate': ar_block_count / total_block_count,
    'full_verify_rate': full_verify_steps / total_steps,  # 3 个 block 都通过的比例
}
```

---

## 8. 总结

### 8.1 核心思想

**Block-wise SD = 标准 SD 的验证逻辑 + Block 粒度 + 检索作为 Draft**

```
标准 SD:  Draft Model → Target Model Verify (token 级) → Accept/Reject
Block SD: Retrieval   → Target Model Verify (block 级) → Accept/AR per block
```

### 8.2 优势

1. **复用 SD 验证逻辑**：`evaluate_posterior` 的核心逻辑几乎不变
2. **粒度更合理**：block 级别更符合 action 的语义结构
3. **Fallback 更精细**：只对失败的 block 进行 AR，而非整个 action
4. **利用检索高效性**：检索比 draft model 更快

### 8.3 下一步

1. 实现代码
2. 调参：`accept_threshold`、`top_k`、`block_pass_mode`
3. 对比实验：与原 SD、纯 AR、Retrieval noverify 比较
