# 运行 SpecVLA 详细计时测试

## 快速开始

### 1. 修改模型路径

编辑 `run_timing_test.sh`，修改以下路径为你的实际路径：

```bash
# 第30-31行
BACKBONE_MODEL="${SPECVLA_ROOT}/backbone_models/openvla-7b-finetuned-libero-goal"
SPEC_CHECKPOINT="${SPECVLA_ROOT}/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/checkpoint-600"
```

### 2. 运行测试

```bash
cd /path/to/SpecVLA
bash run_timing_test.sh
```

### 3. 查看结果

测试完成后，结果保存在 `TGT_DIR/` 目录：

```bash
# 查看完整日志和计时统计
tail -100 TGT_DIR/EVAL-*.txt

# 查看详细计时数据 (JSON格式)
cat TGT_DIR/EVAL-*_detailed_timing.json | python -m json.tool

# 查看最后的统计摘要
grep -A 30 "Detailed Timing Breakdown" TGT_DIR/EVAL-*.txt
```

## 参数说明

### 关键参数

- `--num_trials_per_task 2`: 每个任务测试2次（快速测试，完整测试用10次）
- `--accept_threshold 9`: Speculative decoding 的接受阈值
- `--center_crop True`: 使用中心裁剪（如果模型训练时有数据增强）

### 任务集选择

- `libero_goal`: 10个任务（默认）
- `libero_spatial`: 10个任务
- `libero_object`: 10个任务
- `libero_10`: 10个任务
- `libero_90`: 90个任务（需要更长时间）

## 直接使用 Python 运行

如果你想直接用 Python 命令运行：

```bash
cd /path/to/SpecVLA

export PYTHONPATH="${PWD}:${PWD}/openvla:${PWD}/LIBERO:${PYTHONPATH}"

python openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py \
    --model_family openvla \
    --pretrained_checkpoint backbone_models/openvla-7b-finetuned-libero-goal \
    --spec_checkpoint openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/checkpoint-600 \
    --task_suite_name libero_goal \
    --center_crop True \
    --accept_threshold 9 \
    --num_trials_per_task 2
```

## 输出示例

测试完成后会输出类似如下的计时统计：

```
============================================================
Detailed Timing Breakdown:
============================================================
Drafter (Small Model): mean=0.015234s, std=0.002341s, min=0.012000s, max=0.025000s, total=12.345s, count=810
                       └─ 15.32% of total time
Tokenizer           : mean=0.000000s, std=0.000000s, min=0.000000s, max=0.000000s, total=0.000s, count=810
                       └─ 0.00% of total time
ViT                 : mean=0.042156s, std=0.003201s, min=0.038000s, max=0.055000s, total=34.146s, count=810
                       └─ 42.35% of total time
LLM                 : mean=0.028945s, std=0.002876s, min=0.024000s, max=0.040000s, total=23.446s, count=810
                       └─ 29.08% of total time
De-Tokenizer        : mean=0.000123s, std=0.000045s, min=0.000080s, max=0.000200s, total=0.100s, count=810
                       └─ 0.12% of total time
Verification        : mean=0.016234s, std=0.001234s, min=0.014000s, max=0.022000s, total=13.149s, count=810
                       └─ 16.31% of total time
Total               : mean=0.099234s, std=0.008234s, min=0.085000s, max=0.125000s, total=80.380s, count=810
============================================================
```

## 常见问题

### 1. 模型路径错误
```
错误: Backbone 模型路径不存在
```
**解决**: 修改 `run_timing_test.sh` 中的 `BACKBONE_MODEL` 和 `SPEC_CHECKPOINT` 变量

### 2. PYTHONPATH 错误
```
ModuleNotFoundError: No module named 'libero'
```
**解决**: 确保设置了正确的 PYTHONPATH：
```bash
export PYTHONPATH="/path/to/SpecVLA:${PWD}/openvla:${PWD}/LIBERO:${PYTHONPATH}"
```

### 3. CUDA 内存不足
**解决**: 减少测试次数或使用量化：
```bash
--num_trials_per_task 1  # 减少测试次数
--load_in_8bit True      # 使用8位量化
```

## 恢复原始代码

如果需要恢复到修改前的版本：

```bash
cd /path/to/SpecVLA/openvla/experiments/robot/
cp robot_utils.py.backup robot_utils.py
cp openvla_utils.py.backup openvla_utils.py

cd /path/to/SpecVLA/openvla/prismatic/extern/hf/
cp modeling_speculation.py.backup modeling_speculation.py
cp utils.py.backup utils.py
```
