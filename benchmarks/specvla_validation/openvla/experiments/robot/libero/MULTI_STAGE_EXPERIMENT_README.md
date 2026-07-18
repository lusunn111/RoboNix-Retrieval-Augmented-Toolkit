# 多阶段在线记忆实验

## 概述

该实验测试不同数量的运行时成功轨迹记忆对性能的影响。采用**累进式warmup**策略，在一次运行中测试多个阶段 [5, 10, 20, 30, 40, 50]。

## 工作流程

```
1. 恢复基础数据库 (base)
   ↓
2. Warmup到5条成功轨迹 → 测试所有任务 → 备份为 "base+5"
   ↓
3. 继续Warmup到10条(总共10条) → 测试 → 备份为 "base+10"
   ↓
4. 继续Warmup到20条(总共20条) → 测试 → 备份为 "base+20"
   ↓
5. ... 依此类推到50条
```

## 使用方法

### 1. 前置条件

确保以下服务正在运行：

```bash
# 1. Qdrant 数据库
cd /path/to/rtcache
./start_db.sh

# 2. Embedding 服务器 (端口 9020)
# 请确保 embedding 服务已启动

# 3. 确保有基础数据库备份
# 位置: rtcache/scripts/retrieval/qdrant_backups/latest/
```

### 2. 运行实验

```bash
cd /path/to/SpecVLA

# 使用默认参数 (libero_goal, 50次测试)
bash openvla/experiments/robot/libero/run_libero_Spec_Exp_online_Memory.sh

# 指定任务集和测试次数
bash openvla/experiments/robot/libero/run_libero_Spec_Exp_online_Memory.sh libero_spatial 100
```

### 3. 测试配置

验证配置是否正确（不运行实际实验）：

```bash
python openvla/experiments/robot/libero/test_multi_stage_config.py
```

## 参数说明

```bash
bash run_libero_Spec_Exp_online_Memory.sh [TASK_SUITE] [TEST_TRIALS]
```

- **TASK_SUITE**: 任务集名称
  - `libero_goal` (默认)
  - `libero_spatial`
  - `libero_object`
  - `libero_10`

- **TEST_TRIALS**: 每个任务的测试次数 (默认: 50)

- **Warmup阶段**: 固定为 [5, 10, 20, 30, 40, 50]（代码中定义）

## 输出结构

### 日志文件

```
openvla/specdecoding/test-speed/
└── {task_suite}_Spec_Online_Memory_MultiStage/
    └── EVAL-{suite}-SpecOnlineMem-MultiStage-{timestamp}/
        ├── EVAL-..._GLOBAL.txt              # 所有阶段总结
        ├── EVAL-..._Stage1_W5.txt           # 阶段1详细日志
        ├── EVAL-..._Stage1_W5.json          # 阶段1统计数据
        ├── EVAL-..._Stage2_W10.txt          # 阶段2详细日志
        ├── EVAL-..._Stage2_W10.json         # 阶段2统计数据
        ├── ...
        ├── EVAL-..._Stage6_W50.txt          # 阶段6详细日志
        ├── EVAL-..._Stage6_W50.json         # 阶段6统计数据
        └── EVAL-..._FINAL_SUMMARY.json      # 最终对比总结
```

### 数据库备份

```
rtcache/scripts/retrieval/qdrant_backups/
├── backup_20260105_152828_base+5/      # Warmup=5 后的备份
├── backup_20260105_160432_base+10/     # Warmup=10 后的备份
├── backup_20260105_165103_base+20/     # Warmup=20 后的备份
├── backup_20260105_174521_base+30/     # Warmup=30 后的备份
├── backup_20260105_183902_base+40/     # Warmup=40 后的备份
├── backup_20260105_192145_base+50/     # Warmup=50 后的备份
└── latest -> backup_20260105_192145_base+50/
```

## 实验特点

### 1. 累进式 Warmup

- **不重置数据库**：每个阶段在前一阶段的基础上继续添加轨迹
- **效率高**：避免重复 warmup，只需一次运行即可测试所有阶段
- **示例**：
  - 阶段1: 0 → 5 条（需要收集5条）
  - 阶段2: 5 → 10 条（只需再收集5条）
  - 阶段3: 10 → 20 条（只需再收集10条）
  - ...

### 2. 自动备份

- **阶段标记**：Python脚本在每个阶段完成后创建 `.stage_{N}_complete` 标记文件
- **Shell监控**：Shell脚本监控标记文件，自动调用备份脚本
- **命名规范**：备份目录包含时间戳和标注（如 `base+10`）

### 3. 执行模式

- **Spec + DB (1:1 alternating)**：
  - 奇数步：模型推理 (Speculative Decoding)
  - 偶数步：数据库检索
- **Online Insertion**：只在 Warmup 阶段插入成功轨迹
- **Test Phase**：不插入新轨迹，纯粹测试性能

## 日志示例

### GLOBAL 日志

```
STAGE 1/6: Target Warmup = 5
Test Success Rate: 35/50 (70.0%)

STAGE 2/6: Target Warmup = 10
Test Success Rate: 38/50 (76.0%)

...

FINAL COMPARISON ACROSS ALL STAGES
Warmup= 5: Success Rate =  35/ 50 ( 70.0%)
Warmup=10: Success Rate =  38/ 50 ( 76.0%)
Warmup=20: Success Rate =  42/ 50 ( 84.0%)
Warmup=30: Success Rate =  43/ 50 ( 86.0%)
Warmup=40: Success Rate =  44/ 50 ( 88.0%)
Warmup=50: Success Rate =  45/ 50 ( 90.0%)
```

### Stage 日志

```
Task 0: put the black bowl on top of the checker plate
[WARMUP] Need 5 more warmup trajectories
  [WARMUP] Collected 1/5 (Total: 1/5)
  [WARMUP] Collected 2/5 (Total: 2/5)
  ...
  [WARMUP] ✓ Reached target!

[TEST] Running 50 test trials...
  [TEST] Trial 1/50: SUCCESS | 1/1 (100.0%)
  [TEST] Trial 2/50: FAILED | 1/2 (50.0%)
  ...

Task 0 test success rate: 7/10 (70.0%)
```

## 故障排查

### 1. Retrieval 服务启动超时

```bash
# 查看服务日志
tail -50 /tmp/retrieval_service.log

# 手动检查服务
curl http://127.0.0.1:5002/health
```

### 2. 备份失败

```bash
# 切换到 rt-mzh 环境
conda activate rt-mzh

# 手动执行备份
cd /path/to/rtcache
python scripts/retrieval/backup_qdrant.py --note "test"
```

### 3. 阶段标记未创建

检查结果目录是否存在：
```bash
ls -la openvla/specdecoding/test-speed/{task_suite}_Spec_Online_Memory_MultiStage/
```

## 注意事项

1. **数据库空间**：每个备份约4GB，6个阶段需要约24GB空间
2. **运行时间**：完整实验可能需要数小时（取决于warmup速度）
3. **GPU占用**：实验独占 GPU 1 (`CUDA_VISIBLE_DEVICES=1`)
4. **Conda环境**：
   - Python实验: `specvla`
   - 数据库备份: `rt-mzh`
   - Shell脚本会自动切换环境

## 后续使用备份

如果想从某个阶段开始重新测试：

```bash
# 1. 修改 restore_qdrant.py 指向特定备份
# 2. 或手动指定备份目录
python rtcache/scripts/retrieval/restore_qdrant.py \
    --backup-dir rtcache/scripts/retrieval/qdrant_backups/backup_20260105_160432_base+10
```
