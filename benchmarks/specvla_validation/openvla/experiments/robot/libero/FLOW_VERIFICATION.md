# 完整流程验证

## 执行流程详解

### 初始化阶段
```
[Shell] 停止旧的retrieval服务
[Shell] 启动服务（默认参数，会恢复base数据库）
  └─ start_libero_goal_retrieval.sh 
     └─ SKIP_RESTORE=false (默认)
     └─ 调用 restore_qdrant.py --backup-dir latest
     └─ 从base恢复所有39个collections
     └─ 启动Flask服务，加载到内存
[Shell] 等待服务就绪（最多120秒）
[Shell] 启动Python脚本（后台运行）
[Shell] 开始监控阶段标记
```

**验证点**：
- ✅ Qdrant有52042条数据（base）
- ✅ Retrieval服务内存也是52042条

---

### Stage 1: Warmup=5

#### Warmup阶段
```
[Python] needed_warmup = 5 - 0 = 5
[Python] warmup_completed_for_stage = False

[Python] Task 0:
  └─ 循环执行episode，插入成功轨迹到Qdrant
  └─ cumulative_warmup_count: 0→1→2→3→4→5
  └─ 达到target_warmup=5，设置 warmup_completed_for_stage=True，break
  
[Python] Task 0 warmup完成
  └─ 检测到: warmup_completed_for_stage=True AND needed_warmup>0 AND task_warmup_successes>0
  └─ 创建标记: .stage_5_reload_needed
  └─ 等待: .stage_5_reload_complete
```

**验证点**：
- ✅ Qdrant现在有52047条（base 52042 + 5条新轨迹）
- ❌ Retrieval服务内存还是52042条（旧的）

#### Reload阶段
```
[Shell] 检测到 .stage_5_reload_needed
[Shell] 停止retrieval服务
[Shell] 启动服务 --skip-restore
  └─ start_libero_goal_retrieval.sh --skip-restore
     └─ SKIP_RESTORE=true
     └─ 跳过 restore_qdrant.py
     └─ 直接启动Flask服务
     └─ 从Qdrant加载52047条到内存 ✅
[Shell] 等待服务就绪
[Shell] 创建标记: .stage_5_reload_complete

[Python] 检测到 .stage_5_reload_complete
[Python] 删除标记文件
[Python] 设置 needed_warmup=0（防止重复reload）
```

**验证点**：
- ✅ Qdrant仍然是52047条
- ✅ Retrieval服务内存也是52047条（重新加载了！）

#### Test阶段
```
[Python] Task 0:
  └─ 跳过warmup（warmup_completed_for_stage=True）
  └─ 直接测试10次
  └─ 检索使用的是52047条数据 ✅

[Python] Task 1-9:
  └─ 跳过warmup
  └─ 直接测试10次
  └─ 检索使用的是52047条数据 ✅

[Python] 所有tasks完成
[Python] 创建标记: .stage_5_complete
```

**验证点**：
- ✅ 所有测试都使用base+5的数据

#### 备份阶段
```
[Shell] 检测到 .stage_5_complete
[Shell] 切换到rt-mzh环境
[Shell] 调用 backup_qdrant.py --note "base+5"
  └─ 创建 backup_20260106_HHMMSS_base+5/
  └─ 备份所有39个collections（52047条数据）
[Shell] 切回specvla环境
```

**验证点**：
- ✅ 备份目录包含52047条数据
- ✅ 备份名称包含 "base+5"

---

### Stage 2: Warmup=10

#### Warmup阶段
```
[Python] needed_warmup = 10 - 5 = 5
[Python] warmup_completed_for_stage = False

[Python] Task 0（或其他task，取决于哪个先达到）:
  └─ 继续从Qdrant的52047条基础上插入
  └─ cumulative_warmup_count: 5→6→7→8→9→10
  └─ 达到target_warmup=10，设置 warmup_completed_for_stage=True
```

**验证点**：
- ✅ Qdrant现在有52052条（base 52042 + 10条）
- ❌ Retrieval服务内存还是52047条（Stage 1 reload的状态）

#### Reload阶段
```
[Shell] 检测到 .stage_10_reload_needed
[Shell] 停止retrieval服务
[Shell] 启动服务 --skip-restore ✅
  └─ 从Qdrant加载52052条到内存
```

**验证点**：
- ✅ Retrieval服务内存更新为52052条

#### Test阶段
```
[Python] 所有tasks测试，使用52052条数据 ✅
```

---

### Stage 3-6: 依此类推

每个阶段都是：
1. Warmup: 累积插入到Qdrant（磁盘）
2. Reload: --skip-restore，从Qdrant加载最新数据到内存
3. Test: 使用最新的内存数据检索
4. Backup: 备份当前Qdrant状态

---

## 关键修复点总结

### 修复1: 添加 --skip-restore 参数
```bash
# start_libero_goal_retrieval.sh
SKIP_RESTORE=false  # 默认false，初始启动会恢复base

if [ "$SKIP_RESTORE" = false ]; then
    restore_qdrant.py  # 恢复base数据库
else
    echo "跳过恢复，使用现有数据"  # Reload时保留已插入的轨迹
fi
```

### 修复2: Reload时使用 --skip-restore
```bash
# run_libero_Spec_Exp_online_Memory.sh
# Stage N reload:
bash start_libero_goal_retrieval.sh --skip-restore  # 保留已插入的轨迹
```

### 修复3: Warmup完成后立即reload
```python
# run_libero_Spec_Exp_online_Memory.py
if warmup_completed_for_stage and needed_warmup > 0 and task_warmup_successes > 0:
    # 刚完成warmup的那个task触发reload
    create_reload_marker()
    wait_for_reload_complete()
    needed_warmup = 0  # 防止后续tasks重复reload
```

---

## 预期结果

### 数据库点数变化
```
初始: 52042 (base)
Stage 1 Warmup完成: 52047 (base+5)
Stage 2 Warmup完成: 52052 (base+10)
Stage 3 Warmup完成: 52062 (base+20)
Stage 4 Warmup完成: 52072 (base+30)
Stage 5 Warmup完成: 52082 (base+40)
Stage 6 Warmup完成: 52092 (base+50)
```

### 成功率变化
```
Warmup= 5: Success Rate = X%   (使用52047条检索)
Warmup=10: Success Rate ≥ X%   (使用52052条检索，应该≥Stage1)
Warmup=20: Success Rate ≥ ...  (使用52062条检索)
Warmup=30: Success Rate ≥ ...
Warmup=40: Success Rate ≥ ...
Warmup=50: Success Rate ≥ ...  (使用52092条检索，应该最好)
```

**成功率应该呈现单调递增或稳定趋势！**

---

## 如何验证修复有效

### 1. 检查初始恢复
```bash
# 查看初始启动日志
grep "Restoring Qdrant" /tmp/retrieval_service.log
# 应该看到: [INFO] Restoring Qdrant database from backup...

# 检查点数
curl http://127.0.0.1:6333/collections/libero_goal_task_128 | jq .result.vectors_count
# 应该是base的数量（例如5204）
```

### 2. 检查Stage 1 reload
```bash
# 查看reload日志
grep "Skipping database restore" /tmp/retrieval_service_stage5.log
# 应该看到: [INFO] Skipping database restore (--skip-restore flag set)

# 检查点数增加
curl http://127.0.0.1:6333/collections/libero_goal_task_128 | jq .result.vectors_count
# 应该比base多5条
```

### 3. 检查Python日志
```bash
grep "RELOAD" EVAL-*_Stage1_W5.txt
# 应该看到:
# [RELOAD] Warmup completed for this stage
# [RELOAD] ✓ Retrieval service reloaded successfully!
```

### 4. 检查备份
```bash
ls -lh qdrant_backups/backup_*_base+5/
# 应该看到39个snapshot文件，总大小略大于base
```

---

## 潜在问题排查

### 问题1: Reload没有触发
**症状**: 没有看到 `[RELOAD]` 日志  
**原因**: `warmup_completed_for_stage=True` 但 `task_warmup_successes=0`  
**排查**: 检查是哪个task完成的warmup，确保reload条件满足

### 问题2: 成功率仍然下降
**症状**: Warmup=10的成功率 < Warmup=5  
**原因1**: Reload失败，仍在使用旧内存  
**排查**: 检查 `/tmp/retrieval_service_stage10.log`  
**原因2**: 插入的轨迹质量有问题  
**排查**: 检查warmup阶段的成功率

### 问题3: 数据库点数没有增加
**症状**: 所有stage的点数都一样  
**原因**: 插入失败或插入到了错误的collection  
**排查**: 检查 `[OnlineInsert]` 日志，确认插入成功

---

## 总结

整个系统现在的逻辑是：

1. ✅ **初始启动**: 恢复base
2. ✅ **Stage N Warmup**: 累积插入到Qdrant磁盘
3. ✅ **Stage N Reload**: --skip-restore，从磁盘重新加载到内存
4. ✅ **Stage N Test**: 使用最新内存数据检索
5. ✅ **Stage N Backup**: 备份当前Qdrant状态

**关键**: --skip-restore确保reload时不清空已插入的轨迹！
