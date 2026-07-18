# 🔴 关键Bug修复：Retrieval服务内存未刷新

## 问题诊断

### 观察到的异常现象

```
Warmup= 5: Success Rate =  73/100 ( 73.0%)  ← 插入5条轨迹
Warmup=10: Success Rate =  66/100 ( 66.0%)  ← 插入10条轨迹，成功率反而下降！
Warmup=20: Success Rate =  69/100 ( 69.0%)
Warmup=30: Success Rate =  65/100 ( 65.0%)
Warmup=40: Success Rate =  62/100 ( 62.0%)
Warmup=50: Success Rate =  62/100 ( 62.0%)
```

**预期**：插入更多成功轨迹 → 检索质量提升 → 成功率上升  
**实际**：成功率持续下降（73% → 62%）

### 根本原因

**Retrieval服务的内存缓存未更新！**

#### 执行流程分析：

```
1. [Shell] 启动retrieval服务
   └─ 从Qdrant加载所有collection到内存（~52K个点）
   
2. [Python] Stage 1 Warmup
   └─ 插入5条成功轨迹到Qdrant数据库（磁盘）
   └─ ❌ 但retrieval服务的内存缓存没有更新！
   
3. [Python] Stage 1 Test
   └─ 检索时使用的是启动时加载的旧内存
   └─ ❌ 看不到刚插入的5条轨迹！
   
4. [Python] Stage 2 Warmup
   └─ 继续插入5条轨迹到数据库（总共10条）
   └─ ❌ 内存还是旧的
   
5. [Python] Stage 2 Test
   └─ ❌ 仍然只能检索到最初的52K条，看不到新插入的10条
   
...依此类推，内存越来越"旧"
```

#### 为什么成功率下降？

1. **数据库增大，噪声增加**：虽然插入的是成功轨迹，但如果内存看不到它们，反而增加了检索负担
2. **内存-磁盘不一致**：检索结果可能受到不一致状态的影响
3. **或者其他因素**：但核心问题是新轨迹根本没被使用

## 修复方案

### 在每个阶段的Test开始前重启retrieval服务

```
Stage N:
  1. Warmup: 插入轨迹到数据库
  2. [NEW] 停止retrieval服务
  3. [NEW] 重启retrieval服务 → 重新从Qdrant加载到内存（包含新插入的轨迹）
  4. Test: 现在检索可以看到新轨迹了！
  5. 备份数据库
```

### 实现细节

#### Python脚本修改

在每个stage的warmup完成后（仅第一个task完成后，避免重复重启）：

```python
if task_id == 0 and needed_warmup > 0:
    # 创建reload标记
    reload_marker = ".stage_{target}_reload_needed"
    
    # 等待shell脚本完成reload
    reload_complete_marker = ".stage_{target}_reload_complete"
    while not exists(reload_complete_marker):
        sleep(1)
    
    # 继续测试
```

#### Shell脚本修改

监控reload请求并执行：

```bash
for STAGE in (5 10 20 30 40 50):
    # 1. 等待warmup完成（reload请求）
    wait_for(".stage_{STAGE}_reload_needed")
    
    # 2. 重启retrieval服务
    kill old_service
    start_new_service  # 会重新加载内存
    
    # 3. 通知Python可以继续
    touch ".stage_{STAGE}_reload_complete"
    
    # 4. 等待测试完成
    wait_for(".stage_{STAGE}_complete")
    
    # 5. 备份数据库
    backup_qdrant.py --note "base+{STAGE}"
```

## 预期效果修复后

```
Warmup= 5: Success Rate =  ?/100  ← 使用base + 5条新轨迹
Warmup=10: Success Rate =  ?/100  ← 使用base + 10条新轨迹（应该 >= Stage 1）
Warmup=20: Success Rate =  ?/100  ← 应该继续上升
Warmup=30: Success Rate =  ?/100
Warmup=40: Success Rate =  ?/100
Warmup=50: Success Rate =  ?/100  ← 应该是最高的
```

**成功率应该呈现单调上升或至少稳定趋势，而非下降。**

## 如何验证修复

### 查看日志确认reload

```bash
# Python日志
grep "RELOAD" EVAL-*_GLOBAL.txt

# Shell日志（终端输出）
# 应该看到：
# ✓ Stage 5 Warmup完成，开始重启retrieval服务...
# ✓ Retrieval服务已就绪
# ✓ 服务重载完成，已通知Python脚本继续测试
```

### 查看retrieval服务日志

```bash
# 每个阶段应该有独立的日志文件
ls -lh /tmp/retrieval_service_stage*.log

# 检查服务启动时的collection数量
grep "Found.*collections" /tmp/retrieval_service_stage5.log
# 应该看到点数逐渐增加：
# Stage 5:  52047 points (base 52042 + 5)
# Stage 10: 52052 points (base 52042 + 10)
# ...
```

### 检查数据库点数

```bash
# 手动查询Qdrant
curl http://127.0.0.1:6333/collections/libero_goal_task_128

# 应该看到vectors_count在增长
```

## 其他注意事项

1. **仅在task_id==0时reload**：避免每个任务都重启服务（浪费时间）
2. **超时保护**：等待reload最多5分钟，超时继续执行
3. **日志追踪**：每次reload都有独立日志文件 `/tmp/retrieval_service_stage{N}.log`
4. **标记清理**：reload完成后删除临时标记文件，避免污染结果目录

## 总结

这是一个**关键的系统bug**，导致：
- ❌ 新插入的轨迹完全没被使用
- ❌ 实验结果不可信（测试的不是"插入轨迹的影响"，而是"数据库大小的影响"）
- ✅ 修复后才能正确测试runtime memory的效果

**必须重新运行实验！**
