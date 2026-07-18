# Qdrant 数据库备份恢复 - 快速开始

## ✅ 已完成设置

系统已经配置好备份和自动恢复功能。

### 备份状态
- **备份位置**: `./qdrant_backups/backup_20260105_152828/`
- **符号链接**: `./qdrant_backups/latest/` → `backup_20260105_152828`
- **备份大小**: ~4.0GB
- **包含集合**: 39 个 LIBERO collections
  - Goal: 10 collections (~809MB)
  - Spatial: 10 collections (~827MB)
  - Object: 9 collections (~1008MB)
  - 10: 10 collections (~1387MB)

## 🚀 使用方法

### 1️⃣ 启动检索服务（自动恢复）

每次启动时会自动从备份恢复数据库到初始状态：

```bash
cd /path/to/rtcache/scripts/retrieval
./start_libero_goal_retrieval.sh
```

**启动流程**:
1. 从 `./qdrant_backups/latest/` 读取备份
2. 删除现有的所有 libero_goal_task_* collections
3. 从 snapshot 重新创建（约 30 秒）
4. 启动检索服务 (port 5002)

### 2️⃣ 实验流程示例

#### 实验 0: 基线测试（0 条新增轨迹）
```bash
# 启动服务（自动恢复到初始状态）
./start_libero_goal_retrieval.sh

# 在另一个终端运行实验
cd /path/to/SpecVLA
conda activate specvla
# ... 运行你的测试代码 ...

# 记录结果：成功率、检索时间等
```

#### 实验 1: +10 条成功轨迹
```bash
# 启动服务（自动恢复到初始状态）
./start_libero_goal_retrieval.sh

# 添加 10 条成功轨迹到数据库
# ... 你的插入代码 ...

# 运行测试并记录结果
```

#### 实验 2: +50 条成功轨迹
```bash
# 重复流程
./start_libero_goal_retrieval.sh
# ... 添加 50 条 ...
# ... 测试 ...
```

每次启动都会重置数据库，确保实验的可重复性！

### 3️⃣ 手动备份（如果需要）

如果你修改了数据库，想创建新的备份：

```bash
cd /path/to/rtcache/scripts/retrieval
./backup_libero_goal.sh
```

这会创建新的时间戳备份，并更新 `latest` 符号链接。

### 4️⃣ 手动恢复（如果需要）

```bash
cd /path/to/rtcache/scripts/retrieval

# 恢复最新备份
python3 restore_qdrant.py --force

# 或指定特定备份
python3 restore_qdrant.py --backup-dir ./qdrant_backups/backup_20260105_152416 --force
```

## 📊 实验记录模板

建议在实验笔记中这样记录：

```markdown
### 实验 0: 基线（无新增轨迹）
- **时间**: 2026-01-05 15:30
- **备份**: backup_20260105_152416
- **新增轨迹数**: 0
- **测试任务**: LIBERO-Goal (10 tasks)
- **结果**:
  - 平均成功率: XX%
  - 平均检索时间: XX ms
  - 备注: XXX

### 实验 1: +10 条轨迹
- **时间**: 2026-01-05 16:00
- **备份**: backup_20260105_152416
- **新增轨迹数**: 10 (来源: XXX)
- **结果**:
  - 平均成功率: XX% (提升 +X%)
  - 平均检索时间: XX ms
  - 备注: XXX
```

## ⚠️ 注意事项

1. **确保 Qdrant 正在运行**
   ```bash
   curl http://localhost:6333/collections
   ```

2. **恢复时间**: 约 30 秒，期间不要访问数据库

3. **磁盘空间**: 确保有 1GB+ 可用空间

4. **并发**: 恢复期间不要运行其他检索请求

## 📁 相关文件

- 备份脚本: `backup_libero_goal.sh`, `backup_qdrant.py`
- 恢复脚本: `restore_qdrant.py`
- 启动脚本: `start_libero_goal_retrieval.sh` (已集成自动恢复)
- 检索服务: `retrieval_libero_goal.py`
- 详细文档: `BACKUP_RESTORE_README.md`

## 🐛 故障排除

### 问题: 恢复时报错 "Connection refused"
**解决**: 确保 Qdrant 服务正在运行
```bash
# 检查服务状态
curl http://localhost:6333
```

### 问题: 恢复时间过长
**正常**: 804MB 数据需要约 30-60 秒恢复，耐心等待

### 问题: conda 环境激活失败
**解决**: 确保在正确的环境
```bash
conda activate rt-mzh
```
