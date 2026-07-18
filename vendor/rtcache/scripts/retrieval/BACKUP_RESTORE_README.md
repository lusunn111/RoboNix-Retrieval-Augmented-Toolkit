# Qdrant 数据库备份与恢复指南

本目录包含用于备份和恢复 LIBERO-Goal Qdrant 向量数据库的脚本，用于实验中测试不同数量的运行时轨迹记忆带来的收益。

## 文件说明

- `backup_qdrant.py` - Python 备份脚本（核心功能）
- `restore_qdrant.py` - Python 恢复脚本（核心功能）
- `backup_libero_goal.sh` - Bash 备份脚本（便捷封装）
- `start_libero_goal_retrieval.sh` - 启动检索服务（自动恢复备份）

## 工作流程

### 1. 首次备份当前数据库

在进行实验之前，先备份当前的 Qdrant 数据库状态：

```bash
cd /path/to/rtcache/scripts/retrieval
./backup_libero_goal.sh
```

这会创建一个备份目录：
- `./qdrant_backups/backup_YYYYMMDD_HHMMSS/` - 带时间戳的备份
- `./qdrant_backups/latest/` - 指向最新备份的符号链接

### 2. 启动检索服务（自动恢复备份）

每次启动检索服务时，会自动从备份恢复数据库：

```bash
./start_libero_goal_retrieval.sh
```

启动过程：
1. 从 `./qdrant_backups/latest/` 恢复数据库
2. 删除现有 collections
3. 从 snapshot 文件重新创建 collections
4. 启动 retrieval_libero_goal.py 服务

### 3. 实验流程示例

假设你要测试添加不同数量的轨迹记忆：

#### 实验 1: 基础数据库（0 条新轨迹）

```bash
# 启动检索服务（自动恢复基础数据库）
./start_libero_goal_retrieval.sh

# 在另一个终端运行实验
# ... 运行你的实验代码 ...

# 停止服务
Ctrl+C
```

#### 实验 2: 添加 10 条成功轨迹

```bash
# 1. 恢复基础数据库并启动服务
./start_libero_goal_retrieval.sh

# 2. 运行代码添加 10 条成功轨迹到数据库
# ... 你的代码 ...

# 3. 运行实验测试性能
# ... 运行实验 ...

# 4. 停止服务
Ctrl+C
```

#### 实验 3: 添加 50 条成功轨迹

```bash
# 重复上述流程，添加 50 条轨迹
./start_libero_goal_retrieval.sh
# ... 添加 50 条轨迹 ...
# ... 运行实验 ...
```

每次启动都会恢复到初始状态，确保实验的可重复性。

## 高级用法

### 手动备份

```bash
python3 backup_qdrant.py \
    --backup-dir ./my_backup \
    --qdrant-host localhost \
    --qdrant-port 6333
```

### 手动恢复

```bash
python3 restore_qdrant.py \
    --backup-dir ./qdrant_backups/backup_20250105_120000 \
    --qdrant-host localhost \
    --qdrant-port 6333 \
    --force  # 跳过确认提示
```

### 指定特定备份

修改 `start_libero_goal_retrieval.sh` 中的恢复命令：

```bash
python3 "$SCRIPT_DIR/restore_qdrant.py" \
    --backup-dir "./qdrant_backups/backup_20250105_120000" \  # 指定具体备份
    --qdrant-host "$QDRANT_HOST" \
    --qdrant-port "$QDRANT_PORT" \
    --force
```

## 备份内容

每个备份包含所有 `libero_goal_task_*` collections 的 snapshot 文件：

```
qdrant_backups/
└── backup_20250105_120000/
    ├── libero_goal_task_0.snapshot
    ├── libero_goal_task_1.snapshot
    ├── ...
    └── libero_goal_task_9.snapshot
```

每个 snapshot 文件包含：
- 向量数据（图像 embeddings）
- Payload 数据（actions, instructions, metadata）
- Collection 配置

## 注意事项

1. **确保 Qdrant 服务正在运行**
   ```bash
   # 检查 Qdrant 状态
   curl http://localhost:6333/collections
   ```

2. **备份大小**
   - 每个 collection 的 snapshot 大小取决于数据量
   - LIBERO-Goal 约 5.2w 条数据，预计总备份大小在 GB 级别

3. **恢复时间**
   - 恢复会删除现有 collections 并重新创建
   - 大约需要几分钟，取决于数据量

4. **并发访问**
   - 恢复过程中不要访问 Qdrant
   - 确保没有其他程序在使用数据库

5. **磁盘空间**
   - 确保有足够的磁盘空间存储备份
   - 定期清理旧备份

## 故障排除

### 问题 1: 恢复失败 "Collection does not exist"

这是正常的，脚本会自动创建新 collection。

### 问题 2: HTTP 500 错误

检查 Qdrant 日志，可能是磁盘空间不足或权限问题。

### 问题 3: 备份文件丢失

检查 `./qdrant_backups/latest` 符号链接是否正确指向备份目录。

```bash
ls -l ./qdrant_backups/latest
```

### 问题 4: conda 环境激活失败

确保 `rt-mzh` conda 环境存在：

```bash
conda env list | grep rt-mzh
```

## 实验记录建议

建议在实验笔记中记录：

1. 使用的备份版本（时间戳）
2. 添加的轨迹数量
3. 实验结果指标
4. 任何异常情况

示例：

```markdown
## 实验记录

### 实验 1: 基线（2025-01-05 12:00:00 备份）
- 备份: backup_20250105_120000
- 新增轨迹: 0
- 成功率: 75%
- 平均检索时间: 50ms

### 实验 2: +10 轨迹
- 备份: backup_20250105_120000
- 新增轨迹: 10
- 成功率: 82%
- 平均检索时间: 55ms
```

## 相关文件

- `/path/to/IDEA实现笔记.md` - 实验笔记
- `/path/to/rtcache/scripts/retrieval/retrieval_libero_goal.py` - 检索服务主程序
