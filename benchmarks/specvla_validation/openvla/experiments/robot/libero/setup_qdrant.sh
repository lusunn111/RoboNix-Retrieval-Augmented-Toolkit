#!/bin/bash
# 启动Qdrant并恢复数据

cd /path/to/rtcache

echo "1. 检查Qdrant服务状态..."
curl -s http://localhost:6333/collections 2>/dev/null && echo "✓ Qdrant服务已运行" || echo "✗ Qdrant服务未运行"

echo ""
echo "2. 启动Qdrant服务..."
bash start_db.sh

echo ""
echo "3. 等待服务启动..."
sleep 3

echo ""
echo "4. 检查集合..."
curl -s http://localhost:6333/collections | python -m json.tool

echo ""
echo "5. 如果集合为空，需要恢复数据。查找备份..."
if [ -d "scripts/retrieval/qdrant_backups/latest" ]; then
    echo "找到备份: scripts/retrieval/qdrant_backups/latest"
    echo "运行以下命令恢复:"
    echo "  cd /path/to/rtcache/scripts/retrieval"
    echo "  bash start_libero_goal_retrieval.sh"
else
    echo "未找到备份"
fi
