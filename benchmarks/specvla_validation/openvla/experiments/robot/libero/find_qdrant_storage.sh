#!/bin/bash
# 查找Qdrant存储路径

echo "正在搜索Qdrant存储位置..."
echo ""

# 搜索可能的位置
echo "1. 搜索 qdrant_storage 目录:"
find /path/to -type d -name "qdrant_storage" 2>/dev/null | head -10

echo ""
echo "2. 搜索 .qdrant 文件:"
find /path/to -type d -name "*.qdrant" 2>/dev/null | head -10

echo ""
echo "3. 搜索 qdrant_backups:"
find /path/to -type d -name "qdrant_backups" 2>/dev/null | head -10

echo ""
echo "4. 检查rtcache目录:"
if [ -d "/path/to/rtcache" ]; then
    echo "在rtcache中查找..."
    find /path/to/rtcache -type d \( -name "*qdrant*" -o -name "*storage*" \) 2>/dev/null | head -10
fi

echo ""
echo "5. 检查常见位置:"
for dir in \
    "/path/to/rtcache/storage" \
    "/path/to/rtcache/qdrant_storage" \
    "/path/to/rtcache/scripts/retrieval/qdrant_storage" \
    "/path/to/rtcache/scripts/retrieval/storage" \
    "$HOME/qdrant_storage" \
    "./qdrant_storage"
do
    if [ -d "$dir" ]; then
        echo "  ✓ 找到: $dir"
        ls -lh "$dir" | head -5
    fi
done

echo ""
echo "6. 检查Qdrant备份位置:"
if [ -d "/path/to/rtcache/scripts/retrieval/qdrant_backups/latest" ]; then
    echo "  ✓ 找到备份: /path/to/rtcache/scripts/retrieval/qdrant_backups/latest"
    ls -lh /path/to/rtcache/scripts/retrieval/qdrant_backups/latest | head -5
fi
