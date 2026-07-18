#!/bin/bash
# 快速启动检索验证实验
# 本脚本会检查所有前置条件并启动实验

set -e

echo "========================================"
echo "检索验证实验 - 快速启动"
echo "========================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查函数
check_service() {
    local name=$1
    local url=$2
    
    echo -n "检查 $name ... "
    if curl -s "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi
}

# 1. 检查Qdrant
echo "=== 1. 检查服务状态 ==="
if ! check_service "Qdrant" "http://localhost:6333/collections"; then
    echo -e "${YELLOW}提示: 请先启动Qdrant服务${NC}"
    echo "通常运行: docker run -p 6333:6333 qdrant/qdrant"
    exit 1
fi

# 2. 检查Embedding服务
if ! check_service "Embedding服务" "http://127.0.0.1:9020"; then
    echo -e "${YELLOW}警告: Embedding服务未运行，但将继续执行${NC}"
fi

# 3. 检查Retrieval API
if ! check_service "Retrieval API" "http://127.0.0.1:5002"; then
    echo -e "${RED}错误: Retrieval API未运行！${NC}"
    echo "请运行: cd /path/to/rtcache/scripts/retrieval && bash start_libero_goal_retrieval.sh"
    exit 1
fi

echo ""
echo "=== 2. 检查文件 ==="
SCRIPT_PATH="/path/to/SpecVLA/openvla/experiments/robot/libero/run_libero_goal_Retrieval_Verify.sh"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}错误: 脚本文件不存在: $SCRIPT_PATH${NC}"
    exit 1
fi

if [ ! -x "$SCRIPT_PATH" ]; then
    echo "添加执行权限..."
    chmod +x "$SCRIPT_PATH"
fi
echo -e "${GREEN}✓ 脚本文件就绪${NC}"

echo ""
echo "=== 3. 启动实验 ==="
echo "正在运行: $SCRIPT_PATH"
echo ""

# 执行实验
bash "$SCRIPT_PATH"

echo ""
echo "========================================"
echo "实验完成！"
echo "========================================"
echo "查看结果:"
echo "  日志目录: ./specdecoding/test-speed/libero_goal_Retrieval_Verify/"
echo "  详细文档: ./openvla/experiments/robot/libero/README_Retrieval_Verify.md"
echo "========================================"
