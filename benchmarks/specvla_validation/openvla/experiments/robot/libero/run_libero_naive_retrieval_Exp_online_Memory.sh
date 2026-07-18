#!/bin/bash
# 多数据库状态实验脚本：测试不同数据库备份状态对纯检索成功率的影响
#
# 数据库状态: base, base+5, base+10, base+20, base+30, base+40, base+50
# 每个状态: libero_goal 的 10 个任务，每个任务 10 次试验
#
# 使用方法: 
#   bash run_libero_naive_retrieval_Exp_online_Memory.sh [TASK_SUITE] [TEST_TRIALS]
#
# 参数:
#   TASK_SUITE: libero_goal | libero_spatial | libero_object | libero_10 (默认: libero_goal)
#   TEST_TRIALS: 每个任务的测试次数 (默认: 10)
#
# 示例:
#   bash run_libero_naive_retrieval_Exp_online_Memory.sh libero_goal 10
#
# 前置条件：
# 1. 确保 embedding 服务器正在运行 (http://127.0.0.1:9020/predict)
# 2. 确保 Qdrant 数据库正在运行 (localhost:6333)
# 3. 确保有数据库备份: $RTCACHE_ROOT/scripts/retrieval/qdrant_backups/backup_base*

set -e  # 遇到错误立即退出

# =============================================================================
# 参数设置
# =============================================================================
TASK_SUITE=${1:-"libero_goal"}
TEST_TRIALS=${2:-10}
DB_STATES=("10" "20" "30" "40" "50" "60" "70" "80")

# 设置工作目录
SPECVLA_ROOT="/path/to/SpecVLA"
RTCACHE_ROOT="/path/to/rtcache"
BACKUP_ROOT="$RTCACHE_ROOT/scripts/retrieval/qdrant_backups"
RESTORE_SCRIPT="$RTCACHE_ROOT/scripts/retrieval/restore_qdrant.py"
RETRIEVAL_SCRIPT="$RTCACHE_ROOT/scripts/retrieval/retrieval_libero_goal.py"

cd $SPECVLA_ROOT

# 检查目录是否存在
if [ ! -d "$SPECVLA_ROOT" ]; then
    echo "错误: SpecVLA 根目录不存在: $SPECVLA_ROOT"
    exit 1
fi

if [ ! -d "$RTCACHE_ROOT" ]; then
    echo "错误: rtcache 根目录不存在: $RTCACHE_ROOT"
    exit 1
fi

if [ ! -f "$RESTORE_SCRIPT" ]; then
    echo "错误: 恢复脚本不存在: $RESTORE_SCRIPT"
    exit 1
fi

if [ ! -f "$RETRIEVAL_SCRIPT" ]; then
    echo "错误: Retrieval 服务脚本不存在: $RETRIEVAL_SCRIPT"
    exit 1
fi

# =============================================================================
# 打印实验配置
# =============================================================================
echo "=========================================="
echo "多数据库状态纯检索实验"
echo "=========================================="
echo "任务集: $TASK_SUITE"
echo "数据库状态: ${DB_STATES[@]}"
echo "每任务测试次数: $TEST_TRIALS"
echo "=========================================="
echo ""

# =============================================================================
# 检查服务依赖
# =============================================================================
echo "=========================================="
echo "步骤 1: 检查服务依赖"
echo "=========================================="

# 检查 Qdrant 是否运行
echo "检查 Qdrant 数据库..."
if ! curl -s --connect-timeout 2 http://127.0.0.1:6333/collections > /dev/null 2>&1; then
    echo "错误: Qdrant 数据库未运行 (http://127.0.0.1:6333)"
    echo "请在 $RTCACHE_ROOT 目录运行 ./start_db.sh"
    exit 1
else
    echo "✓ Qdrant 数据库正在运行"
fi

# 检查 embedding server
echo "检查 embedding 服务器..."
if ! curl -s --connect-timeout 2 http://127.0.0.1:9020/health > /dev/null 2>&1; then
    echo "错误: embedding 服务器未运行 (http://127.0.0.1:9020)"
    echo "请先启动 embedding 服务器"
    exit 1
else
    echo "✓ embedding 服务器正在运行"
fi

# 检查所有数据库备份是否存在
echo "检查数据库备份..."
for DB_STATE in "${DB_STATES[@]}"; do
    BACKUP_DIR="$BACKUP_ROOT/diff_backup_base+${DB_STATE}"
    if [ ! -d "$BACKUP_DIR" ]; then
        echo "错误: 数据库备份不存在: $BACKUP_DIR"
        echo "请确保已创建所有必需的备份"
        exit 1
    fi
    echo "  ✓ 找到备份: diff_backup_base+$DB_STATE"
done
echo "✓ 所有数据库备份已就绪"
echo ""

# =============================================================================
# 创建结果目录和全局日志
# =============================================================================
RESULTS_BASE_DIR="$SPECVLA_ROOT/openvla/specdecoding/test-speed/${TASK_SUITE}_Naive_Retrieval_MultiDB"

RUN_ID_BASE="EVAL-${TASK_SUITE}-NaiveRetrieval-MultiDB-$(date +%Y_%m_%d-%H_%M_%S)"
GLOBAL_LOG_FILE="$RESULTS_BASE_DIR/${RUN_ID_BASE}_GLOBAL.txt"

echo "创建结果目录: $RESULTS_BASE_DIR"
mkdir -p "$RESULTS_BASE_DIR"

echo "创建全局日志: $GLOBAL_LOG_FILE"
{
    echo "多数据库状态纯检索实验"
    echo "任务集: $TASK_SUITE"
    echo "数据库状态: ${DB_STATES[@]}"
    echo "每任务测试次数: $TEST_TRIALS"
    echo "实验开始时间: $(date)"
    echo "="$(printf '=%.0s' {1..79})
    echo ""
} > "$GLOBAL_LOG_FILE"

# =============================================================================
# 停止旧的 Retrieval 服务（如果存在）
# =============================================================================
echo "检查并停止旧的 retrieval 服务..."
if curl -s --connect-timeout 2 http://127.0.0.1:5002/health > /dev/null 2>&1; then
    RETRIEVAL_PID=$(lsof -ti:5002 2>/dev/null)
    if [ -n "$RETRIEVAL_PID" ]; then
        echo "  停止旧服务 (PID: $RETRIEVAL_PID)..."
        kill -SIGTERM $RETRIEVAL_PID 2>/dev/null || true
        sleep 2
        if kill -0 $RETRIEVAL_PID 2>/dev/null; then
            kill -SIGKILL $RETRIEVAL_PID 2>/dev/null || true
            sleep 1
        fi
        echo "  ✓ 已停止"
    fi
fi
echo ""

# =============================================================================
# 主实验循环：遍历所有数据库状态
# =============================================================================
ALL_STAGES_RESULTS=()

for STAGE_IDX in "${!DB_STATES[@]}"; do
    DB_STATE="${DB_STATES[$STAGE_IDX]}"
    STAGE_NUM=$((STAGE_IDX + 1))
    
    echo ""
    echo "###############################################################################"
    echo "# STAGE $STAGE_NUM/${#DB_STATES[@]}: 数据库状态 = $DB_STATE"
    echo "###############################################################################"
    echo ""
    
    {
        echo ""
        echo "###############################################################################"
        echo "# STAGE $STAGE_NUM/${#DB_STATES[@]}: 数据库状态 = $DB_STATE"
        echo "###############################################################################"
        echo ""
    } >> "$GLOBAL_LOG_FILE"
    
    # 创建阶段日志文件
    STAGE_LOG_FILE="$RESULTS_BASE_DIR/${RUN_ID_BASE}_Stage${STAGE_NUM}_${DB_STATE}.txt"
    {
        echo "Stage $STAGE_NUM: 数据库状态 = $DB_STATE"
        echo "任务集: $TASK_SUITE"
        echo "测试次数: $TEST_TRIALS 次/任务"
        echo "="$(printf '=%.0s' {1..79})
        echo ""
    } > "$STAGE_LOG_FILE"
    
    # =========================================================================
    # 步骤 1: 恢复数据库备份
    # =========================================================================
    echo "----------------------------------------"
    echo "步骤 1: 恢复数据库备份 'diff_backup_base+$DB_STATE'"
    echo "----------------------------------------"
    
    BACKUP_DIR="$BACKUP_ROOT/diff_backup_base+${DB_STATE}"
    echo "备份目录: $BACKUP_DIR"
    
    # 切换到 rt-mzh 环境（需要 qdrant_client）
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate rt-mzh
    
    echo "执行数据库恢复..."
    python "$RESTORE_SCRIPT" \
        --backup-dir "$BACKUP_DIR" \
        --qdrant-host "localhost" \
        --qdrant-port 6333 \
        --force
    
    if [ $? -ne 0 ]; then
        echo "✗ 数据库恢复失败！"
        exit 1
    fi
    echo "✓ 数据库已恢复为 '$DB_STATE'"
    echo ""
    
    # =========================================================================
    # 步骤 2: 验证数据库恢复成功
    # =========================================================================
    echo "----------------------------------------"
    echo "步骤 2: 验证数据库恢复"
    echo "----------------------------------------"
    
    # 获取 collection 信息
    COLLECTION_NAME="specvla_online"
    COLLECTION_INFO=$(curl -s http://127.0.0.1:6333/collections/${COLLECTION_NAME} 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        POINTS_COUNT=$(echo "$COLLECTION_INFO" | python3 -c "import sys, json; print(json.load(sys.stdin).get('result', {}).get('points_count', 'N/A'))" 2>/dev/null || echo "N/A")
        echo "✓ 验证成功："
        echo "  Collection: $COLLECTION_NAME"
        echo "  Points 数量: $POINTS_COUNT"
        
        {
            echo "数据库验证："
            echo "  Collection: $COLLECTION_NAME"
            echo "  Points 数量: $POINTS_COUNT"
            echo ""
        } >> "$STAGE_LOG_FILE"
    else
        echo "警告: 无法验证 collection 信息"
    fi
    echo ""
    
    # =========================================================================
    # 步骤 3: 重启 Retrieval 服务（加载恢复后的数据）
    # =========================================================================
    echo "----------------------------------------"
    echo "步骤 3: 启动 Retrieval 服务"
    echo "----------------------------------------"
    
    echo "直接启动 Python retrieval 服务..."
    export CUDA_VISIBLE_DEVICES=1
    python3 "$RETRIEVAL_SCRIPT" \
        --host "0.0.0.0" \
        --port 5002 \
        --embedding-url "http://127.0.0.1:9020/predict" \
        --qdrant-host "localhost" \
        --qdrant-port 6333 \
        --log-level "INFO" \
        --dataset-types "$TASK_SUITE" \
        > "/tmp/retrieval_service_stage${STAGE_NUM}.log" 2>&1 &
    
    RETRIEVAL_PID=$!
    echo "  等待服务启动 (PID: $RETRIEVAL_PID)..."
    
    for i in {1..120}; do
        if curl -s --connect-timeout 2 http://127.0.0.1:5002/health > /dev/null 2>&1; then
            echo ""
            echo "  ✓ Retrieval 服务已就绪（等待了 $i 秒）"
            break
        fi
        if [ $i -eq 120 ]; then
            echo ""
            echo "  ✗ 服务启动超时（120秒）！"
            echo "  日志文件: /tmp/retrieval_service_stage${STAGE_NUM}.log"
            tail -30 "/tmp/retrieval_service_stage${STAGE_NUM}.log"
            exit 1
        fi
        if [ $((i % 5)) -eq 0 ]; then
            echo -n "."
        fi
        sleep 1
    done
    echo ""
    
    # =========================================================================
    # 步骤 4: 切换到 specvla 环境并运行测试
    # =========================================================================
    echo "----------------------------------------"
    echo "步骤 4: 运行纯检索测试"
    echo "----------------------------------------"
    
    # 切换到 specvla 环境
    conda activate specvla
    
    # 设置环境变量
    export PYTHONPATH=$SPECVLA_ROOT:$SPECVLA_ROOT/openvla:$SPECVLA_ROOT/LIBERO
    export MUJOCO_GL=egl
    export MUJOCO_EGL_DEVICE_ID=1
    export CUDA_VISIBLE_DEVICES=1
    export ROBOSUITE_LOG_FILE=$SPECVLA_ROOT/robosuite.log
    
    echo "开始测试 (数据库状态: $DB_STATE)..."
    STAGE_JSON_FILE="$RESULTS_BASE_DIR/${RUN_ID_BASE}_Stage${STAGE_NUM}_${DB_STATE}.json"
    
    # 运行 Python 测试脚本，捕获输出
    python openvla/experiments/robot/libero/run_libero_naive_retrieval_Exp_online_Memory.py \
        --task_suite_name "$TASK_SUITE" \
        --db_state_name "$DB_STATE" \
        --stage_index "$STAGE_NUM" \
        --num_trials_per_task "$TEST_TRIALS" \
        --center_crop True \
        2>&1 | tee -a "$STAGE_LOG_FILE"
    
    PYTHON_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $PYTHON_EXIT_CODE -ne 0 ]; then
        echo "✗ 测试失败！退出码: $PYTHON_EXIT_CODE"
        {
            echo ""
            echo "✗ 测试失败！退出码: $PYTHON_EXIT_CODE"
            echo ""
        } >> "$STAGE_LOG_FILE"
        exit 1
    fi
    
    echo "✓ 测试完成"
    echo ""
    
    # =========================================================================
    # 步骤 5: 提取结果并保存到 JSON
    # =========================================================================
    echo "----------------------------------------"
    echo "步骤 5: 保存结果"
    echo "----------------------------------------"
    
    # 从日志中提取关键结果（简化版，实际可以更精确）
    SUCCESS_RATE=$(grep "Success Rate:" "$STAGE_LOG_FILE" | tail -1 | sed -n 's/.*(\(.*\)%).*/\1/p' || echo "0.0")
    TOTAL_EPISODES=$(grep "Success Rate:" "$STAGE_LOG_FILE" | tail -1 | sed -n 's/.*: \([0-9]*\)\/.*/\1/p' || echo "0")
    TOTAL_SUCCESSES=$(grep "Success Rate:" "$STAGE_LOG_FILE" | tail -1 | sed -n 's/.*: \([0-9]*\).*/\1/p' || echo "0")
    AVG_DB_TIME=$(grep "Average DB Retrieval Time:" "$STAGE_LOG_FILE" | tail -1 | awk '{print $5}' | sed 's/s//' || echo "0.0")
    
    # 创建 JSON 结果
    cat > "$STAGE_JSON_FILE" << EOF
{
  "stage": $STAGE_NUM,
  "db_state_name": "$DB_STATE",
  "task_suite": "$TASK_SUITE",
  "num_trials_per_task": $TEST_TRIALS,
  "total_episodes": $TOTAL_EPISODES,
  "total_successes": $TOTAL_SUCCESSES,
  "success_rate": $SUCCESS_RATE,
  "average_db_time": $AVG_DB_TIME,
  "database_points": "$POINTS_COUNT"
}
EOF
    
    echo "✓ 结果已保存: $STAGE_JSON_FILE"
    
    # 添加到全局日志
    {
        echo "Stage $STAGE_NUM 结果 (数据库: $DB_STATE):"
        echo "  成功率: $SUCCESS_RATE%"
        echo "  Episodes: $TOTAL_SUCCESSES/$TOTAL_EPISODES"
        echo "  平均检索时间: ${AVG_DB_TIME}s"
        echo "  数据库 Points: $POINTS_COUNT"
        echo ""
    } >> "$GLOBAL_LOG_FILE"
    
    # 停止 retrieval 服务（为下一个 stage 做准备）
    if [ $STAGE_NUM -lt ${#DB_STATES[@]} ]; then
        echo "停止 retrieval 服务，准备下一个 stage..."
        if [ -n "$RETRIEVAL_PID" ] && kill -0 $RETRIEVAL_PID 2>/dev/null; then
            kill -SIGTERM $RETRIEVAL_PID 2>/dev/null || true
            sleep 2
            if kill -0 $RETRIEVAL_PID 2>/dev/null; then
                kill -SIGKILL $RETRIEVAL_PID 2>/dev/null || true
                sleep 1
            fi
        fi
        echo "✓ 服务已停止"
        echo ""
    fi
    
    echo "Stage $STAGE_NUM 完成！"
    echo ""
done

# =============================================================================
# 生成最终汇总
# =============================================================================
echo ""
echo "==============================================================================="
echo "所有阶段完成！生成最终汇总..."
echo "==============================================================================="
echo ""

FINAL_SUMMARY_FILE="$RESULTS_BASE_DIR/${RUN_ID_BASE}_FINAL_SUMMARY.json"

# 创建 JSON 数组
echo "{" > "$FINAL_SUMMARY_FILE"
echo '  "experiment": "Pure DB Retrieval - Multiple Database States",' >> "$FINAL_SUMMARY_FILE"
echo "  \"task_suite\": \"$TASK_SUITE\"," >> "$FINAL_SUMMARY_FILE"
echo "  \"num_trials_per_task\": $TEST_TRIALS," >> "$FINAL_SUMMARY_FILE"
echo "  \"db_states\": [\"$(IFS='", "'; echo "${DB_STATES[*]}")\"]," >> "$FINAL_SUMMARY_FILE"
echo '  "stages": [' >> "$FINAL_SUMMARY_FILE"

for STAGE_IDX in "${!DB_STATES[@]}"; do
    STAGE_NUM=$((STAGE_IDX + 1))
    DB_STATE="${DB_STATES[$STAGE_IDX]}"
    STAGE_JSON_FILE="$RESULTS_BASE_DIR/${RUN_ID_BASE}_Stage${STAGE_NUM}_${DB_STATE}.json"
    
    if [ -f "$STAGE_JSON_FILE" ]; then
        cat "$STAGE_JSON_FILE" >> "$FINAL_SUMMARY_FILE"
        if [ $STAGE_NUM -lt ${#DB_STATES[@]} ]; then
            echo "," >> "$FINAL_SUMMARY_FILE"
        fi
    fi
done

echo "" >> "$FINAL_SUMMARY_FILE"
echo '  ]' >> "$FINAL_SUMMARY_FILE"
echo "}" >> "$FINAL_SUMMARY_FILE"

echo "✓ 最终汇总已保存: $FINAL_SUMMARY_FILE"
echo ""

# =============================================================================
# 打印最终对比
# =============================================================================
{
    echo ""
    echo "==============================================================================="
    echo "最终对比（所有数据库状态）"
    echo "==============================================================================="
} | tee -a "$GLOBAL_LOG_FILE"

for STAGE_IDX in "${!DB_STATES[@]}"; do
    STAGE_NUM=$((STAGE_IDX + 1))
    DB_STATE="${DB_STATES[$STAGE_IDX]}"
    STAGE_JSON_FILE="$RESULTS_BASE_DIR/${RUN_ID_BASE}_Stage${STAGE_NUM}_${DB_STATE}.json"
    
    if [ -f "$STAGE_JSON_FILE" ]; then
        SUCCESS_RATE=$(grep '"success_rate"' "$STAGE_JSON_FILE" | awk '{print $2}' | sed 's/,//')
        TOTAL_EPISODES=$(grep '"total_episodes"' "$STAGE_JSON_FILE" | awk '{print $2}' | sed 's/,//')
        TOTAL_SUCCESSES=$(grep '"total_successes"' "$STAGE_JSON_FILE" | awk '{print $2}' | sed 's/,//')
        AVG_DB_TIME=$(grep '"average_db_time"' "$STAGE_JSON_FILE" | awk '{print $2}' | sed 's/,//')
        
        {
            echo "[$DB_STATE] 成功率: ${SUCCESS_RATE}% ($TOTAL_SUCCESSES/$TOTAL_EPISODES) | 平均检索时间: ${AVG_DB_TIME}s"
        } | tee -a "$GLOBAL_LOG_FILE"
    fi
done

{
    echo "==============================================================================="
    echo ""
    echo "实验完成时间: $(date)"
    echo ""
} | tee -a "$GLOBAL_LOG_FILE"

echo ""
echo "=========================================="
echo "✓ 所有实验完成！"
echo "=========================================="
echo "结果目录: $RESULTS_BASE_DIR"
echo "全局日志: $GLOBAL_LOG_FILE"
echo "最终汇总: $FINAL_SUMMARY_FILE"
echo "=========================================="
