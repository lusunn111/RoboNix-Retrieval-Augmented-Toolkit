#!/bin/bash
# 多阶段实验脚本：测试不同运行时记忆数量 [5, 10, 20, 30, 40, 50] 对成功率的影响
#
# 使用方法: 
#   bash run_libero_Spec_Exp_online_Memory.sh [TASK_SUITE] [TEST_TRIALS]
#
# 参数:
#   TASK_SUITE: libero_goal | libero_spatial | libero_object | libero_10 (默认: libero_goal)
#   TEST_TRIALS: 每个任务的测试次数 (默认: 50)
#
# 示例:
#   bash run_libero_Spec_Exp_online_Memory.sh libero_goal 50
#   bash run_libero_Spec_Exp_online_Memory.sh libero_spatial 100
#
# 工作流程:
#   1. 恢复基础数据库 (base)
#   2. Warmup到5 → 测试 → 备份为 "base+5"
#   3. 继续Warmup到10 → 测试 → 备份为 "base+10"
#   4. 继续Warmup到20 → 测试 → 备份为 "base+20"
#   5. ... 依此类推到50
#
# 前置条件：
# 1. 确保 embedding 服务器正在运行 (http://127.0.0.1:9020/predict)
# 2. 确保 Qdrant 数据库正在运行 (localhost:6333)
# 3. 确保有基础数据库备份: $RTCACHE_ROOT/scripts/retrieval/qdrant_backups/latest/

set -e  # 遇到错误立即退出

# =============================================================================
# 参数设置
# =============================================================================
TASK_SUITE=${1:-"libero_goal"}
TEST_TRIALS=${2:-50}
WARMUP_STAGES=(5 10 20 30 40 50)  # 累进式 warmup 阶段

# 设置工作目录
SPECVLA_ROOT="/path/to/SpecVLA"
RTCACHE_ROOT="/path/to/rtcache"
BACKUP_SCRIPT="$RTCACHE_ROOT/scripts/retrieval/backup_qdrant.py"

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

if [ ! -f "$BACKUP_SCRIPT" ]; then
    echo "错误: 备份脚本不存在: $BACKUP_SCRIPT"
    exit 1
fi

# =============================================================================
# 停止并重启 Retrieval 服务（仅 Stage 1 重置数据库 + 重新加载内存）
# =============================================================================
echo "=========================================="
echo "步骤 1: 为 Stage 1 重置向量数据库和检索服务"
echo "=========================================="

# 检查并确保 latest symlink 指向 base（初始状态）
BACKUP_ROOT="$RTCACHE_ROOT/scripts/retrieval/qdrant_backups"
LATEST_LINK="$BACKUP_ROOT/latest"
BASE_BACKUP="$BACKUP_ROOT/backup_base"

echo "检查基础备份..."
if [ ! -d "$BASE_BACKUP" ]; then
    echo "错误: 基础备份不存在: $BASE_BACKUP"
    echo "请先创建基础备份: python backup_qdrant.py --note 'base'"
    exit 1
fi

# 确保 latest 指向 base（实验开始前的初始状态）
if [ -L "$LATEST_LINK" ]; then
    rm "$LATEST_LINK"
fi
ln -s "backup_base" "$LATEST_LINK"
echo "✓ latest symlink 已设置为指向 base"

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

# 检查并停止旧的 retrieval 服务
echo "检查 DB retrieval 服务器..."
if curl -s --connect-timeout 2 http://127.0.0.1:5002/health > /dev/null 2>&1; then
    echo "检测到 retrieval 服务正在运行，正在停止..."
    
    # 尝试优雅停止（发送 Ctrl+C 信号到进程）
    RETRIEVAL_PID=$(lsof -ti:5002 2>/dev/null)
    if [ -n "$RETRIEVAL_PID" ]; then
        echo "  找到 retrieval 服务进程 PID: $RETRIEVAL_PID"
        kill -SIGTERM $RETRIEVAL_PID 2>/dev/null || true
        sleep 2
        
        # 检查是否还在运行
        if kill -0 $RETRIEVAL_PID 2>/dev/null; then
            echo "  进程未响应，强制终止..."
            kill -SIGKILL $RETRIEVAL_PID 2>/dev/null || true
            sleep 1
        fi
        echo "  ✓ 已停止旧的 retrieval 服务"
    fi

# 启动新的 retrieval 服务（会自动恢复基础数据库）
echo "启动 retrieval 服务（恢复基础数据库 + 加载到内存）..."
echo "注意: 这将恢复数据库到基础备份状态"
bash "$RTCACHE_ROOT/scripts/retrieval/start_libero_goal_retrieval.sh" \
    --dataset-types "$TASK_SUITE" > /tmp/retrieval_service.log 2>&1 &

RETRIEVAL_START_PID=$!
echo "  retrieval 服务启动中 (PID: $RETRIEVAL_START_PID)..."

# 等待服务启动（数据库恢复需要时间，最多等待120秒）
echo "  等待数据库恢复和服务启动（这可能需要1-2分钟，请耐心等待）..."
for i in {1..120}; do
    if curl -s --connect-timeout 2 http://127.0.0.1:5002/health > /dev/null 2>&1; then
        echo ""
        echo "  ✓ retrieval 服务已就绪（等待了 $i 秒）"
        break
    fi
    if [ $i -eq 120 ]; then
        echo ""
        echo "错误: retrieval 服务启动超时（超过120秒）！"
        echo "日志文件: /tmp/retrieval_service.log"
        tail -30 /tmp/retrieval_service.log
        exit 1
    fi
    # 每5秒打印一个进度点
    if [ $((i % 5)) -eq 0 ]; then
        echo -n "."
    fi
    sleep 1
done

echo "✓ 基础数据库已恢复，检索服务已就绪（内存已重新加载）"
echo ""

# =============================================================================
# 激活 conda 环境
# =============================================================================
echo "激活 conda 环境: specvla"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate specvla

# 检查 conda 环境是否激活成功
if [ "$CONDA_DEFAULT_ENV" != "specvla" ]; then
    echo "警告: conda 环境可能未正确激活，当前环境: $CONDA_DEFAULT_ENV"
fi

# =============================================================================
# 设置环境变量
# =============================================================================
export PYTHONPATH=$PWD:$PWD/openvla:$PWD/LIBERO
export MUJOCO_GL=egl
export ROBOSUITE_LOG_FILE=$SPECVLA_ROOT/robosuite.log
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1

# =============================================================================
# 检查模型文件
# =============================================================================
PRETRAINED_CHECKPOINT="$SPECVLA_ROOT/backbone_models/openvla-7b-finetuned-libero-goal"
SPEC_CHECKPOINT="$SPECVLA_ROOT/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190"

if [ ! -d "$PRETRAINED_CHECKPOINT" ]; then
    echo "警告: 预训练模型检查点不存在: $PRETRAINED_CHECKPOINT"
fi

if [ ! -d "$SPEC_CHECKPOINT" ]; then
    echo "警告: SpecVLA 检查点不存在: $SPEC_CHECKPOINT"
fi

# =============================================================================
# 打印实验配置
# =============================================================================
echo "=========================================="
echo "多阶段实验配置:"
echo "  任务集: $TASK_SUITE"
echo "  Warmup阶段: ${WARMUP_STAGES[@]}"
echo "  每任务测试次数: $TEST_TRIALS"
echo "  执行模式: Spec + DB (1:1)"
echo "  预训练模型: $PRETRAINED_CHECKPOINT"
echo "  SpecVLA检查点: $SPEC_CHECKPOINT"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# =============================================================================
# 运行多阶段实验 (Python 脚本会自动处理所有阶段)
# =============================================================================
echo "=========================================="
echo "步骤 2: 运行多阶段实验"
echo "=========================================="
echo "开始运行实验（累进式 warmup: 5 → 10 → 20 → 30 → 40 → 50）..."
python openvla/experiments/robot/libero/run_libero_Spec_Exp_online_Memory.py \
    --model_family openvla \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --spec_checkpoint "$SPEC_CHECKPOINT" \
    --task_suite_name "$TASK_SUITE" \
    --center_crop True \
    --use_spec True \
    --parallel_draft False \
    --accept_threshold 9 \
    --num_trials_per_task "$TEST_TRIALS" \
    --run_id_note "MultiStage" \
    --use_wandb False &

PYTHON_PID=$!
echo "Python 实验进程 PID: $PYTHON_PID"

# =============================================================================
# 监控阶段完成并备份数据库
# =============================================================================
echo ""
echo "=========================================="
echo "步骤 3: 监控阶段完成并备份数据库"
echo "=========================================="

# 获取结果目录（Python脚本会创建）
RESULTS_BASE_DIR="$SPECVLA_ROOT/openvla/specdecoding/test-speed/${TASK_SUITE}_Spec_Online_Memory_MultiStage"

# 等待Python进程创建结果目录
echo "等待结果目录创建..."
for i in {1..60}; do
    if [ -d "$RESULTS_BASE_DIR" ]; then
        echo "✓ 结果目录已创建: $RESULTS_BASE_DIR"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "警告: 60秒后仍未找到结果目录，将不进行自动备份"
        echo "Python 进程可能还在初始化..."
    fi
    sleep 1
done

# 监控各个阶段的reload请求和完成标记
for STAGE_TARGET in "${WARMUP_STAGES[@]}"; do
    RELOAD_MARKER="$RESULTS_BASE_DIR/.stage_${STAGE_TARGET}_reload_needed"
    STAGE_MARKER="$RESULTS_BASE_DIR/.stage_${STAGE_TARGET}_complete"
    
    echo ""
    echo "=========================================="
    echo "监控 Stage Warmup=${STAGE_TARGET}"
    echo "=========================================="
    
    # 1. 先等待reload标记（warmup完成，需要重启服务）
    echo "等待 Stage ${STAGE_TARGET} Warmup完成（reload请求）..."
    WAIT_COUNT=0
    MAX_WAIT=7200  # 2小时
    
    while [ ! -f "$RELOAD_MARKER" ]; do
        if ! kill -0 $PYTHON_PID 2>/dev/null; then
            echo "警告: Python 进程已退出，停止监控"
            break 2
        fi
        
        sleep 10
        WAIT_COUNT=$((WAIT_COUNT + 10))
        
        if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
            echo "警告: 等待 Stage ${STAGE_TARGET} 超时（2小时），跳过"
            continue 2
        fi
        
        if [ $((WAIT_COUNT % 60)) -eq 0 ]; then
            echo "  已等待 $((WAIT_COUNT / 60)) 分钟..."
        fi
    done
    
    echo "✓ Stage ${STAGE_TARGET} Warmup完成，开始重启retrieval服务..."
    
    # 2. 重启retrieval服务（重新加载内存，但不恢复数据库）
    echo "停止旧的retrieval服务..."
    RETRIEVAL_PID=$(lsof -ti:5002 2>/dev/null)
    if [ -n "$RETRIEVAL_PID" ]; then
        kill -SIGTERM $RETRIEVAL_PID 2>/dev/null || true
        sleep 2
        if kill -0 $RETRIEVAL_PID 2>/dev/null; then
            kill -SIGKILL $RETRIEVAL_PID 2>/dev/null || true
            sleep 1
        fi
        echo "  ✓ 已停止旧服务"
    fi
    
    echo "直接启动Python retrieval服务（不恢复数据库，只重新加载内存）..."
    echo "  [重要] 保留Qdrant中已插入的Stage ${STAGE_TARGET}轨迹"
    
    # 切换到 rt-mzh 环境（retrieval_libero_goal.py 需要 qdrant_client）
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate rt-mzh
    
    # 直接调用Python服务，不经过shell包装脚本
    export CUDA_VISIBLE_DEVICES=1
    python3 "$RTCACHE_ROOT/scripts/retrieval/retrieval_libero_goal.py" \
        --host "0.0.0.0" \
        --port 5002 \
        --embedding-url "http://127.0.0.1:9020/predict" \
        --qdrant-host "localhost" \
        --qdrant-port 6333 \
        --log-level "INFO" \
        --dataset-types "$TASK_SUITE" \
        > /tmp/retrieval_service_stage${STAGE_TARGET}.log 2>&1 &
    
    # 切回 specvla 环境（Python 实验脚本需要）
    conda activate specvla
    
    RETRIEVAL_NEW_PID=$!
    echo "  等待服务启动..."
    
    for i in {1..120}; do
        if curl -s --connect-timeout 2 http://127.0.0.1:5002/health > /dev/null 2>&1; then
            echo "  ✓ Retrieval服务已就绪（等待了 $i 秒）"
            break
        fi
        if [ $i -eq 120 ]; then
            echo "  ✗ 服务启动超时！"
            tail -30 /tmp/retrieval_service_stage${STAGE_TARGET}.log
            exit 1
        fi
        if [ $((i % 5)) -eq 0 ]; then
            echo -n "."
        fi
        sleep 1
    done
    
    # 3. 通知Python脚本服务已重载
    RELOAD_COMPLETE_MARKER="$RESULTS_BASE_DIR/.stage_${STAGE_TARGET}_reload_complete"
    touch "$RELOAD_COMPLETE_MARKER"
    echo "✓ 服务重载完成，已通知Python脚本继续测试"
    
    # 4. 等待测试阶段完成
    echo ""
    echo "等待 Stage ${STAGE_TARGET} 测试完成..."
    WAIT_COUNT=0
    
    while [ ! -f "$STAGE_MARKER" ]; do
        if ! kill -0 $PYTHON_PID 2>/dev/null; then
            echo "警告: Python 进程已退出"
            break 2
        fi
        
        sleep 10
        WAIT_COUNT=$((WAIT_COUNT + 10))
        
        if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
            echo "警告: 等待测试完成超时"
            continue 2
        fi
        
        if [ $((WAIT_COUNT % 60)) -eq 0 ]; then
            echo "  已等待 $((WAIT_COUNT / 60)) 分钟..."
        fi
    done
    
    echo "✓ Stage ${STAGE_TARGET} 测试完成！"
    
    # 5. 备份数据库
    echo "开始备份数据库为 'base+${STAGE_TARGET}'..."
    
    # 切换到rt-mzh环境执行备份
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate rt-mzh
    
    # 执行备份
    python "$BACKUP_SCRIPT" --note "base+${STAGE_TARGET}" --backup-dir "$RTCACHE_ROOT/scripts/retrieval/qdrant_backups"
    
    if [ $? -eq 0 ]; then
        echo "✓ 数据库已备份为 'base+${STAGE_TARGET}'"
    else
        echo "✗ 备份失败！"
    fi
    
    # 切回specvla环境
    conda activate specvla
    
    # 6. 如果不是最后一个 stage，为下一个 stage 恢复数据库
    # 找到当前 stage 在数组中的索引
    CURRENT_STAGE_INDEX=-1
    for i in "${!WARMUP_STAGES[@]}"; do
        if [ "${WARMUP_STAGES[$i]}" = "$STAGE_TARGET" ]; then
            CURRENT_STAGE_INDEX=$i
            break
        fi
    done
    
    NEXT_STAGE_INDEX=$((CURRENT_STAGE_INDEX + 1))
    
    if [ $NEXT_STAGE_INDEX -lt ${#WARMUP_STAGES[@]} ]; then
        NEXT_STAGE_TARGET=${WARMUP_STAGES[$NEXT_STAGE_INDEX]}
        echo ""
        echo "=========================================="
        echo "准备下一个 Stage: Warmup=${NEXT_STAGE_TARGET}"
        echo "=========================================="
        echo "停止retrieval服务以便恢复数据库..."
        
        # 停止服务
        RETRIEVAL_PID=$(lsof -ti:5002 2>/dev/null)
        if [ -n "$RETRIEVAL_PID" ]; then
            kill -SIGTERM $RETRIEVAL_PID 2>/dev/null || true
            sleep 2
            if kill -0 $RETRIEVAL_PID 2>/dev/null; then
                kill -SIGKILL $RETRIEVAL_PID 2>/dev/null || true
                sleep 1
            fi
            echo "  ✓ 服务已停止"
        fi
        
        # 恢复上一个 stage 的备份（base+5 → base+10 → base+20...）
        echo "恢复数据库从 'base+${STAGE_TARGET}' 备份..."
        
        # 切换到rt-mzh环境执行恢复
        conda activate rt-mzh
        
        # 使用固定命名的备份目录
        BACKUP_ROOT="$RTCACHE_ROOT/scripts/retrieval/qdrant_backups"
        BACKUP_TO_RESTORE="$BACKUP_ROOT/backup_base+${STAGE_TARGET}"
        
        if [ ! -d "$BACKUP_TO_RESTORE" ]; then
            echo "  ✗ 未找到备份目录: $BACKUP_TO_RESTORE"
            exit 1
        fi
        
        echo "  使用备份目录: $BACKUP_TO_RESTORE"
        
        python "$RTCACHE_ROOT/scripts/retrieval/restore_qdrant.py" \
            --backup-dir "$BACKUP_TO_RESTORE" \
            --qdrant-host "localhost" \
            --qdrant-port 6333 \
            --force
        
        if [ $? -eq 0 ]; then
            echo "  ✓ 数据库已恢复为 'base+${STAGE_TARGET}'"
        else
            echo "  ✗ 数据库恢复失败！"
            exit 1
        fi
        
        # 切回specvla环境
        conda activate specvla
        
        # 重启 retrieval 服务（直接启动Python服务，加载恢复后的数据）
        echo "直接启动Python retrieval服务（加载恢复后的 'base+${STAGE_TARGET}' 数据）..."
        
        # 切换到 rt-mzh 环境
        conda activate rt-mzh
        
        export CUDA_VISIBLE_DEVICES=1
        python3 "$RTCACHE_ROOT/scripts/retrieval/retrieval_libero_goal.py" \
            --host "0.0.0.0" \
            --port 5002 \
            --embedding-url "http://127.0.0.1:9020/predict" \
            --qdrant-host "localhost" \
            --qdrant-port 6333 \
            --log-level "INFO" \
            --dataset-types "$TASK_SUITE" \
            > /tmp/retrieval_service_next_stage.log 2>&1 &
        
        # 切回 specvla 环境
        conda activate specvla
        
        echo "  等待服务启动..."
        for i in {1..120}; do
            if curl -s --connect-timeout 2 http://127.0.0.1:5002/health > /dev/null 2>&1; then
                echo "  ✓ Retrieval服务已就绪（等待了 $i 秒）"
                break
            fi
            if [ $i -eq 120 ]; then
                echo "  ✗ 服务启动超时！"
                tail -30 /tmp/retrieval_service_next_stage.log
                exit 1
            fi
            if [ $((i % 5)) -eq 0 ]; then
                echo -n "."
            fi
            sleep 1
        done
        
        echo "✓ 已准备好 Stage $((NEXT_STAGE_INDEX + 1)) (Warmup=${NEXT_STAGE_TARGET})"
    else
        echo ""
        echo "✓ 所有 stage 已完成，无需恢复下一个数据库"
    fi
    
    echo "继续监控下一阶段..."
done

# =============================================================================
# 等待Python进程完成
# =============================================================================
echo ""
echo "等待所有阶段完成..."
wait $PYTHON_PID
PYTHON_EXIT_CODE=$?

# =============================================================================
# 检查运行结果
# =============================================================================
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 多阶段实验完成！"
    echo "=========================================="
    echo "结果文件保存在:"
    echo "  $RESULTS_BASE_DIR"
    echo ""
    echo "实验总结:"
    echo "  - 累进式 Warmup: ${WARMUP_STAGES[@]}"
    echo "  - 每阶段测试: 每个任务 $TEST_TRIALS 次"
    echo "  - 数据库备份: base+5, base+10, base+20, base+30, base+40, base+50"
    echo "  - 执行模式: Spec + DB (1:1 alternating)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ 实验失败！退出码: $PYTHON_EXIT_CODE"
    echo "=========================================="
    echo "请检查错误信息并重试"
    exit 1
fi
    if [ -n "$RETRIEVAL_PID" ]; then
        echo "  找到 retrieval 服务进程 PID: $RETRIEVAL_PID"
        kill -SIGTERM $RETRIEVAL_PID 2>/dev/null || true
        sleep 2
        
        # 检查是否还在运行
        if kill -0 $RETRIEVAL_PID 2>/dev/null; then
            echo "  进程未响应，强制终止..."
            kill -SIGKILL $RETRIEVAL_PID 2>/dev/null || true
            sleep 1
        fi
        echo "  ✓ 已停止旧的 retrieval 服务"
    fi
fi
