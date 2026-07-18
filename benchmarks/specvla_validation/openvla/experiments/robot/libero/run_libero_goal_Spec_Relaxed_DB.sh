#!/bin/bash
# 运行脚本：使用DB检索和Speculative Decoding交替执行机制
# 使用方法: bash run_libero_goal_Spec_Relaxed_DB.sh [--db_steps N] [--model_steps M]
#
# 前置条件：
# 1. 确保 DB retrieval 服务器正在运行 (http://127.0.0.1:5002/pipeline)
# 2. 确保 embedding 服务器正在运行 (http://127.0.0.1:9020/predict)
# 3. 确保 Qdrant 数据库正在运行 (localhost:6333)

set -e  # 遇到错误立即退出

# =============================================================================
# 可选功能：成功轨迹在线插入向量数据库（Qdrant）
# =============================================================================
# 说明：当 episode 成功 (done=True) 后，将本次执行轨迹写入 Qdrant，
#      schema 与 rtcache/scripts/data_processing/process_libero_goal.py 对齐。
#
# 开关：设置 ENABLE_ONLINE_DB_INSERT=true 即可开启
ENABLE_ONLINE_DB_INSERT=${ENABLE_ONLINE_DB_INSERT:-false}
ONLINE_DB_DATASET_NAME=${ONLINE_DB_DATASET_NAME:-"specvla_online"}
ONLINE_DB_QDRANT_URL=${ONLINE_DB_QDRANT_URL:-"http://127.0.0.1:6333"}
ONLINE_DB_EMBEDDING_URL=${ONLINE_DB_EMBEDDING_URL:-"http://127.0.0.1:9020/predict"}
ONLINE_DB_INSERT_STRIDE=${ONLINE_DB_INSERT_STRIDE:-1}
ONLINE_DB_INSERT_MAX_STEPS=${ONLINE_DB_INSERT_MAX_STEPS:--1}
ONLINE_DB_UPSERT_BATCH_SIZE=${ONLINE_DB_UPSERT_BATCH_SIZE:-16}

# 设置工作目录（根据实际情况修改）
SPECVLA_ROOT="/path/to/SpecVLA"
cd $SPECVLA_ROOT

# 检查目录是否存在
if [ ! -d "$SPECVLA_ROOT" ]; then
    echo "错误: SpecVLA 根目录不存在: $SPECVLA_ROOT"
    exit 1
fi

# 解析命令行参数
DB_STEPS=1
MODEL_STEPS=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --db_steps)
            DB_STEPS="$2"
            shift 2
            ;;
        --model_steps)
            MODEL_STEPS="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: bash run_libero_goal_Spec_Relaxed_DB.sh [--db_steps N] [--model_steps M]"
            exit 1
            ;;
    esac
done

# 激活conda环境
echo "激活 conda 环境: specvla"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate specvla

# 检查 conda 环境是否激活成功
if [ "$CONDA_DEFAULT_ENV" != "specvla" ]; then
    echo "警告: conda 环境可能未正确激活，当前环境: $CONDA_DEFAULT_ENV"
fi

# 设置环境变量
export PYTHONPATH=$PWD:$PWD/openvla:$PWD/LIBERO
export MUJOCO_GL=egl
export ROBOSUITE_LOG_FILE=$SPECVLA_ROOT/robosuite.log
export CUDA_VISIBLE_DEVICES=1  # 根据你的GPU情况修改
export MUJOCO_EGL_DEVICE_ID=1   # 根据你的GPU情况修改

# 检查必要的文件是否存在
PRETRAINED_CHECKPOINT="$SPECVLA_ROOT/backbone_models/openvla-7b-finetuned-libero-goal"
SPEC_CHECKPOINT="$SPECVLA_ROOT/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190"

if [ ! -d "$PRETRAINED_CHECKPOINT" ]; then
    echo "警告: 预训练模型检查点不存在: $PRETRAINED_CHECKPOINT"
    echo "请检查路径是否正确"
fi

if [ ! -d "$SPEC_CHECKPOINT" ]; then
    echo "警告: SpecVLA 检查点不存在: $SPEC_CHECKPOINT"
    echo "请检查路径是否正确"
fi

# 检查 DB retrieval 服务器是否运行
echo "检查 DB retrieval 服务器..."
if ! curl -s --connect-timeout 2 http://127.0.0.1:5002/health > /dev/null 2>&1; then
    echo "警告: DB retrieval 服务器可能未运行 (http://127.0.0.1:5002)"
    echo "请确保服务器正在运行，否则程序可能会失败"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ DB retrieval 服务器正在运行"
fi

# 如果开启在线插库，额外检查 embedding server 和 Qdrant
if [[ "$ENABLE_ONLINE_DB_INSERT" == "true" || "$ENABLE_ONLINE_DB_INSERT" == "True" || "$ENABLE_ONLINE_DB_INSERT" == "1" ]]; then
    echo "检查 embedding 服务器..."
    EMBEDDING_HEALTH_URL="${ONLINE_DB_EMBEDDING_URL%/predict}/health"
    if ! curl -s --connect-timeout 2 "$EMBEDDING_HEALTH_URL" > /dev/null 2>&1; then
        echo "警告: embedding 服务器可能未运行 ($EMBEDDING_HEALTH_URL)"
        echo "请确保服务器正在运行，否则在线插库会失败"
        read -p "是否继续? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "✓ embedding 服务器正在运行"
    fi

    echo "检查 Qdrant 数据库..."
    QDRANT_CHECK_URL="${ONLINE_DB_QDRANT_URL%/}/collections"
    if ! curl -s --connect-timeout 2 "$QDRANT_CHECK_URL" > /dev/null 2>&1; then
        echo "警告: Qdrant 数据库可能未运行 ($QDRANT_CHECK_URL)"
        echo "请先在 /path/to/rtcache 目录运行 ./start_db.sh"
        read -p "是否继续? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "✓ Qdrant 数据库正在运行"
    fi
fi

# 打印配置信息
echo "=========================================="
echo "运行配置:"
echo "  SpecVLA 根目录: $SPECVLA_ROOT"
echo "  预训练模型: $PRETRAINED_CHECKPOINT"
echo "  SpecVLA 检查点: $SPEC_CHECKPOINT"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  DB Retrieval: 启用"
echo "  Speculative Decoding: 启用"
echo "  DB Steps (N): $DB_STEPS"
echo "  Model Steps (M): $MODEL_STEPS"
echo "  Online DB Insert: $ENABLE_ONLINE_DB_INSERT"
echo "=========================================="
echo ""

# 运行脚本
echo "开始运行评估..."
ONLINE_INSERT_ARGS=()
if [[ "$ENABLE_ONLINE_DB_INSERT" == "true" || "$ENABLE_ONLINE_DB_INSERT" == "True" || "$ENABLE_ONLINE_DB_INSERT" == "1" ]]; then
    ONLINE_INSERT_ARGS+=(--enable_online_db_insert True)
    ONLINE_INSERT_ARGS+=(--online_db_dataset_name "$ONLINE_DB_DATASET_NAME")
    ONLINE_INSERT_ARGS+=(--online_db_qdrant_url "$ONLINE_DB_QDRANT_URL")
    ONLINE_INSERT_ARGS+=(--online_db_embedding_server_url "$ONLINE_DB_EMBEDDING_URL")
    ONLINE_INSERT_ARGS+=(--online_db_insert_stride "$ONLINE_DB_INSERT_STRIDE")
    ONLINE_INSERT_ARGS+=(--online_db_insert_max_steps "$ONLINE_DB_INSERT_MAX_STEPS")
    ONLINE_INSERT_ARGS+=(--online_db_upsert_batch_size "$ONLINE_DB_UPSERT_BATCH_SIZE")
fi

python openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed_DB.py \
    --model_family openvla \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --spec_checkpoint "$SPEC_CHECKPOINT" \
    --task_suite_name libero_goal \
    --center_crop True \
    --use_spec True \
    --parallel_draft False \
    --accept_threshold 9 \
    --db_steps $DB_STEPS \
    --model_steps $MODEL_STEPS \
    --num_trials_per_task 10 \
    --run_id_note "Spec_Relaxed_DB_N${DB_STEPS}_M${MODEL_STEPS}" \
    --use_wandb False \
    "${ONLINE_INSERT_ARGS[@]}"

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 运行完成！"
    echo "=========================================="
    echo "结果文件保存在: $SPECVLA_ROOT/openvla/specdecoding/test-speed/libero_goal_Spec_DB_N${DB_STEPS}_M${MODEL_STEPS}"
else
    echo ""
    echo "=========================================="
    echo "✗ 运行失败！"
    echo "=========================================="
    echo "请检查错误信息并重试"
    exit 1
fi
