#!/bin/bash
# Start RT-Cache Mix Retrieval Server for LIBERO (Third-Person + Wrist View)
#
# This script starts the retrieval server that uses mix view embeddings
# (4352 dims = DINOv2 + SigLIP from both third-person and wrist cameras)
#
# Usage:
#   ./start_libero_goal_retrieval_mix.sh                           # Default: goal dataset
#   ./start_libero_goal_retrieval_mix.sh --dataset-types all       # All datasets
#   ./start_libero_goal_retrieval_mix.sh --skip-restore            # Skip DB restore

# Default configurations
HOST="0.0.0.0"
PORT=5003                                          # Different from single-view (5002)
EMBEDDING_URL="http://127.0.0.1:9021/predict"      # Mix embedding server
QDRANT_HOST="localhost"
QDRANT_PORT=6333
LOG_LEVEL="INFO"
DATASET_TYPES="goal "
SKIP_RESTORE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --embedding-url)
            EMBEDDING_URL="$2"
            shift 2
            ;;
        --qdrant-host)
            QDRANT_HOST="$2"
            shift 2
            ;;
        --qdrant-port)
            QDRANT_PORT="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --dataset-types)
            DATASET_TYPES="$2"
            shift 2
            ;;
        --skip-restore)
            SKIP_RESTORE=true
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Starting RT-Cache Mix Retrieval Server"
echo "=========================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Mix Embedding URL: $EMBEDDING_URL"
echo "Qdrant: $QDRANT_HOST:$QDRANT_PORT"
echo "Log Level: $LOG_LEVEL"
echo "Dataset Types: $DATASET_TYPES"
echo "View Type: MIX (Third-Person + Wrist)"
echo "Embedding Dim: 4352"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate conda environment
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate rt-mzh || {
        echo "[ERROR] Failed to activate conda env 'rt-mzh'. Please ensure it exists."
        exit 1
    }
    echo "[INFO] Conda environment activated: $(conda info --envs | awk '/\*/{print $1}')"
else
    echo "[WARNING] 'conda' command not found. Skipping environment activation."
fi

# Conditionally restore Qdrant database from mix backup
if [ "$SKIP_RESTORE" = false ]; then
    echo "[INFO] Restoring Qdrant database from mix backup..."
    BACKUP_DIR="$SCRIPT_DIR/qdrant_backups/latest_mix"

    if [ ! -d "$BACKUP_DIR" ] && [ ! -L "$BACKUP_DIR" ]; then
        # Try alternative backup location
        BACKUP_DIR="$SCRIPT_DIR/qdrant_backups/mix_base"
    fi

    if [ ! -d "$BACKUP_DIR" ]; then
        echo "[WARNING] Mix backup directory does not exist: $BACKUP_DIR"
        echo "[WARNING] Continuing without restore. Make sure data is already in Qdrant."
        echo "[WARNING] To create mix backup, run: python process_libero_goal_mix.py --process_all --backup"
    else
        python3 "$SCRIPT_DIR/restore_qdrant.py" \
            --backup-dir "$BACKUP_DIR" \
            --qdrant-host "$QDRANT_HOST" \
            --qdrant-port "$QDRANT_PORT" \
            --force || {
            echo "[WARNING] Failed to restore database. Continuing anyway."
        }
        echo "[INFO] Database restore completed."
    fi
else
    echo "[INFO] Skipping database restore (--skip-restore flag set)"
    echo "[INFO] Will use existing data in Qdrant"
fi
echo "=========================================="

# Pin GPU device to 1
export CUDA_VISIBLE_DEVICES=1
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Run the mix retrieval server
python3 "$SCRIPT_DIR/retrieval_libero_goal_mix.py" \
    --host "$HOST" \
    --port "$PORT" \
    --embedding-url "$EMBEDDING_URL" \
    --qdrant-host "$QDRANT_HOST" \
    --qdrant-port "$QDRANT_PORT" \
    --log-level "$LOG_LEVEL" \
    --dataset-types "$DATASET_TYPES"
