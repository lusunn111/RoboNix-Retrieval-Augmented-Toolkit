#!/bin/bash
# Start RT-Cache Retrieval Server for LIBERO-Goal

# Default configurations
HOST="0.0.0.0"
PORT=5002
EMBEDDING_URL="http://127.0.0.1:9020/predict"
QDRANT_HOST="localhost"
QDRANT_PORT=6333
LOG_LEVEL="INFO"
DATASET_TYPES="goal"  # comma-separated or 'all'
SKIP_RESTORE=false    # Skip database restore (for reload scenarios)

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
echo "Starting RT-Cache Retrieval Server"
echo "=========================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Embedding URL: $EMBEDDING_URL"
echo "Qdrant: $QDRANT_HOST:$QDRANT_PORT"
echo "Log Level: $LOG_LEVEL"
echo "Dataset Types: $DATASET_TYPES"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate conda environment FIRST (restore_qdrant.py needs qdrant_client)
if command -v conda >/dev/null 2>&1; then
    # Initialize conda for this shell
    eval "$(conda shell.bash hook)"
    # Activate requested environment
    conda activate rt-mzh || {
        echo "[ERROR] Failed to activate conda env 'rt-mzh'. Please ensure it exists."
        exit 1
    }
    echo "[INFO] Conda environment activated: $(conda info --envs | awk '/\*/{print $1}')"
else
    echo "[WARNING] 'conda' command not found. Skipping environment activation."
fi

# Conditionally restore Qdrant database from backup
if [ "$SKIP_RESTORE" = false ]; then
    echo "[INFO] Restoring Qdrant database from backup..."
    # Backup is in the same directory as the script
    BACKUP_DIR="$SCRIPT_DIR/qdrant_backups/latest"

    if [ ! -d "$BACKUP_DIR" ]; then
        echo "[ERROR] Backup directory does not exist: $BACKUP_DIR"
        echo "[ERROR] Please run backup script first: cd $SCRIPT_DIR && ./backup_libero_goal.sh"
        exit 1
    fi

    python3 "$SCRIPT_DIR/restore_qdrant.py" \
        --backup-dir "$BACKUP_DIR" \
        --qdrant-host "$QDRANT_HOST" \
        --qdrant-port "$QDRANT_PORT" \
        --force || {
        echo "[ERROR] Failed to restore database. Aborting."
        exit 1
    }
    echo "[INFO] Database restore completed."
else
    echo "[INFO] Skipping database restore (--skip-restore flag set)"
    echo "[INFO] Will use existing data in Qdrant"
fi
echo "=========================================="

# Pin GPU device to 1 (first visible device becomes index 0 inside process)
export CUDA_VISIBLE_DEVICES=1
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Run the server
python3 "$SCRIPT_DIR/retrieval_libero_goal.py" \
    --host "$HOST" \
    --port "$PORT" \
    --embedding-url "$EMBEDDING_URL" \
    --qdrant-host "$QDRANT_HOST" \
    --qdrant-port "$QDRANT_PORT" \
    --log-level "$LOG_LEVEL" \
    --dataset-types "$DATASET_TYPES"
