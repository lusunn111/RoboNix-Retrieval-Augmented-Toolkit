#!/bin/bash
# Backup Qdrant database for LIBERO experiments (goal, spatial, object, 10)

# Default configurations
QDRANT_HOST="localhost"
QDRANT_PORT=6333
BACKUP_DIR="./qdrant_backups"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --qdrant-host)
            QDRANT_HOST="$2"
            shift 2
            ;;
        --qdrant-port)
            QDRANT_PORT="$2"
            shift 2
            ;;
        --backup-dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Backing up Qdrant Database"
echo "=========================================="
echo "Qdrant: $QDRANT_HOST:$QDRANT_PORT"
echo "Backup Directory: $BACKUP_DIR"
echo "=========================================="

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

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run backup script
python3 "$SCRIPT_DIR/backup_qdrant.py" \
    --qdrant-host "$QDRANT_HOST" \
    --qdrant-port "$QDRANT_PORT" \
    --backup-dir "$BACKUP_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Backup completed successfully!"
    echo "Backup location: $BACKUP_DIR/latest"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Backup failed!"
    echo "=========================================="
    exit 1
fi
