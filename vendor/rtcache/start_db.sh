#!/usr/bin/env bash
set -euo pipefail

# Start local MongoDB and Qdrant without Docker.
# Run from the rtcache repo root: ./start_db.sh

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DB_DIR="$BASE_DIR/DB"
MONGO_BIN="$DB_DIR/bin/mongodb/bin/mongod"
QDRANT_BIN="$DB_DIR/bin/qdrant"
MONGO_DATA="$DB_DIR/data/mongo"
QDRANT_DATA="$DB_DIR/data/qdrant"
QDRANT_CONFIG="$DB_DIR/qdrant_config.yaml"
QDRANT_PID_FILE="$QDRANT_DATA/qdrant.pid"
MONGO_LOG="$MONGO_DATA/mongod.log"
QDRANT_LOG="$QDRANT_DATA/qdrant.log"

mkdir -p "$MONGO_DATA" "$QDRANT_DATA"
chmod +x "$MONGO_BIN" "$QDRANT_BIN"

# Start MongoDB (forks to background)
"$MONGO_BIN" \
  --dbpath "$MONGO_DATA" \
  --bind_ip 0.0.0.0 \
  --port 27017 \
  --fork \
  --logpath "$MONGO_LOG"

echo "MongoDB started (log: $MONGO_LOG)"

# Start Qdrant (run in background via nohup)
nohup "$QDRANT_BIN" --config-path "$QDRANT_CONFIG" > "$QDRANT_LOG" 2>&1 &
QDRANT_PID=$!
echo $QDRANT_PID > "$QDRANT_PID_FILE"
echo "Qdrant started (pid: $QDRANT_PID, log: $QDRANT_LOG)"

echo "Databases are up. MongoDB on 27017, Qdrant on 6333/6334 (per config)."
