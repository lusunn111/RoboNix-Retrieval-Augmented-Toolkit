#!/usr/bin/env bash
set -euo pipefail

# Stop local MongoDB and Qdrant started by start_db.sh.
# Run from the rtcache repo root: ./stop_db.sh

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DB_DIR="$BASE_DIR/DB"
MONGO_BIN="$DB_DIR/bin/mongodb/bin/mongod"
MONGO_DATA="$DB_DIR/data/mongo"
QDRANT_DATA="$DB_DIR/data/qdrant"
QDRANT_PID_FILE="$QDRANT_DATA/qdrant.pid"

# Stop MongoDB
if pgrep -f "$MONGO_BIN" > /dev/null; then
  "$MONGO_BIN" --dbpath "$MONGO_DATA" --shutdown || true
  echo "MongoDB stopped."
else
  echo "MongoDB not running."
fi

# Stop Qdrant
if [[ -f "$QDRANT_PID_FILE" ]]; then
  QPID=$(cat "$QDRANT_PID_FILE" || true)
  if [[ -n "$QPID" ]] && kill -0 "$QPID" 2>/dev/null; then
    kill "$QPID" || true
    echo "Qdrant stopped (pid $QPID)."
  else
    echo "Qdrant pid file present but process not running."
  fi
  rm -f "$QDRANT_PID_FILE"
else
  # Fallback: try to kill by process name
  if pgrep -f "$BASE_DIR/DB/bin/qdrant" > 0; then
    pkill -f "$BASE_DIR/DB/bin/qdrant" || true
    echo "Qdrant processes killed by name."
  else
    echo "Qdrant not running."
  fi
fi

echo "Databases stopped."
