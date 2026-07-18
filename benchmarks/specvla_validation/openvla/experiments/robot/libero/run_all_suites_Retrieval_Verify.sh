#!/bin/bash

# Master control script for Retrieval Verification experiments across all task suites
# Usage: ./run_all_suites_Retrieval_Verify.sh [DB_MODEL_RATIO]
# Example: ./run_all_suites_Retrieval_Verify.sh "2 3"  (2 DB steps, 3 Model steps)

set -e  # Exit on any error

# Configuration
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1

# DB Model ratio (default "1 0" - only DB)
DB_MODEL_RATIO="${1:-1 0}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${SCRIPT_DIR}/../../../../TGT_DIR"
SUMMARY_DIR="${LOG_ROOT}/retrieval_verify_summary"
DB_BACKUP_BASE="/path/to/rtcache/scripts/retrieval/qdrant_backups/backup_base"
DB_RESTORE_SCRIPT="/path/to/rtcache/scripts/retrieval/restore_qdrant.py"

# Create summary directory
mkdir -p "${SUMMARY_DIR}"
TIMESTAMP=$(date +%Y_%m_%d-%H_%M_%S)
SUMMARY_LOG="${SUMMARY_DIR}/summary_${TIMESTAMP}.txt"

echo "======================================================================"
echo "Retrieval Verification Experiment Suite"
echo "Timestamp: ${TIMESTAMP}"
echo "DB:Model Ratio: ${DB_MODEL_RATIO}"
echo "======================================================================"

# Task suites to run
SUITES=("goal")
declare -A SUITE_SCRIPTS=(
    ["goal"]="run_libero_goal_Retrieval_Verify.py"
    ["object"]="run_libero_object_Retrieval_Verify.py"
    ["spatial"]="run_libero_spatial_Retrieval_Verify.py"
    ["10"]="run_libero_10_Retrieval_Verify.py"
)

# Conda setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate specvla

# Check services
echo "Checking service availability..."

# Check Retrieval API
if ! curl -s http://127.0.0.1:5002/health > /dev/null 2>&1; then
    echo "WARNING: Retrieval API (port 5002) not responding"
    echo "Please start the retrieval service first"
    exit 1
fi

# Check Embedding Server
if ! curl -s http://127.0.0.1:9020/health > /dev/null 2>&1; then
    echo "WARNING: Embedding server (port 9020) not responding"
    echo "Please start the embedding service first"
    exit 1
fi

# Check Qdrant
if ! curl -s http://localhost:6333 > /dev/null 2>&1; then
    echo "WARNING: Qdrant (port 6333) not responding"
    echo "Please start Qdrant first"
    exit 1
fi

echo "All services are available!"
echo ""

# Function to restore database
restore_database() {
    echo "Restoring database from base backup..."
    if [ -d "${DB_BACKUP_BASE}" ]; then
        # Switch to rt-mzh environment for database restoration
        set +e  # Temporarily disable exit on error
        conda activate rt-mzh
        python "${DB_RESTORE_SCRIPT}" --backup-dir "${DB_BACKUP_BASE}" --force
        RESTORE_EXIT_CODE=$?
        # Switch back to specvla environment
        conda activate specvla
        set -e  # Re-enable exit on error
        if [ $RESTORE_EXIT_CODE -eq 0 ]; then
            echo "Database restored successfully"
        else
            echo "WARNING: Database restoration failed with exit code $RESTORE_EXIT_CODE"
            echo "Continuing anyway..."
        fi
    else
        echo "WARNING: Backup base directory not found: ${DB_BACKUP_BASE}"
        echo "Continuing without database restoration..."
    fi
}

# Initialize summary log
cat > "${SUMMARY_LOG}" << EOF
====================================================================
Retrieval Verification Experiment Summary
Timestamp: ${TIMESTAMP}
====================================================================

Task Suites: ${SUITES[@]}
Accept Threshold: 9

EOF

# Run each suite
declare -A SUITE_RESULTS

for suite in "${SUITES[@]}"; do
    echo "======================================================================"
    echo "Running Task Suite: libero_${suite}"
    echo "======================================================================"
    
    # Restore database to base before each suite
    restore_database
    sleep 5  # Wait for database to stabilize
    
    # Create log directory for this suite
    SUITE_LOG_DIR="${LOG_ROOT}/retrieval_verify_${suite}_${TIMESTAMP}"
    mkdir -p "${SUITE_LOG_DIR}"
    
    # Run the experiment
    SCRIPT_NAME="${SUITE_SCRIPTS[$suite]}"
    echo "Running: python ${SCRIPT_NAME} --db_model_ratio ${DB_MODEL_RATIO}"
    echo "Logs will be saved to: ${SUITE_LOG_DIR}"
    
    START_TIME=$(date +%s)
    
    # Run experiment and capture log file path
    cd "${SCRIPT_DIR}"
    if ! python "${SCRIPT_NAME}" --db_model_ratio "${DB_MODEL_RATIO}" 2>&1 | tee "${SUITE_LOG_DIR}/console_output.txt"; then
        echo "ERROR: Task suite libero_${suite} failed!"
        SUITE_RESULTS[$suite]="FAILED"
        
        # Add failure to summary log
        echo "" >> "${SUMMARY_LOG}"
        echo "Task Suite: libero_${suite} - FAILED" >> "${SUMMARY_LOG}"
        echo "See ${SUITE_LOG_DIR}/console_output.txt for details" >> "${SUMMARY_LOG}"
        
        # Continue to next suite instead of exiting
        echo ""
        continue
    fi
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo "Task suite libero_${suite} completed in ${DURATION} seconds"
    echo ""
    
    # Find the log file that was just created (most recent one matching the pattern)
    # Use a timestamp-based approach to find files created after START_TIME
    LATEST_LOG=$(find "${LOG_ROOT}" -name "EVAL-libero_${suite}-*.txt" -newermt "@${START_TIME}" 2>/dev/null | sort | tail -1)
    
    if [ -z "${LATEST_LOG}" ]; then
        # Fallback: try to find the most recent log file
        LATEST_LOG=$(ls -t "${LOG_ROOT}"/EVAL-libero_${suite}-*.txt 2>/dev/null | head -1)
    fi
    
    if [ -n "${LATEST_LOG}" ]; then
        # Copy log file to suite directory
        cp "${LATEST_LOG}" "${SUITE_LOG_DIR}/"
        
        # Extract statistics from log file
        echo "Extracting statistics from: ${LATEST_LOG}"
        
        # Get success rate
        SR=$(grep -oP "Success rate: \K[0-9.]+%" "${LATEST_LOG}" 2>/dev/null || echo "N/A")
        
        # Get accept length mean
        ACCEPT_MEAN=$(grep -A 4 "Accept Length Statistics (all tasks):" "${LATEST_LOG}" | grep "Mean:" | grep -oP "Mean: \K[0-9.]+" 2>/dev/null || echo "N/A")
        
        # Get total episodes
        TOTAL_EPISODES=$(grep -oP "Total episodes: \K[0-9]+" "${LATEST_LOG}" 2>/dev/null || echo "N/A")
        
        SUITE_RESULTS[$suite]="${SR} | ${ACCEPT_MEAN} | ${TOTAL_EPISODES} | ${DURATION}s"
        
        echo "libero_${suite}: SR=${SR}, Accept Mean=${ACCEPT_MEAN}, Episodes=${TOTAL_EPISODES}, Time=${DURATION}s"
    else
        echo "WARNING: No log file found for libero_${suite}"
        SUITE_RESULTS[$suite]="No log file found"
    fi
    
    echo ""
    sleep 2
done

# Generate summary report
echo "======================================================================"
echo "Generating Summary Report"
echo "======================================================================"

cat >> "${SUMMARY_LOG}" << EOF

Results Summary:
====================================================================
Suite          | SR      | Accept Mean | Episodes | Duration
--------------------------------------------------------------------
EOF

for suite in "${SUITES[@]}"; do
    printf "libero_%-8s | %s\n" "${suite}" "${SUITE_RESULTS[$suite]}" >> "${SUMMARY_LOG}"
done

cat >> "${SUMMARY_LOG}" << EOF
====================================================================

Note: SR = Success Rate, Accept Mean = Average Accept Length

Individual Suite Logs:
EOF

for suite in "${SUITES[@]}"; do
    SUITE_LOG_DIR="${LOG_ROOT}/retrieval_verify_${suite}_${TIMESTAMP}"
    echo "  libero_${suite}: ${SUITE_LOG_DIR}" >> "${SUMMARY_LOG}"
done

cat >> "${SUMMARY_LOG}" << EOF

====================================================================
Experiment completed at: $(date)
====================================================================
EOF

# Display summary
echo ""
echo "======================================================================"
echo "All Experiments Completed!"
echo "======================================================================"
cat "${SUMMARY_LOG}"
echo ""
echo "Summary saved to: ${SUMMARY_LOG}"
echo "======================================================================"
