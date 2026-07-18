#!/bin/bash
# Process all LIBERO datasets (goal/10/object/spatial) for RT-Cache
# This script will sequentially process all four LIBERO dataset subsets

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Processing All LIBERO Datasets${NC}"
echo -e "${GREEN}========================================${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../.."

# Dataset types to process
DATASETS=("goal" "10" "object" "spatial")

# Track statistics
START_TIME=$(date +%s)
TOTAL_DATASETS=${#DATASETS[@]}
SUCCESS_COUNT=0
FAILED_DATASETS=()

# Process each dataset
for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    dataset_num=$((i + 1))
    
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}[$dataset_num/$TOTAL_DATASETS] Processing LIBERO-${dataset^^}${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    # Run processing script
    if python scripts/data_processing/process_libero_goal.py --dataset_type "$dataset"; then
        echo -e "${GREEN}✓ LIBERO-${dataset^^} processing completed successfully${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "${RED}✗ LIBERO-${dataset^^} processing failed${NC}"
        FAILED_DATASETS+=("$dataset")
    fi
done

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

# Print final summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Processing Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Total datasets: ${TOTAL_DATASETS}"
echo -e "${GREEN}Successful: ${SUCCESS_COUNT}${NC}"
echo -e "${RED}Failed: ${#FAILED_DATASETS[@]}${NC}"

if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo -e "${RED}Failed datasets: ${FAILED_DATASETS[*]}${NC}"
fi

echo ""
echo -e "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo -e "${GREEN}========================================${NC}"

# Exit with error if any dataset failed
if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    exit 1
fi

echo -e "${GREEN}All datasets processed successfully!${NC}"
