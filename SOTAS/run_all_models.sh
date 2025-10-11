#!/bin/bash

# Script to run all models in SOTAS directory
# Usage: ./run_all_models.sh [gpu_id] [gpu_threshold] [max_wait_time]
#   gpu_id: GPU device ID (-1 for CPU, 0+ for GPU) [default: -1]
#   gpu_threshold: GPU memory usage threshold in % [default: 10]
#   max_wait_time: Maximum wait time in seconds [default: 3600]

# Base directory
BASE_DIR="/home/lim/project/RE-SORT/SOTAS"

# Output directory and files
OUTPUT_DIR="${BASE_DIR}/results"
mkdir -p "${OUTPUT_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="${OUTPUT_DIR}/summary_${TIMESTAMP}.txt"
DETAILED_LOG="${OUTPUT_DIR}/detailed_log_${TIMESTAMP}.log"

# Array of model directories that have run_expid.py
MODELS=(
    "Baseline"
    "Baseline_DoubleStream"
    "Baseline_SA"
    "DCNv2"
    "FM"
    "FinalMLP"
    "FinalNet"
    "LR"
    "MaskNet"
    "xDeepFM"
)

# Parse arguments with defaults
GPU=${1:--1}
GPU_THRESHOLD=${2:-10}
MAX_WAIT_TIME=${3:-3600}

echo "========================================" | tee "${SUMMARY_FILE}"
echo "Configuration:" | tee -a "${SUMMARY_FILE}"
echo "  GPU ID: ${GPU} $([ ${GPU} -eq -1 ] && echo '(CPU mode)' || echo '(GPU mode)')" | tee -a "${SUMMARY_FILE}"
echo "  GPU Memory Threshold: ${GPU_THRESHOLD}%" | tee -a "${SUMMARY_FILE}"
echo "  Max Wait Time: ${MAX_WAIT_TIME}s ($(($MAX_WAIT_TIME / 60)) minutes)" | tee -a "${SUMMARY_FILE}"
echo "  Results Directory: ${OUTPUT_DIR}" | tee -a "${SUMMARY_FILE}"
echo "========================================" | tee -a "${SUMMARY_FILE}"

# Check GPU availability if GPU is specified
if [ ${GPU} -ge 0 ]; then
    echo "" | tee -a "${SUMMARY_FILE}"
    echo "Checking GPU ${GPU} availability..." | tee -a "${SUMMARY_FILE}"

    # First check if GPU is accessible
    CHECK_OUTPUT=$("${BASE_DIR}/check_gpu.sh" ${GPU} ${GPU_THRESHOLD} 2>&1)
    CHECK_EXIT=$?

    # If check_gpu.sh failed with ERROR (not just busy), abort
    if [ ${CHECK_EXIT} -ne 0 ] && echo "${CHECK_OUTPUT}" | grep -q "ERROR:"; then
        echo "${CHECK_OUTPUT}" | tee -a "${SUMMARY_FILE}"
        echo "" | tee -a "${SUMMARY_FILE}"
        echo "GPU check failed. Please fix GPU issues or use CPU mode." | tee -a "${SUMMARY_FILE}"
        echo "To run in CPU mode: $0 -1" | tee -a "${SUMMARY_FILE}"
        exit 1
    fi

    # GPU is accessible, wait for it to be available
    WAIT_COUNT=0
    WAIT_INTERVAL=60  # Check every 60 seconds

    while true; do
        # Capture output and exit code separately
        GPU_CHECK_OUTPUT=$("${BASE_DIR}/check_gpu.sh" ${GPU} ${GPU_THRESHOLD} 2>&1)
        GPU_CHECK_EXIT=$?

        echo "${GPU_CHECK_OUTPUT}" | tee -a "${SUMMARY_FILE}"

        if [ ${GPU_CHECK_EXIT} -eq 0 ]; then
            echo "GPU ${GPU} is available. Starting training..." | tee -a "${SUMMARY_FILE}"
            break
        else
            WAIT_COUNT=$((WAIT_COUNT + WAIT_INTERVAL))

            if [ ${WAIT_COUNT} -ge ${MAX_WAIT_TIME} ]; then
                echo "Timeout: GPU ${GPU} not available after ${MAX_WAIT_TIME}s" | tee -a "${SUMMARY_FILE}"
                echo "Exiting..." | tee -a "${SUMMARY_FILE}"
                exit 1
            fi

            echo "Waiting for GPU ${GPU} to be available... (${WAIT_COUNT}s / ${MAX_WAIT_TIME}s)" | tee -a "${SUMMARY_FILE}"
            sleep ${WAIT_INTERVAL}
        fi
    done
fi

echo "==================================" | tee -a "${SUMMARY_FILE}"
echo "" | tee -a "${SUMMARY_FILE}"

# Track completed models
COMPLETED=0
FAILED=0

for model in "${MODELS[@]}"; do
    echo "" | tee -a "${SUMMARY_FILE}"
    echo "======================================" | tee -a "${SUMMARY_FILE}"
    echo "Running ${model}..." | tee -a "${SUMMARY_FILE}"
    echo "======================================" | tee -a "${SUMMARY_FILE}"

    cd "${BASE_DIR}/${model}"

    MODEL_LOG="${OUTPUT_DIR}/${model}_${TIMESTAMP}.log"

    if [ -f "run_expid.py" ]; then
        # Check if there's a default config
        if [ -d "config" ]; then
            # Run with train_parquet dataset
            START_TIME=$(date +%s)

            if python run_expid.py --config ./config/ --gpu ${GPU} 2>&1 | tee "${MODEL_LOG}"; then
                END_TIME=$(date +%s)
                DURATION=$((END_TIME - START_TIME))

                echo "✓ ${model} completed successfully in ${DURATION}s" | tee -a "${SUMMARY_FILE}"

                # Extract results from log
                echo "" | tee -a "${SUMMARY_FILE}"
                echo "Results for ${model}:" | tee -a "${SUMMARY_FILE}"
                grep -E "(Validation|Test|logloss|AUC)" "${MODEL_LOG}" | tail -20 | tee -a "${SUMMARY_FILE}"
                echo "" | tee -a "${SUMMARY_FILE}"

                COMPLETED=$((COMPLETED + 1))
            else
                echo "✗ ${model} failed" | tee -a "${SUMMARY_FILE}"
                FAILED=$((FAILED + 1))
            fi
        else
            echo "No config directory found for ${model}, skipping..." | tee -a "${SUMMARY_FILE}"
        fi
    else
        echo "No run_expid.py found for ${model}, skipping..." | tee -a "${SUMMARY_FILE}"
    fi

    # Append to detailed log
    echo "========================================" >> "${DETAILED_LOG}"
    echo "Model: ${model}" >> "${DETAILED_LOG}"
    echo "========================================" >> "${DETAILED_LOG}"
    cat "${MODEL_LOG}" >> "${DETAILED_LOG}" 2>/dev/null
    echo "" >> "${DETAILED_LOG}"
done

echo "" | tee -a "${SUMMARY_FILE}"
echo "==================================" | tee -a "${SUMMARY_FILE}"
echo "All models completed!" | tee -a "${SUMMARY_FILE}"
echo "Completed: ${COMPLETED}" | tee -a "${SUMMARY_FILE}"
echo "Failed: ${FAILED}" | tee -a "${SUMMARY_FILE}"
echo "" | tee -a "${SUMMARY_FILE}"
echo "Summary saved to: ${SUMMARY_FILE}" | tee -a "${SUMMARY_FILE}"
echo "Detailed log saved to: ${DETAILED_LOG}" | tee -a "${SUMMARY_FILE}"
