#!/bin/bash

# Script to run a single model
# Usage: ./run_single_model.sh <model_name> [gpu_id] [expid] [gpu_threshold] [max_wait_time]
#   model_name: Name of the model to run (required)
#   gpu_id: GPU device ID (-1 for CPU, 0+ for GPU) [default: -1]
#   expid: Experiment ID [default: none]
#   gpu_threshold: GPU memory usage threshold in % [default: 10]
#   max_wait_time: Maximum wait time in seconds [default: 3600]
# Example: ./run_single_model.sh Baseline 0 DeepFM_test 20 7200

if [ -z "$1" ]; then
    echo "Usage: $0 <model_name> [gpu_id] [expid] [gpu_threshold] [max_wait_time]"
    echo ""
    echo "Arguments:"
    echo "  model_name: Name of the model to run (required)"
    echo "  gpu_id: GPU device ID [default: -1 (CPU)]"
    echo "  expid: Experiment ID [default: none]"
    echo "  gpu_threshold: GPU memory threshold % [default: 10]"
    echo "  max_wait_time: Max wait time in seconds [default: 3600]"
    echo ""
    echo "Available models:"
    echo "  Baseline"
    echo "  Baseline_DoubleStream"
    echo "  Baseline_SA"
    echo "  DCNv2"
    echo "  FM"
    echo "  FinalMLP"
    echo "  FinalNet"
    echo "  LR"
    echo "  MaskNet"
    echo "  xDeepFM"
    exit 1
fi

MODEL=$1
GPU=${2:--1}
EXPID=${3:-""}
GPU_THRESHOLD=${4:-10}  # GPU memory threshold in percentage
MAX_WAIT_TIME=${5:-3600}  # Max wait time in seconds (default 1 hour)

BASE_DIR="/home/lim/project/RE-SORT/SOTAS"
MODEL_DIR="${BASE_DIR}/${MODEL}"

# Output directory
OUTPUT_DIR="${BASE_DIR}/results"
mkdir -p "${OUTPUT_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MODEL_LOG="${OUTPUT_DIR}/${MODEL}_${TIMESTAMP}.log"
MODEL_SUMMARY="${OUTPUT_DIR}/${MODEL}_${TIMESTAMP}_summary.txt"

if [ ! -d "${MODEL_DIR}" ]; then
    echo "Error: Model directory ${MODEL_DIR} does not exist"
    exit 1
fi

cd "${MODEL_DIR}"

if [ ! -f "run_expid.py" ]; then
    echo "Error: run_expid.py not found in ${MODEL_DIR}"
    exit 1
fi

echo "========================================" | tee "${MODEL_SUMMARY}"
echo "Configuration:" | tee -a "${MODEL_SUMMARY}"
echo "  Model: ${MODEL}" | tee -a "${MODEL_SUMMARY}"
echo "  GPU ID: ${GPU} $([ ${GPU} -eq -1 ] && echo '(CPU mode)' || echo '(GPU mode)')" | tee -a "${MODEL_SUMMARY}"
[ -n "${EXPID}" ] && echo "  Experiment ID: ${EXPID}" | tee -a "${MODEL_SUMMARY}"
echo "  GPU Memory Threshold: ${GPU_THRESHOLD}%" | tee -a "${MODEL_SUMMARY}"
echo "  Max Wait Time: ${MAX_WAIT_TIME}s ($(($MAX_WAIT_TIME / 60)) minutes)" | tee -a "${MODEL_SUMMARY}"
echo "  Working Directory: $(pwd)" | tee -a "${MODEL_SUMMARY}"
echo "  Log File: ${MODEL_LOG}" | tee -a "${MODEL_SUMMARY}"
echo "========================================" | tee -a "${MODEL_SUMMARY}"
echo "" | tee -a "${MODEL_SUMMARY}"

# Check GPU availability if GPU is specified
if [ ${GPU} -ge 0 ]; then
    echo "Checking GPU ${GPU} availability..." | tee -a "${MODEL_SUMMARY}"

    # First check if GPU is accessible
    CHECK_OUTPUT=$("${BASE_DIR}/check_gpu.sh" ${GPU} ${GPU_THRESHOLD} 2>&1)
    CHECK_EXIT=$?

    # If check_gpu.sh failed with ERROR (not just busy), abort
    if [ ${CHECK_EXIT} -ne 0 ] && echo "${CHECK_OUTPUT}" | grep -q "ERROR:"; then
        echo "${CHECK_OUTPUT}" | tee -a "${MODEL_SUMMARY}"
        echo "" | tee -a "${MODEL_SUMMARY}"
        echo "GPU check failed. Please fix GPU issues or use CPU mode." | tee -a "${MODEL_SUMMARY}"
        echo "To run in CPU mode: $0 ${MODEL}" | tee -a "${MODEL_SUMMARY}"
        exit 1
    fi

    # GPU is accessible, wait for it to be available
    WAIT_COUNT=0
    WAIT_INTERVAL=60  # Check every 60 seconds

    while true; do
        # Capture output and exit code separately
        GPU_CHECK_OUTPUT=$("${BASE_DIR}/check_gpu.sh" ${GPU} ${GPU_THRESHOLD} 2>&1)
        GPU_CHECK_EXIT=$?

        echo "${GPU_CHECK_OUTPUT}" | tee -a "${MODEL_SUMMARY}"

        if [ ${GPU_CHECK_EXIT} -eq 0 ]; then
            echo "GPU ${GPU} is available. Starting training..." | tee -a "${MODEL_SUMMARY}"
            echo "" | tee -a "${MODEL_SUMMARY}"
            break
        else
            WAIT_COUNT=$((WAIT_COUNT + WAIT_INTERVAL))

            if [ ${WAIT_COUNT} -ge ${MAX_WAIT_TIME} ]; then
                echo "Timeout: GPU ${GPU} not available after ${MAX_WAIT_TIME}s" | tee -a "${MODEL_SUMMARY}"
                echo "Exiting..." | tee -a "${MODEL_SUMMARY}"
                exit 1
            fi

            echo "Waiting for GPU ${GPU} to be available... (${WAIT_COUNT}s / ${MAX_WAIT_TIME}s)" | tee -a "${MODEL_SUMMARY}"
            sleep ${WAIT_INTERVAL}
        fi
    done
fi

START_TIME=$(date +%s)

if [ -z "${EXPID}" ]; then
    python run_expid.py --config ./config/ --gpu ${GPU} 2>&1 | tee "${MODEL_LOG}"
else
    python run_expid.py --config ./config/ --expid ${EXPID} --gpu ${GPU} 2>&1 | tee "${MODEL_LOG}"
fi

EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "" | tee -a "${MODEL_SUMMARY}"
echo "======================================" | tee -a "${MODEL_SUMMARY}"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ ${MODEL} completed successfully in ${DURATION}s" | tee -a "${MODEL_SUMMARY}"
else
    echo "✗ ${MODEL} failed with exit code ${EXIT_CODE}" | tee -a "${MODEL_SUMMARY}"
fi
echo "======================================" | tee -a "${MODEL_SUMMARY}"

# Extract and display results
echo "" | tee -a "${MODEL_SUMMARY}"
echo "Results:" | tee -a "${MODEL_SUMMARY}"
echo "--------" | tee -a "${MODEL_SUMMARY}"
grep -E "(Validation|Test|logloss|AUC)" "${MODEL_LOG}" | tail -20 | tee -a "${MODEL_SUMMARY}"
echo "" | tee -a "${MODEL_SUMMARY}"

echo "Summary saved to: ${MODEL_SUMMARY}" | tee -a "${MODEL_SUMMARY}"
echo "Full log saved to: ${MODEL_LOG}" | tee -a "${MODEL_SUMMARY}"
