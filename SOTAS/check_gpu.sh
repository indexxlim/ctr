#!/bin/bash

# Script to check GPU availability and usage
# Usage: ./check_gpu.sh [gpu_id] [threshold]
#   gpu_id: GPU device ID [default: 0]
#   threshold: GPU memory usage threshold in % [default: 10]
# Returns 0 if GPU is available, 1 otherwise

GPU_ID=${1:-0}
THRESHOLD=${2:-10}

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. GPU monitoring not available."
    echo "Please install NVIDIA drivers or use CPU mode (gpu_id=-1)"
    exit 1
fi

# Check if nvidia-smi works (driver loaded)
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi found but cannot access GPU."
    echo "Possible causes:"
    echo "  - NVIDIA driver not loaded"
    echo "  - Permission issues"
    echo "  - No NVIDIA GPU available"
    echo ""
    echo "Run 'nvidia-smi' manually to see detailed error."
    echo "Or use CPU mode (gpu_id=-1)"
    exit 1
fi

# Check if specific GPU exists
if ! nvidia-smi -i ${GPU_ID} &> /dev/null; then
    echo "ERROR: GPU ${GPU_ID} not found."
    echo ""
    echo "Available GPUs:"
    nvidia-smi -L 2>/dev/null || echo "  No GPUs detected"
    exit 1
fi

# Get GPU memory usage percentage
MEMORY_USED=$(nvidia-smi -i ${GPU_ID} --query-gpu=memory.used --format=csv,noheader,nounits)
MEMORY_TOTAL=$(nvidia-smi -i ${GPU_ID} --query-gpu=memory.total --format=csv,noheader,nounits)
MEMORY_PERCENT=$((100 * MEMORY_USED / MEMORY_TOTAL))

# Get GPU utilization
GPU_UTIL=$(nvidia-smi -i ${GPU_ID} --query-gpu=utilization.gpu --format=csv,noheader,nounits)

echo "GPU ${GPU_ID} Status:"
echo "  Memory: ${MEMORY_USED}MB / ${MEMORY_TOTAL}MB (${MEMORY_PERCENT}%)"
echo "  Utilization: ${GPU_UTIL}%"

# Check if GPU is available (memory usage below threshold)
if [ ${MEMORY_PERCENT} -lt ${THRESHOLD} ]; then
    echo "  Status: Available ✓"
    exit 0
else
    echo "  Status: Busy (memory usage ${MEMORY_PERCENT}% >= ${THRESHOLD}%)"
    exit 1
fi
