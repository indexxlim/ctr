#!/bin/bash

# Merlin 환경 패키지 설치 스크립트
# CUDA 12.0, Python 3.10 환경

# conda 환경 활성화
source ~/anaconda3/etc/profile.d/conda.sh
conda activate merlin

echo "Installing RAPIDS cuDF and related packages via conda..."
# Step 1: cuDF, cuPy 및 CUDA 관련 패키지 설치 (conda)
conda install -c rapidsai -c conda-forge -c nvidia \
    cudf=23.10 \
    python=3.10 \
    cuda-version=12.0 \
    -y

echo "Installing nvtabular via conda..."
# Step 2: nvtabular 및 merlin 패키지 설치 (conda)
conda install -c nvidia -c rapidsai -c conda-forge \
    nvtabular=23.08 \
    -y

echo "Installing machine learning packages via pip..."
# Step 3: XGBoost, scikit-learn 및 기타 Python 패키지 설치 (pip)
pip install \
    'xgboost>=3.0' \
    'scikit-learn==1.3.*' \
    'scipy>=1.11'

# Step 4: numpy 호환성 맞추기 (nvtabular이 numpy < 1.25 요구)
pip install 'numpy<1.25,>=1.24' --force-reinstall

echo "Fixing npy-append-array compatibility with numpy 1.24..."
# Step 5: npy-append-array 패치 (numpy 1.24+ 호환성)
python /home/lim/project/ctr/fix_npy_append_array.py

echo ""
echo "Installation complete!"
echo ""
echo "Installed package versions:"
conda list | grep -E "nvtabular|cudf|cupy|xgboost|dask|pandas|numpy|scikit-learn|psutil|pyarrow|scipy"

echo ""
echo "Testing imports..."
python -c "import nvtabular as nvt; import cudf; import xgboost as xgb; print('✓ nvtabular:', nvt.__version__); print('✓ cudf:', cudf.__version__); print('✓ xgboost:', xgb.__version__)"
