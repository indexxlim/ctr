#conda install nvidia/label/cuda-11.8.0::cuda
#conda install anaconda::cudnn
#!/bin/bash
# Merlin 환경 설치 스크립트

echo "🔧 Merlin 환경 설치 시작..."

# 1. conda 환경 생성
echo "1️⃣ conda 환경 생성 (Python 3.10)"
#conda create -n merlin python=3.10 -y

# 2. CUDA 12 toolkit 및 런타임 설치 (conda 환경 내에서만)
echo "2️⃣ CUDA 12 toolkit 및 런타임 설치 (conda 환경 전용)"
conda install -n merlin -c conda-forge cudatoolkit=12.0 cudnn -y
conda install -n merlin -c nvidia cuda-cudart=12.0 -y

# 3. 기본 패키지 먼저 설치 (RAPIDS 전에)
echo "3️⃣ 기본 패키지 설치"
/home/lim/anaconda3/envs/merlin/bin/pip install \
    numpy==1.24.4 \
    pandas==2.0.3 \
    scipy==1.11.4 \
    scikit-learn==1.3.2

# 4. GPU Computing 패키지 설치 (CUDA 12)
echo "4️⃣ GPU Computing 패키지 설치 (CUDA 12)"
/home/lim/anaconda3/envs/merlin/bin/pip install \
    cupy-cuda12x \
    --extra-index-url=https://pypi.nvidia.com

# 5. cuDF, cuML 설치 (CUDA 12)
echo "5️⃣ cuDF, cuML 설치"
/home/lim/anaconda3/envs/merlin/bin/pip install \
    cudf-cu12 \
    cuml-cu12 \
    dask-cudf-cu12 \
    --extra-index-url=https://pypi.nvidia.com

echo "6️⃣ XGBoost 설치"
/home/lim/anaconda3/envs/merlin/bin/pip install xgboost==2.0.3

echo "7️⃣ Deep Learning 패키지 설치"
/home/lim/anaconda3/envs/merlin/bin/pip install tensorflow==2.13.0

echo "9️⃣ Visualization 패키지 설치"
/home/lim/anaconda3/envs/merlin/bin/pip install \
    matplotlib==3.7.2 \
    seaborn==0.13.0

echo "🔟 Jupyter 환경 설치"
/home/lim/anaconda3/envs/merlin/bin/pip install \
    jupyter==1.0.0 \
    jupyterlab==4.0.9 \
    ipywidgets==8.1.1

echo "1️⃣1️⃣ Utilities 설치"
/home/lim/anaconda3/envs/merlin/bin/pip install \
    tqdm==4.66.1 \
    pyarrow==12.0.1 \
    fastparquet==2023.10.1

echo "1️⃣2️⃣ Additional Dependencies 설치"
/home/lim/anaconda3/envs/merlin/bin/pip install \
    protobuf==4.21.12 \
    typing-extensions==4.5.0 \
    fsspec==2023.10.0 \
    dask==2023.9.2 \
    distributed==2023.9.2

# 13. npy-append-array 패치 (선택사항 - 에러 발생 시에만 실행)
# echo "1️⃣3️⃣ npy-append-array 패치 적용"
# /home/lim/anaconda3/envs/merlin/bin/python /tmp/fix_npy.py

echo ""
echo "✅ 설치 완료!"
echo ""
echo "📦 설치된 패키지:"
echo "  - Python 3.10"
echo "  - NumPy 1.24.3"
echo "  - Pandas 2.0.3"
echo "  - Scikit-learn (latest)"
echo "  - CuPy (CUDA 12)"
echo "  - cuDF-cu12 (CUDA 12 compatible)"
echo "  - cuML-cu12 (CUDA 12 compatible)"
echo "  - XGBoost 2.0.3 (GPU)"
echo "  - TensorFlow 2.13.0"
echo "  - Jupyter Lab 4.0.9"
echo ""
echo "⚠️  주의사항:"
echo "  - RTX 5090 (compute capability 12.0)은 CUDA 12 필수"
echo "  - nvtabular은 CUDA 12와 호환되지 않음"
echo "  - 대신 cuDF를 직접 사용하여 데이터 전처리"
echo ""
echo "🚀 실행 방법:"
echo "  conda activate merlin"
echo "  python merlin_train_cudf.py  # nvtabular 대신 cuDF 직접 사용"
