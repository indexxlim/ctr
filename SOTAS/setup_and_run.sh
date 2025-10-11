#!/bin/bash

# Complete setup and run script
# This script sets up everything and runs the models

set -e  # Exit on error

echo "========================================"
echo "RE-SORT 모델 자동 설정 및 실행"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Base directories
PROJECT_ROOT="/home/lim/project/RE-SORT"
SOTAS_DIR="${PROJECT_ROOT}/SOTAS"

echo "1️⃣ 환경 확인 중..."
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python이 설치되지 않았습니다${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python: $(python --version)${NC}"

# Check if we're in the right directory
if [ ! -d "${PROJECT_ROOT}" ]; then
    echo -e "${RED}✗ 프로젝트 디렉토리를 찾을 수 없습니다: ${PROJECT_ROOT}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 프로젝트 디렉토리 확인${NC}"

# Check data files
if [ ! -f "${PROJECT_ROOT}/data/train.parquet" ]; then
    echo -e "${YELLOW}⚠ train.parquet를 찾을 수 없습니다${NC}"
else
    echo -e "${GREEN}✓ 데이터 파일 확인${NC}"
fi

echo ""
echo "2️⃣ Python 경로 설정 중..."
echo ""

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
echo -e "${GREEN}✓ PYTHONPATH=${PROJECT_ROOT}${NC}"

# Test imports
echo "모듈 import 테스트 중..."
cd "${PROJECT_ROOT}"

if python -c "import sys; sys.path.insert(0, '.'); import re_sortctr_version" 2>/dev/null; then
    echo -e "${GREEN}✓ re_sortctr_version 모듈 OK${NC}"
else
    echo -e "${YELLOW}⚠ re_sortctr_version 모듈을 찾을 수 없지만 계속 진행합니다${NC}"
fi

if python -c "import sys; sys.path.insert(0, '.'); import re_sortctr" 2>/dev/null; then
    echo -e "${GREEN}✓ re_sortctr 모듈 OK${NC}"
else
    echo -e "${YELLOW}⚠ re_sortctr 모듈을 찾을 수 없지만 계속 진행합니다${NC}"
fi

echo ""
echo "3️⃣ 스크립트 수정 중..."
echo ""

cd "${SOTAS_DIR}"

# Fix line endings
for file in *.sh; do
    if [ -f "$file" ]; then
        sed -i 's/\r$//' "$file"
        chmod +x "$file"
        echo -e "${GREEN}✓ $file${NC}"
    fi
done

echo ""
echo "4️⃣ GPU 상태 확인 중..."
echo ""

if [ -x "./check_gpu.sh" ]; then
    GPU_STATUS=$(./check_gpu.sh 0 2>&1)
    echo "$GPU_STATUS"

    if echo "$GPU_STATUS" | grep -q "Available"; then
        echo ""
        echo -e "${GREEN}✓ GPU 사용 가능!${NC}"
        GPU_MODE=0
    else
        echo ""
        echo -e "${YELLOW}⚠ GPU가 바쁩니다 (91% 사용 중)${NC}"
        echo ""
        echo "선택하세요:"
        echo "  1) CPU 모드로 바로 실행"
        echo "  2) GPU가 여유 있을 때까지 대기 (최대 2시간)"
        echo "  3) 취소"
        echo ""
        read -p "선택 (1-3): " choice

        case $choice in
            1)
                echo -e "${GREEN}→ CPU 모드로 실행합니다${NC}"
                GPU_MODE=-1
                ;;
            2)
                echo -e "${GREEN}→ GPU 대기 모드로 실행합니다 (최대 2시간)${NC}"
                GPU_MODE=0
                MAX_WAIT=7200
                ;;
            *)
                echo "취소되었습니다"
                exit 0
                ;;
        esac
    fi
else
    echo -e "${YELLOW}⚠ check_gpu.sh를 실행할 수 없습니다${NC}"
    echo "CPU 모드로 실행합니다"
    GPU_MODE=-1
fi

echo ""
echo "5️⃣ 모델 실행 시작..."
echo ""
echo "설정:"
echo "  GPU ID: ${GPU_MODE} $([ ${GPU_MODE} -eq -1 ] && echo '(CPU mode)' || echo '(GPU mode)')"
echo "  최대 대기: ${MAX_WAIT:-3600}초"
echo ""
read -p "실행하시겠습니까? (y/N): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "취소되었습니다"
    exit 0
fi

echo ""
echo "========================================"
echo "모델 학습 시작"
echo "========================================"
echo ""

# Run models
if [ ${GPU_MODE} -eq -1 ]; then
    ./run_all_models.sh -1
else
    ./run_all_models.sh ${GPU_MODE} 10 ${MAX_WAIT:-7200}
fi

EXIT_CODE=$?

echo ""
echo "========================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo -e "${GREEN}✓ 모든 모델 실행 완료!${NC}"
    echo ""
    echo "결과 분석:"
    echo "  python analyze_results.py"
    echo ""
    echo "결과 확인:"
    echo "  cat results/model_comparison.csv"
else
    echo -e "${RED}✗ 실행 중 에러가 발생했습니다${NC}"
    echo ""
    echo "로그 확인:"
    echo "  ls -lt results/*.log | head -5"
fi
echo "========================================"
