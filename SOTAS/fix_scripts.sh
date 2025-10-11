#!/bin/bash

# Script to fix common issues with shell scripts

echo "================================================"
echo "Shell Scripts Fix Utility"
echo "================================================"
echo ""

cd "$(dirname "$0")"

echo "현재 디렉토리: $(pwd)"
echo ""

# 1. Fix line endings (CRLF -> LF)
echo "1. 줄바꿈 문자 수정 (CRLF -> LF)..."
for file in *.sh; do
    if [ -f "$file" ]; then
        sed -i 's/\r$//' "$file"
        echo "  ✓ $file"
    fi
done
echo ""

# 2. Add execute permissions
echo "2. 실행 권한 추가..."
chmod +x *.sh 2>/dev/null
for file in *.sh; do
    if [ -f "$file" ] && [ -x "$file" ]; then
        echo "  ✓ $file"
    fi
done
echo ""

# 3. Verify file format
echo "3. 파일 형식 확인..."
for file in check_gpu.sh run_all_models.sh run_single_model.sh; do
    if [ -f "$file" ]; then
        format=$(file "$file" | grep -o "CRLF" || echo "LF")
        if [ "$format" = "LF" ]; then
            echo "  ✓ $file: OK (Unix format)"
        else
            echo "  ✗ $file: $format (수정 필요)"
        fi
    fi
done
echo ""

# 4. Test GPU check
echo "4. GPU 확인..."
if [ -x "./check_gpu.sh" ]; then
    if ./check_gpu.sh 0 2>/dev/null; then
        echo "  ✓ GPU 사용 가능"
    else
        if command -v nvidia-smi &> /dev/null; then
            echo "  ⚠ GPU가 바쁘거나 사용 중입니다"
            echo "    해결: threshold를 높이거나 CPU 모드 사용"
            echo "    예: ./run_all_models.sh 0 30"
            echo "    또는: ./run_all_models.sh -1"
        else
            echo "  ⚠ nvidia-smi를 찾을 수 없습니다"
            echo "    해결: CPU 모드로 실행"
            echo "    예: ./run_all_models.sh -1"
        fi
    fi
else
    echo "  ✗ check_gpu.sh를 실행할 수 없습니다"
fi
echo ""

# 5. Summary
echo "================================================"
echo "수정 완료!"
echo "================================================"
echo ""
echo "다음 명령어로 실행할 수 있습니다:"
echo ""
echo "  # GPU 상태 확인"
echo "  ./check_gpu.sh 0"
echo ""
echo "  # 모든 모델 실행"
echo "  ./run_all_models.sh 0"
echo ""
echo "  # 개별 모델 실행"
echo "  ./run_single_model.sh Baseline 0"
echo ""
echo "  # CPU 모드로 실행"
echo "  ./run_all_models.sh -1"
echo ""
