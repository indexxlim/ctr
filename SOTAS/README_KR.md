# SOTAS 모델 학습 가이드 (한국어)

## 🚀 빠른 시작

### 1단계: 스크립트 수정 (첫 실행 시)

```bash
cd /home/lim/project/RE-SORT/SOTAS
./fix_scripts.sh
```

### 2단계: 실행

```bash
# GPU 확인
./check_gpu.sh 0

# 모든 모델 실행 (GPU가 여유 있을 때 자동 시작)
./run_all_models.sh 0

# 결과 분석
python analyze_results.py
```

---

## 📋 기본값

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| gpu_id | -1 | CPU 모드 |
| gpu_threshold | 10% | GPU 메모리 임계값 |
| max_wait_time | 3600초 | 최대 대기 시간 (1시간) |

---

## 💡 주요 명령어

### GPU 상태 확인
```bash
./check_gpu.sh 0                # GPU 0 확인
./check_gpu.sh 0 20             # threshold 20%로 확인
```

### 모든 모델 실행
```bash
./run_all_models.sh             # CPU 모드
./run_all_models.sh 0           # GPU 0, 기본 설정
./run_all_models.sh 0 20        # GPU 0, threshold 20%
./run_all_models.sh 0 10 7200   # GPU 0, 2시간까지 대기
```

### 개별 모델 실행
```bash
./run_single_model.sh Baseline          # CPU 모드
./run_single_model.sh Baseline 0        # GPU 0
./run_single_model.sh Baseline 0 "" 20  # GPU 0, threshold 20%
```

### 결과 분석
```bash
python analyze_results.py               # 모든 결과 비교
cat results/model_comparison.csv        # CSV 확인
```

---

## 🔧 문제 해결

### ❌ "실행할 수 없음: 필요한 파일이 없습니다"

**원인:** Windows 줄바꿈 (CRLF) 문제

**해결:**
```bash
./fix_scripts.sh

# 또는
sed -i 's/\r$//' *.sh
chmod +x *.sh
```

### ❌ GPU가 계속 busy

**현재 상태:**
```bash
./check_gpu.sh 0
# GPU 0 Status:
#   Memory: 29989MB / 32607MB (91%)
#   Status: Busy
```

**해결:**
```bash
# 옵션 1: threshold 높이기
./run_all_models.sh 0 30

# 옵션 2: CPU 모드
./run_all_models.sh -1

# 옵션 3: 다른 GPU
./run_all_models.sh 1
```

### ❌ nvidia-smi not found

**해결:**
```bash
# CPU 모드로 실행
./run_all_models.sh -1

# 또는 NVIDIA 드라이버 설치 (INSTALL_NVIDIA.md 참고)
```

---

## 📁 파일 구조

```
SOTAS/
├── fix_scripts.sh              # 🔧 스크립트 자동 수정
├── check_gpu.sh                # GPU 상태 확인
├── run_all_models.sh           # 모든 모델 실행
├── run_single_model.sh         # 개별 모델 실행
├── analyze_results.py          # 결과 분석
├── README.md                   # 영문 문서
├── README_KR.md                # 이 파일
├── QUICK_START.md              # 빠른 시작
├── USAGE_SUMMARY.md            # 사용 예시
├── 문제해결.md                  # 상세 문제 해결
├── INSTALL_NVIDIA.md           # NVIDIA 설치
└── results/                    # 결과 저장
```

---

## 📊 실행 화면 예시

### GPU 확인
```
$ ./check_gpu.sh 0
GPU 0 Status:
  Memory: 29989MB / 32607MB (91%)
  Utilization: 44%
  Status: Busy (memory usage 91% >= 10%)
```

### 모델 실행
```
$ ./run_all_models.sh 0
========================================
Configuration:
  GPU ID: 0 (GPU mode)
  GPU Memory Threshold: 10%
  Max Wait Time: 3600s (60 minutes)
  Results Directory: /home/lim/project/RE-SORT/SOTAS/results
========================================

Checking GPU 0 availability...
GPU 0 Status:
  Memory: 29989MB / 32607MB (91%)
  Utilization: 44%
  Status: Busy (memory usage 91% >= 10%)
Waiting for GPU 0 to be available... (60s / 3600s)
...
```

---

## 📚 추가 문서

- **[QUICK_START.md](QUICK_START.md)**: 가장 빠른 시작 방법
- **[USAGE_SUMMARY.md](USAGE_SUMMARY.md)**: 상세 사용 예시
- **[문제해결.md](문제해결.md)**: 모든 에러 해결 방법
- **[INSTALL_NVIDIA.md](INSTALL_NVIDIA.md)**: NVIDIA 드라이버 설치
- **[README.md](README.md)**: 영문 상세 문서

---

## 💡 팁

1. **첫 실행**: 반드시 `./fix_scripts.sh` 실행
2. **GPU 바쁨**: threshold를 30% 이상으로 설정
3. **빠른 테스트**: CPU 모드 사용 (`-1`)
4. **장시간 실행**: `nohup ./run_all_models.sh 0 > output.log 2>&1 &`
5. **도움말**: 인자 없이 실행 (`./run_single_model.sh`)

---

## 🎯 사용 예시

```bash
cd /home/lim/project/RE-SORT/SOTAS

# 1. 스크립트 수정
./fix_scripts.sh

# 2. GPU 확인
./check_gpu.sh 0

# 3. 모든 모델 실행 (GPU가 여유 있을 때)
./run_all_models.sh 0 10 7200  # 2시간까지 대기

# 4. 결과 분석
python analyze_results.py

# 5. 최고 성능 모델 확인
cat results/model_comparison.csv
```

---

## 현재 시스템 정보

- **GPU**: NVIDIA GeForce RTX 5090
- **Driver**: 580.82.09
- **CUDA**: 13.0
- **상태**: 사용 중 (91%)

**추천 명령어:**
```bash
# GPU가 여유 있을 때 자동 시작 (2시간 대기)
./run_all_models.sh 0 10 7200

# 또는 CPU 모드로 바로 시작
./run_all_models.sh -1
```
