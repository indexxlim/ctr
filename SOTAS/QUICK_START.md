# SOTAS Models - Quick Start

## 🔧 첫 실행 시 (에러 발생 시)

스크립트 실행 에러가 발생하면 먼저 수정:

```bash
cd /home/lim/project/RE-SORT/SOTAS

# 자동 수정
./fix_scripts.sh

# 또는 수동으로
sed -i 's/\r$//' *.sh && chmod +x *.sh
```

---

## ⚡ 가장 빠른 시작 방법

```bash
cd /home/lim/project/RE-SORT/SOTAS

# 1. GPU 확인
./check_gpu.sh 0

# 2. 모든 모델 실행 (GPU가 여유 있을 때 자동 시작)
./run_all_models.sh 0

# 3. 결과 분석
python analyze_results.py
```

---

## 📋 기본값 (아무것도 입력 안 하면)

| 파라미터 | 기본값 | 의미 |
|---------|--------|------|
| gpu_id | -1 | CPU 모드 |
| gpu_threshold | 10% | GPU 메모리 10% 미만일 때 시작 |
| max_wait_time | 3600초 | 최대 1시간 대기 |

---

## 🎯 주요 명령어

### GPU 상태 확인
```bash
./check_gpu.sh 0                    # GPU 0 확인 (threshold 10%)
./check_gpu.sh 0 20                 # GPU 0 확인 (threshold 20%)
```

### 모든 모델 실행
```bash
./run_all_models.sh                 # CPU 모드
./run_all_models.sh 0               # GPU 0, threshold 10%, 1시간 대기
./run_all_models.sh 0 20            # GPU 0, threshold 20%, 1시간 대기
./run_all_models.sh 0 10 7200       # GPU 0, threshold 10%, 2시간 대기
```

### 개별 모델 실행
```bash
./run_single_model.sh Baseline      # CPU 모드
./run_single_model.sh Baseline 0    # GPU 0
```

### 결과 분석
```bash
python analyze_results.py           # 모든 결과 비교
cat results/model_comparison.csv    # CSV로 확인
```

---

## ⚙️ 실행 시 화면 출력

```
========================================
Configuration:
  GPU ID: 0 (GPU mode)
  GPU Memory Threshold: 10%
  Max Wait Time: 3600s (60 minutes)
  Results Directory: /home/lim/project/RE-SORT/SOTAS/results
========================================

Checking GPU 0 availability...
GPU 0 Status:
  Memory: 29978MB / 32607MB (91%)
  Utilization: 72%
  Status: Busy (memory usage 91% >= 10%)
Waiting for GPU 0 to be available... (60s / 3600s)
...
GPU 0 is available. Starting training...
==================================
```

---

## 🔧 문제 해결

### GPU를 못 찾는 경우
```bash
# CPU 모드로 실행
./run_all_models.sh -1
./run_single_model.sh Baseline
```

### GPU가 계속 바쁜 경우
```bash
# threshold를 높여서 실행
./run_all_models.sh 0 30            # 30% 미만일 때 시작

# 또는 다른 GPU 사용
./run_all_models.sh 1               # GPU 1 사용
```

### 권한 문제
```bash
chmod +x *.sh
```

---

## 📁 결과 파일 위치

```
results/
├── summary_20231007_120000.txt         # 전체 요약
├── Baseline_20231007_120000.log        # 개별 로그
└── model_comparison.csv                # 비교 테이블
```

---

## 📚 더 자세한 정보

- 상세 문서: [README.md](README.md)
- 사용 예시: [USAGE_SUMMARY.md](USAGE_SUMMARY.md)
