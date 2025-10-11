# Quick Start Guide

## 1. GPU 사용량 확인

```bash
cd /home/lim/project/RE-SORT/SOTAS
./check_gpu.sh 0
```

## 2. 모든 모델 실행 (GPU가 여유 있을 때까지 대기)

```bash
# GPU 0 사용, 메모리 사용량 10% 미만일 때 시작
./run_all_models.sh 0

# 또는 threshold 조정 (20% 미만일 때 시작)
./run_all_models.sh 0 20

# 최대 대기 시간 지정 (2시간)
./run_all_models.sh 0 10 7200
```

## 3. 개별 모델 실행

```bash
# Baseline 모델만 실행
./run_single_model.sh Baseline 0

# 특정 experiment ID로 실행
./run_single_model.sh Baseline 0 DeepFM_test
```

## 4. 결과 분석

```bash
# 모든 로그 파일에서 결과 추출 및 비교
python analyze_results.py
```

## 5. 결과 확인

```bash
# 요약 결과
cat results/summary_*.txt | tail -50

# 모델 비교
cat results/model_comparison.csv

# 개별 모델 로그
cat results/Baseline_*.log
```

---

## 주요 기능

### ✅ 자동 GPU 대기
- GPU 사용량이 threshold 이하로 떨어질 때까지 자동 대기
- 60초마다 체크
- 최대 대기 시간 설정 가능

### ✅ 실시간 출력 + 파일 저장
- 화면에서 실시간으로 결과 확인
- 동시에 로그 파일에 저장

### ✅ 자동 결과 추출
- 각 모델 실행 후 자동으로 metrics 추출
- 요약 파일 생성

### ✅ 비교 분석
- 모든 모델 결과 자동 비교
- 최고 성능 모델 자동 식별

---

## 파일 구조

```
SOTAS/
├── check_gpu.sh              # GPU 사용량 체크
├── run_all_models.sh         # 모든 모델 순차 실행
├── run_single_model.sh       # 개별 모델 실행
├── analyze_results.py        # 결과 분석 스크립트
├── README.md                 # 상세 문서
├── USAGE_SUMMARY.md          # 이 파일
└── results/                  # 실행 결과 저장
    ├── summary_*.txt         # 전체 요약
    ├── detailed_log_*.log    # 상세 로그
    ├── ModelName_*.log       # 개별 모델 로그
    └── model_comparison.csv  # 모델 비교 테이블
```

---

## 예시 워크플로우

```bash
cd /home/lim/project/RE-SORT/SOTAS

# 1. GPU 상태 확인
./check_gpu.sh 0

# 2. 모든 모델 실행 (GPU가 여유 있을 때까지 대기)
./run_all_models.sh 0 10 7200  # 2시간까지 대기

# 3. 실행이 끝난 후 결과 분석
python analyze_results.py

# 4. 최고 성능 모델 확인
cat results/model_comparison.csv
```

---

## GPU가 바쁠 때 출력 예시

```
Running all models with GPU=0
Results will be saved to: /home/lim/project/RE-SORT/SOTAS/results

Checking GPU 0 availability...
GPU 0 Status:
  Memory: 29884MB / 32607MB (91%)
  Utilization: 88%
  Status: Busy (memory usage 91% >= 10%)
Waiting for GPU 0 to be available... (60s / 7200s)
Waiting for GPU 0 to be available... (120s / 7200s)
...
GPU 0 is available. Starting training...
==================================
```
