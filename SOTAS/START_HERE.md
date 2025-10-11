# 🚀 여기서 시작하세요!

## 1️⃣ 첫 실행 전 필수 설정 (5분)

```bash
cd /home/lim/project/RE-SORT

# Python 경로 설정 (필수!)
export PYTHONPATH="/home/lim/project/RE-SORT:$PYTHONPATH"

# 영구 적용하려면
echo 'export PYTHONPATH="/home/lim/project/RE-SORT:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc

# 스크립트 수정
cd SOTAS
./fix_scripts.sh
```

---

## 2️⃣ GPU 상태 확인

```bash
./check_gpu.sh 0
```

**출력 예시:**
```
GPU 0 Status:
  Memory: 29990MB / 32607MB (91%)
  Utilization: 46%
  Status: Busy (memory usage 91% >= 10%)
```

---

## 3️⃣ 모델 실행

### GPU가 busy한 경우 (현재 상태)

```bash
# 옵션 1: GPU가 여유 있을 때까지 대기 (2시간)
./run_all_models.sh 0 10 7200

# 옵션 2: threshold를 높여서 실행
./run_all_models.sh 0 30

# 옵션 3: CPU 모드로 바로 실행 (추천)
./run_all_models.sh -1
```

### GPU가 여유 있는 경우

```bash
./run_all_models.sh 0
```

---

## 4️⃣ 결과 확인

```bash
# 결과 분석
python analyze_results.py

# 요약 파일 확인
cat results/model_comparison.csv

# 최신 로그 확인
ls -ltr results/*.txt | tail -5
```

---

## ⚠️ 문제 발생 시

### "실행할 수 없음: 필요한 파일이 없습니다"
```bash
./fix_scripts.sh
```

### "No module named 're_sortctr'"
```bash
export PYTHONPATH="/home/lim/project/RE-SORT:$PYTHONPATH"
```

### "GPU is busy" 메시지만 나오고 실행 안 됨
**✅ 정상입니다!** GPU가 여유 있을 때까지 대기 중입니다.

다음과 같이 표시됩니다:
```
GPU 0 Status:
  Memory: 29990MB / 32607MB (91%)
  Status: Busy (memory usage 91% >= 10%)
Waiting for GPU 0 to be available... (60s / 7200s)
```

CPU 모드로 바로 실행하려면:
```bash
./run_all_models.sh -1
```

---

## 📚 더 자세한 정보

| 문서 | 내용 |
|------|------|
| [SETUP.md](SETUP.md) | 환경 설정 상세 가이드 |
| [QUICK_START.md](QUICK_START.md) | 빠른 시작 가이드 |
| [README.md](README.md) | 전체 문서 (영문) |
| [README_KR.md](README_KR.md) | 전체 문서 (한글) |
| [문제해결.md](문제해결.md) | 모든 에러 해결 방법 |
| [USAGE_SUMMARY.md](USAGE_SUMMARY.md) | 사용 예시 모음 |

---

## 🎯 현재 시스템 상태

- **GPU**: NVIDIA GeForce RTX 5090
- **메모리**: 29990MB / 32607MB (91% 사용 중)
- **상태**: Busy

**추천 명령어:**
```bash
# CPU 모드로 실행 (대기 없이 바로 시작)
./run_all_models.sh -1

# 또는 GPU가 여유 있을 때까지 대기
./run_all_models.sh 0 10 7200
```

---

## ✅ 완료 체크리스트

- [ ] PYTHONPATH 설정: `export PYTHONPATH="/home/lim/project/RE-SORT:$PYTHONPATH"`
- [ ] 스크립트 수정: `./fix_scripts.sh`
- [ ] GPU 확인: `./check_gpu.sh 0`
- [ ] 모델 실행: `./run_all_models.sh 0` 또는 `./run_all_models.sh -1`
- [ ] 결과 분석: `python analyze_results.py`

---

**문제가 있나요?** → [문제해결.md](문제해결.md)를 확인하세요!
