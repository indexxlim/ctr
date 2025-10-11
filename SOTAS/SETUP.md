# 환경 설정 가이드

## 필수 설정

### 1. Python 패키지 설치

```bash
cd /home/lim/project/RE-SORT

# requirements 설치
pip install -r requirements.txt

# 추가 패키지 (필요시)
pip install pyarrow fastparquet
```

### 2. Python Path 설정

RE-SORT 프로젝트의 `re_sortctr` 모듈을 사용하려면 Python 경로를 설정해야 합니다.

#### 방법 1: 환경 변수 설정 (권장)

```bash
# 현재 세션에만 적용
export PYTHONPATH="/home/lim/project/RE-SORT:$PYTHONPATH"

# 또는 ~/.bashrc에 추가하여 영구 적용
echo 'export PYTHONPATH="/home/lim/project/RE-SORT:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

#### 방법 2: 스크립트에서 자동 설정

각 모델의 `run_expid.py`는 이미 다음 코드로 경로를 설정합니다:

```python
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
# sys.path에 프로젝트 루트 추가 (필요시)
```

### 3. 확인

```bash
cd /home/lim/project/RE-SORT

# re_sortctr 모듈 확인
python -c "import sys; sys.path.insert(0, '.'); import re_sortctr; print('✓ re_sortctr OK')"

# re_sortctr_version 모듈 확인
python -c "import sys; sys.path.insert(0, '.'); import re_sortctr_version; print('✓ re_sortctr_version OK')"
```

---

## 실행 전 체크리스트

### ✅ 1단계: 스크립트 수정
```bash
cd /home/lim/project/RE-SORT/SOTAS
./fix_scripts.sh
```

### ✅ 2단계: Python 환경 확인
```bash
cd /home/lim/project/RE-SORT

# Python 버전 확인 (3.7+)
python --version

# 필수 패키지 확인
python -c "import torch; import pandas; import numpy; print('✓ Packages OK')"
```

### ✅ 3단계: 모듈 확인
```bash
# PYTHONPATH 설정
export PYTHONPATH="/home/lim/project/RE-SORT:$PYTHONPATH"

# 모듈 import 테스트
python -c "import re_sortctr; import re_sortctr_version; print('✓ Modules OK')"
```

### ✅ 4단계: 데이터 확인
```bash
ls -lh /home/lim/project/RE-SORT/data/

# 필요한 파일:
# - train.parquet (9,633,761 rows, 90%)
# - valid.parquet (1,070,418 rows, 10%)
# - test.parquet
```

### ✅ 5단계: GPU 확인 (선택)
```bash
cd /home/lim/project/RE-SORT/SOTAS
./check_gpu.sh 0
```

---

## 실행

모든 설정이 완료되면:

```bash
cd /home/lim/project/RE-SORT/SOTAS

# PYTHONPATH 설정
export PYTHONPATH="/home/lim/project/RE-SORT:$PYTHONPATH"

# GPU 모드로 실행
./run_all_models.sh 0

# 또는 CPU 모드
./run_all_models.sh -1
```

---

## 문제 해결

### ❌ ModuleNotFoundError: No module named 're_sortctr'

**해결:**
```bash
export PYTHONPATH="/home/lim/project/RE-SORT:$PYTHONPATH"
```

### ❌ ModuleNotFoundError: No module named 're_sortctr_version'

**해결:**
이 파일이 생성되어 있어야 합니다:
- `/home/lim/project/RE-SORT/re_sortctr_version.py`
- `/home/lim/project/RE-SORT/re_sortctr/__init__.py`

```bash
cd /home/lim/project/RE-SORT

# 확인
ls -la re_sortctr_version.py re_sortctr/__init__.py
```

### ❌ ImportError: No module named 'torch'

**해결:**
```bash
pip install torch pandas numpy scikit-learn PyYAML h5py tqdm
```

### ❌ FileNotFoundError: train.parquet not found

**해결:**
```bash
ls -la /home/lim/project/RE-SORT/data/

# 데이터 파일이 없으면 데이터 준비 필요
```

---

## 한 번에 설정하기

```bash
#!/bin/bash

cd /home/lim/project/RE-SORT

# 1. 패키지 설치
pip install -r requirements.txt

# 2. PYTHONPATH 설정
export PYTHONPATH="/home/lim/project/RE-SORT:$PYTHONPATH"
echo 'export PYTHONPATH="/home/lim/project/RE-SORT:$PYTHONPATH"' >> ~/.bashrc

# 3. 스크립트 수정
cd SOTAS
./fix_scripts.sh

# 4. 테스트
python -c "import re_sortctr; import re_sortctr_version; print('✓ Setup complete!')"

echo ""
echo "설정 완료! 이제 다음 명령어로 실행할 수 있습니다:"
echo "  ./run_all_models.sh 0    # GPU 모드"
echo "  ./run_all_models.sh -1   # CPU 모드"
```

---

## 참고

- 자세한 실행 방법: [README.md](README.md)
- 빠른 시작: [QUICK_START.md](QUICK_START.md)
- 문제 해결: [문제해결.md](문제해결.md)
