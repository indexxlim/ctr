# CTR (Click-Through Rate) Prediction Model

(A) 구체적 아키텍처 다이어그램 + API 설계 (서빙 엔드포인트, 피처 스펙, 레이턴시 예측)
(B) 멀티태스크 Transformer 모델의 PyTorch/TF 구현 템플릿 (학습 루프, 캘리브레이션 포함)
(C) 오프라인 OPE + DR/IPS 예제 코드 (로그를 넣으면 새로운 정책 가치 추정)
(D) 컨텍스트 밴딧(Thompson Sampling) + Lagrangian 예산 제약 입찰 파이프라인 코드 샘플

## Data Preprocessing & Normalization

### 데이터 정규화의 중요성

딥러닝 모델, 특히 CTR 예측에서 데이터 정규화는 성능 향상에 핵심적인 역할을 합니다.

#### Dense Features에 StandardScaler 적용하는 이유:

1. **Gradient 안정성**
   - 스케일이 다른 features들이 gradient 계산에 불균형을 만듦
   - 정규화하면 모든 features가 동일한 스케일로 학습에 기여

2. **학습 속도 향상**
   - 균일한 스케일로 optimizer가 더 효율적으로 수렴
   - Learning rate 조정이 더 쉬워짐

3. **Weight 초기화 효과**
   - Xavier/He 초기화가 정규화된 입력에서 더 잘 작동
   - 활성화 함수의 saturation 방지

4. **Batch Normalization과의 시너지**
   - 입력 정규화 + Batch Norm으로 더 안정적인 학습

#### CTR 모델에서 특별히 중요한 이유:

- **Dense features**: 보통 매우 다양한 범위 (조회수, 클릭수, 가격 등)
- **Sparse features**: 임베딩으로 처리되므로 별도 정규화 불필요
- **Multi-task learning**: 서로 다른 태스크 간 균형 유지

### 사용 방법

```bash
# 데이터 전처리 및 train/val/test 분리
python train.py --prepare-data

# 일반 훈련 (기존 데이터 있으면 로드, 없으면 자동 생성)
python train.py
```

전처리된 데이터는 `./data/processed/` 디렉토리에 저장됩니다:
- `train_data.parquet`: 훈련 데이터 (정규화 완료)
- `val_data.parquet`: 검증 데이터 (정규화 완료)
- `test_data.parquet`: 테스트 데이터 (정규화 완료)
- `preprocessors.pkl`: LabelEncoders와 StandardScaler

## TODO List - CTR 모델 개선 사항

### 🎯 High Priority

- [ ] **Feature Interaction 구현**
  - [ ] Cross features 추가
  - [ ] Factorization Machines (FM) 레이어 구현
  - [ ] DeepFM 또는 xDeepFM 아키텍처 적용

- [ ] **Class Imbalance 처리**
  - [ ] Focal Loss 구현 (CTR 데이터는 보통 클릭률 1-5%)
  - [ ] Weighted Loss 추가
  - [ ] SMOTE 또는 다른 샘플링 기법 검토

- [ ] **Model Calibration (보정)**
  - [ ] Platt Scaling 구현
  - [ ] Isotonic Regression 추가
  - [ ] 예측 확률과 실제 클릭률 일치도 평가

### 🔧 Medium Priority

- [ ] **Training 최적화**
  - [ ] Early Stopping 구현
  - [ ] Model Checkpointing 시스템
  - [ ] Learning Rate Scheduling 개선 (CosineAnnealing, ReduceLROnPlateau)
  - [ ] Gradient Clipping 추가

- [ ] **Regularization 강화**
  - [ ] L2 Regularization 추가
  - [ ] Dropout rate 하이퍼파라미터 튜닝
  - [ ] Batch Normalization 레이어 검토

- [ ] **Embedding 최적화**
  - [ ] Embedding dimension 자동 조정 (rule-based)
  - [ ] Hash Embedding for high cardinality features
  - [ ] Embedding 초기화 방법 최적화

### 🚀 Advanced Features

- [ ] **Negative Sampling**
  - [ ] 대규모 categorical features 효율적 처리
  - [ ] Hierarchical Softmax 검토

- [ ] **Multi-Task Learning 확장**
  - [ ] CVR (Conversion Rate) 예측 태스크 추가
  - [ ] CTR + CVR joint training (MMOE, PLE 등)
  - [ ] Task-specific 가중치 학습

- [ ] **Advanced Architectures**
  - [ ] Attention 메커니즘 추가
  - [ ] Feature Selection 자동화
  - [ ] Neural Architecture Search (NAS) 적용 검토

### 📊 Monitoring & Evaluation

- [ ] **평가 지표 확장**
  - [ ] Calibration metrics (Brier Score, Reliability Diagram)
  - [ ] Business metrics (Revenue, ROAS)
  - [ ] A/B Test framework 준비

- [ ] **모델 해석성**
  - [ ] Feature Importance 분석
  - [ ] SHAP values 계산
  - [ ] 모델 예측 결과 시각화
