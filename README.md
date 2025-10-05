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

# RE-SORT 모델 훈련
python train.py --model resort

# Transformer 모델 훈련 (기본값)
python train.py --model transformer
```

전처리된 데이터는 `./data/processed/` 디렉토리에 저장됩니다:
- `train_data.parquet`: 훈련 데이터 (정규화 완료)
- `val_data.parquet`: 검증 데이터 (정규화 완료)
- `test_data.parquet`: 테스트 데이터 (정규화 완료)
- `preprocessors.pkl`: LabelEncoders와 StandardScaler

실험 결과는 모델 타입별로 저장됩니다:
- `./experiments/transformer_YYYYMMDD_HHMMSS/`: Transformer 모델 실험
- `./experiments/resort_YYYYMMDD_HHMMSS/`: RE-SORT 모델 실험

## Available Models

### 1. MultiTaskTransformer (기본 모델)
- Transformer 기반 CTR 예측 모델
- Adaptive embedding dimensions
- Shared MLP with task-specific heads

### 2. RE-SORT (Removing Spurious Correlation)
**논문**: [RE-SORT: Removing Spurious Correlation in Multilevel Interaction for CTR Prediction](https://arxiv.org/abs/2309.14891)

**주요 특징**:
- **Multi-scale Retention (MSR)**: Transformer의 self-attention을 개선한 retention mechanism으로 다양한 레벨의 feature interaction 학습
- **Dual Stream Architecture**: 2개의 병렬 MSR 스트림 (Deep & Shallow)으로 global/local 패턴 동시 포착
- **Feature Selection Module**: Gating mechanism으로 spurious correlation 제거
- **XPOS Positional Encoding**: Enhanced rotary position embedding
- **Interaction Aggregation**: Chunk-based bilinear interaction으로 효율적인 feature fusion

**아키텍처 구성**:
```
Input Features (Embeddings)
    ↓
Feature Selection (Gating)
    ↓
MSR Stream 1 (Deep) ← Retention Mechanism (γ decay)
MSR Stream 2 (Shallow) ← Retention Mechanism (γ decay)
    ↓
Interaction Aggregation (Bilinear)
    ↓
CTR Prediction
```

**하이퍼파라미터**:
- MSR layers: 2
- MSR dimension: 32
- Number of heads: 2
- Embedding dimension: 16
- Dropout: 0.1
- Feature Selection hidden units: [64]

**장점**:
- Spurious correlation 제거로 일반화 성능 향상
- Multi-scale feature interaction으로 복잡한 패턴 학습
- Retention mechanism으로 long-range dependency 효과적 처리

**파라미터 수**: ~34K (경량 모델)

## TODO List - CTR 모델 개선 사항

### 🎯 High Priority

- [ ] **Feature Interaction 구현**
  - [ ] DeepFM 또는 xDeepFM 아키텍처 적용 (FM + DNN의 end-to-end 학습)
  - [ ] DCN v2 (Deep & Cross Network v2) - 명시적 feature crossing
  - [ ] AutoInt - multi-head self-attention 기반 feature interaction

- [ ] **Class Imbalance 처리**
  - [ ] Focal Loss 구현 (gamma=2 추천, CTR 클릭률 1-5%)
  - [ ] Class-balanced Loss (effective number 기반 리샘플링)
  - [ ] Negative sampling with hard negative mining

- [ ] **Model Calibration (보정)**
  - [ ] Temperature Scaling (단일 파라미터로 빠른 보정)
  - [ ] Platt Scaling (로지스틱 회귀 기반)
  - [ ] Isotonic Regression (비모수적 보정)
  - [ ] Expected Calibration Error (ECE) 및 Reliability Diagram으로 평가

### 🔧 Medium Priority

- [x] **Training 최적화** (이미 구현됨)
  - [x] Model Checkpointing 시스템 (성능 기반 저장)
  - [x] Learning Rate Scheduling (CosineAnnealingWarmRestarts 적용)
  - [x] Mixed Precision Training (AMP 사용)
  - [x] Early Stopping with patience (validation loss 기반)
  - [x] Gradient Clipping (norm=1.0 추천)
  - [x] SWA (Stochastic Weight Averaging) - 마지막 epoch들 평균

- [x] **Regularization 강화**
  - [x] L2 Regularization (weight_decay=1e-5 적용됨)
  - [x] Dropout (0.1 적용됨)
  - [x] Label Smoothing (hard label을 soft label로)
  - [x] Embedding Dropout (sparse feature용)
  - ~~DropConnect (weight dropout)~~ - 현재 regularization으로 충분, 과적합 심할 시 실험 예정

- [x] **Embedding 최적화**
  - [x] Adaptive Embedding dimension: 데이터 기반 vocab size별 차원 설정
    - gender (vocab=2): 8차원
    - age_group (vocab=8): 8차원
    - inventory_id (vocab=18): 16차원
    - day_of_week (vocab=7): 8차원
    - hour (vocab=24): 16차원
    - 14.4% 파라미터 감소 (944 → 808)
  - [ ] Hash Embedding for high cardinality features (>10K vocab)
  - [ ] Shared Embeddings for related features (user_id, session_id 등)
  - [ ] Embedding Regularization (L2 on embeddings)

### 🚀 Advanced Features

- [ ] **Context-aware Feature Engineering**
  - [ ] Temporal features (시간대별 CTR 패턴, recency)
  - [ ] User behavior sequence modeling (LSTM/GRU for click history)
  - [ ] Cross-domain features (user profile + item features interaction)

- [ ] **Multi-Task Learning 확장**
  - [ ] ESMM (Entire Space Multi-Task Model) - CVR 예측 추가
  - [ ] MMOE (Multi-gate Mixture-of-Experts) - task-specific experts
  - [ ] PLE (Progressive Layered Extraction) - task separation 개선
  - [ ] Uncertainty-based task weighting (homoscedastic uncertainty)

- [x] **Advanced Architectures**
  - [x] RE-SORT (Removing Spurious Correlation) - Multi-scale Retention with Feature Selection
    - Multi-scale Retention (MSR) mechanism
    - Dual stream architecture (Deep & Shallow)
    - XPOS positional encoding
    - Chunk-based interaction aggregation
    - ~34K parameters (경량 모델)
  - [ ] FiBiNET - Bilinear feature interaction
  - [ ] DLRM (Deep Learning Recommendation Model) - Facebook 아키텍처
  - [ ] BST (Behavior Sequence Transformer) - Transformer for user sequences
  - [ ] Feature Gating Network (FGN) - 동적 feature selection

### 📊 Monitoring & Evaluation

- [x] **평가 지표 확장**
  - [x] Calibration metrics (Brier Score, ECE, MCE, Calibration Gap)
  - [x] Calibration Curve (예측 확률 vs 실제 확률, 10 bins)
  - [ ] GAUC (Group AUC) - user별 AUC 평균
  - [ ] NDCG@K - ranking quality
  - [ ] Business metrics (CTR, Revenue, ROAS, eCPM)

- [ ] **온라인 평가 준비**
  - [ ] A/B Test framework (treatment/control split)
  - [ ] Interleaving 실험 설계
  - [ ] Online model serving latency 측정 (<100ms)

- [ ] **모델 해석성**
  - [ ] Integrated Gradients (attribution 기반 importance)
  - [ ] SHAP values (TreeSHAP 또는 DeepSHAP)
  - [ ] Attention weight visualization
  - [ ] Embedding space visualization (t-SNE, UMAP)

### ⚡ 성능 최적화

- [x] **추론 속도 개선** (일부 구현됨)
  - [x] torch.compile() 적용 (PyTorch 2.0+)
  - [x] Mixed Precision Inference
  - [ ] ONNX 변환 및 최적화
  - [ ] TensorRT 또는 OpenVINO 가속
  - [ ] Embedding 양자화 (INT8)
  - [ ] Knowledge Distillation (큰 모델 → 작은 모델)

- [ ] **분산 학습**
  - [ ] DDP (Distributed Data Parallel)
  - [ ] FSDP (Fully Sharded Data Parallel) for large models
  - [ ] Gradient Accumulation (메모리 부족 시)

### 🔧 엔지니어링 개선

- [x] **실험 관리**
  - [x] 모델별 실험 디렉토리 자동 생성 (transformer_*, resort_*)
  - [x] 성능 기반 모델 저장 시스템
  - [ ] MLflow/Weights&Biases 연동

- [ ] **데이터 파이프라인**
  - [ ] Feature store 연동 (Feast, Tecton)
  - [ ] Online feature serving (Redis, DynamoDB)
  - [ ] Feature versioning 시스템

- [ ] **모델 서빙**
  - [ ] FastAPI 또는 TorchServe 기반 API
  - [ ] Model versioning (A/B test 지원)
  - [ ] Batch prediction pipeline
  - [ ] Cold start 문제 해결 (default model)


### 이렇게 했는데 lightbgm보다 낮음
Tabular 데이터에 최적화 - CTR 데이터처럼 categorical + numerical feature가 섞인 테이블 데이터에 매우 강력함
Feature interaction 자동 학습 - Tree 기반이라 feature 간 복잡한 상호작용을 자동으로 잡아냄
적은 데이터 전처리 - Label encoding만으로 충분
Overfitting 방지 - Built-in regularization이 잘 되어있음
