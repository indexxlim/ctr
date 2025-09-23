import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import time
import os

print("🚀 CTR Baseline Model Training")
print("="*50)

# 1. 전처리된 데이터 로드
start_time = time.time()
print("Loading preprocessed data...")

if os.path.exists("./data/processed/train_data.parquet"):
    train_data = pd.read_parquet("./data/processed/train_data.parquet")
    val_data = pd.read_parquet("./data/processed/val_data.parquet")
    test_data = pd.read_parquet("./data/processed/test_data.parquet")
    print(f"Train: {len(train_data):,}, Val: {len(val_data):,}, Test: {len(test_data):,}")
else:
    print("Preprocessed data not found. Please run: python train.py --prepare-data")
    exit(1)

# 2. 피처/라벨 분리
X_train = train_data.drop(columns=["clicked"])
y_train = train_data["clicked"]
X_val = val_data.drop(columns=["clicked"])
y_val = val_data["clicked"]
X_test = test_data.drop(columns=["clicked"])
y_test = test_data["clicked"]

print(f"Features: {len(X_train.columns)}, Train positive rate: {y_train.mean():.4f}")
load_time = time.time() - start_time
print(f"Data loaded in {load_time:.2f}s")

# 3. 범주형 feature 식별 (전처리된 데이터는 이미 인코딩됨)
categorical_cols = [col for col in X_train.columns if "feat_" in col or "l_feat_" in col or col in ["gender", "age_group", "inventory_id", "day_of_week", "hour"]]
print(f"Categorical features: {len(categorical_cols)}")

# 4. LightGBM 모델 (최적화된 설정)
from lightgbm import LGBMClassifier

print("\n🎯 Training LightGBM baseline...")
train_start = time.time()

model = LGBMClassifier(
    n_estimators=500,     # 더 많은 트리
    learning_rate=0.05,   # 더 낮은 학습률
    max_depth=8,          # 더 깊은 트리
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,        # L1 정규화
    reg_lambda=0.1,       # L2 정규화
    metric="auc",
    objective="binary",
    random_state=42,
    verbose=-1,
    n_jobs=-1             # 멀티프로세싱
)

# 5. 학습 (validation set으로 early stopping)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    categorical_feature=categorical_cols,
    eval_metric='auc',
    callbacks=[
        # early stopping
        # verbose evaluation
    ]
)

train_time = time.time() - train_start
print(f"Training completed in {train_time:.2f}s")

# 6. 예측 및 평가
print("\n📊 Evaluation Results:")

# Validation 평가
val_preds = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_preds)
val_logloss = log_loss(y_val, val_preds)

# Test 평가
test_preds = model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_preds)
test_logloss = log_loss(y_test, test_preds)

print(f"Validation AUC: {val_auc:.4f}")
print(f"Validation LogLoss: {val_logloss:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test LogLoss: {test_logloss:.4f}")

# Feature importance 출력
print(f"\n🔍 Top 10 Important Features:")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# 7. 모델 및 결과 저장
import pickle
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"./experiments/baseline_{timestamp}"
os.makedirs(save_dir, exist_ok=True)

print(f"\n💾 Saving baseline results to: {save_dir}")

# 모델 저장
with open(f"{save_dir}/lightgbm_model.pkl", 'wb') as f:
    pickle.dump(model, f)

# Feature importance 저장
feature_importance.to_csv(f"{save_dir}/feature_importance.csv", index=False)

# 결과 저장
results = {
    'timestamp': timestamp,
    'model_type': 'LightGBM',
    'validation_auc': val_auc,
    'validation_logloss': val_logloss,
    'test_auc': test_auc,
    'test_logloss': test_logloss,
    'training_time_seconds': train_time,
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_features': len(X_train.columns),
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test)
}

import json
with open(f"{save_dir}/results.json", 'w') as f:
    json.dump(results, f, indent=2)

# 예측값 저장 (추후 앙상블이나 분석용)
test_predictions = pd.DataFrame({
    'test_predictions': test_preds,
    'test_labels': y_test
})
test_predictions.to_csv(f"{save_dir}/test_predictions.csv", index=False)

print(f"✅ Baseline model and results saved successfully!")
print(f"📁 Files saved:")
print(f"   - lightgbm_model.pkl (model)")
print(f"   - feature_importance.csv")
print(f"   - results.json")
print(f"   - test_predictions.csv")
