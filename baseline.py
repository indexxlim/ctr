import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# 1. 데이터 로드 (샘플링)
data = pd.read_parquet("/home/klcube/lim/train/ctr/data/train.parquet")
data = data.sample(n=100000, random_state=42)  # 10만 샘플로 테스트

# 2. 피처/라벨 분리
X = data.drop(columns=["clicked"])
y = data["clicked"]

# 3. 범주형 피처 인코딩
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# 4. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 범주형 & 수치형 feature 분류
categorical_cols = [col for col in X_train.columns if "feat_" in col or "l_feat_" in col or col in ["gender", "age_group", "inventory_id", "day_of_week", "hour"]]
numeric_cols = [col for col in X_train.columns if "history_" in col]

# 4. 전처리
# (a) 범주형: Label Encoding
# (b) 수치형: Scaling (Optional)

# 5. 모델 (예: LightGBM baseline)
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=100, #2000
    learning_rate=0.1, #0.05
    max_depth=6,   #-1
    subsample=0.8,
    colsample_bytree=0.8,
    metric="auc",
    verbose=-1 #
)

model.fit(X_train, y_train, categorical_feature=categorical_cols)

# 6. 예측 및 평가
preds = model.predict_proba(X_test)[:,1]
auc_score = roc_auc_score(y_test, preds)

print(f"Test AUC: {auc_score:.4f}")
