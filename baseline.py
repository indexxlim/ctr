# 1. 데이터 로드
train = pd.read_parquet("train.parquet")
test = pd.read_parquet("test.parquet")
submission = pd.read_csv("sample_submission.csv")

# 2. 피처/라벨 분리
X_train = train.drop(columns=["clicked"])
y_train = train["clicked"]
X_test  = test.drop(columns=["ID"])

# 3. 범주형 & 수치형 feature 분류
categorical_cols = [col for col in X_train.columns if "feat_" in col or "l_feat_" in col or col in ["gender", "age_group", "inventory_id", "day_of_week", "hour"]]
numeric_cols = [col for col in X_train.columns if "history_" in col]

# 4. 전처리
# (a) 범주형: Label Encoding
# (b) 수치형: Scaling (Optional)

# 5. 모델 (예: LightGBM baseline)
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    metric="auc"
)

model.fit(X_train, y_train, categorical_feature=categorical_cols)

# 6. 예측
preds = model.predict_proba(X_test)[:,1]

# 7. 제출 파일 생성
submission["clicked"] = preds
submission.to_csv("submission.csv", index=False)
