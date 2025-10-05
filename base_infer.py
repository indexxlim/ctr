import pickle
import pandas as pd
import lightgbm as lgb

# Load model
model_path = '/home/lim/project/ctr/experiments/baseline_20251005_015524/lightgbm_model.pkl'
print(f'Loading model from: {model_path}')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load test data
test_path = '/home/lim/project/data/test.parquet'
print(f'Loading test data from: {test_path}')
test_df = pd.read_parquet(test_path)
print(f'Test data shape: {test_df.shape}')

# Store IDs
test_ids = test_df['ID'].values if 'ID' in test_df.columns else [f'TEST_{i:07d}' for i in range(len(test_df))]

# Preprocess
test_df = test_df.fillna(0)
categorical_cols = [col for col in test_df.columns if test_df[col].dtype == 'object' and col != 'ID']
print(f'Categorical columns: {categorical_cols}')

# Load preprocessors
preprocessor_path = '/home/lim/project/ctr/data/processed/preprocessors.pkl'
with open(preprocessor_path, 'rb') as f:
    preprocessors = pickle.load(f)
label_encoders = preprocessors['label_encoders']

# Encode categoricals
for col in categorical_cols:
    if col in label_encoders:
        le = label_encoders[col]
        mapping = {label: idx for idx, label in enumerate(le.classes_)}
        test_df[col] = test_df[col].astype(str).map(mapping).fillna(0).astype(int)

# Drop ID for prediction
X_test = test_df.drop('ID', axis=1, errors='ignore')

# Predict
print('Generating predictions...')
predictions = model.predict(X_test)

# Create submission
submission_df = pd.DataFrame({
    'ID': test_ids,
    'clicked': predictions
})

output_path = '/home/lim/project/data/submission_baseline_20251005_015524.csv'
submission_df.to_csv(output_path, index=False)

print(f'Submission saved to: {output_path}')
print(f'Shape: {submission_df.shape}')
print(f'Sample predictions:')
print(submission_df.head(10))
print(f'Prediction statistics:')
print(f'Mean: {submission_df[\"clicked\"].mean():.6f}')
print(f'Min: {submission_df[\"clicked\"].min():.6f}')
print(f'Max: {submission_df[\"clicked\"].max():.6f}')
