import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import json
import sys
import os
from sklearn.preprocessing import LabelEncoder

# Add experiment directory to path to import model
EXP_DIR = '/home/lim/project/ctr/experiments/exp_20250923_094110'
sys.path.insert(0, EXP_DIR)
from model import MultiTaskTransformer, ModelConfig

class TestDataset(Dataset):
    def __init__(self, df, config):
        self.df = df
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Sparse features
        sparse_features = {}
        for feat_name in self.config.sparse_feats.keys():
            if feat_name in self.df.columns:
                sparse_features[feat_name] = torch.tensor(row[feat_name], dtype=torch.long)
            else:
                sparse_features[feat_name] = torch.tensor(0, dtype=torch.long)

        # Dense features
        dense_values = []
        for feat in self.config.dense_feats:
            if feat in self.df.columns:
                dense_values.append(row[feat])
            else:
                dense_values.append(0.0)
        dense_features = torch.tensor(dense_values, dtype=torch.float32)

        return {
            **sparse_features,
            'dense_features': dense_features,
            'index': idx
        }

def generate_test_predictions(model_path, test_data_path, output_path, exp_dir):
    """Generate predictions for test data using trained model"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load preprocessors
    preprocessor_path = '/home/lim/project/ctr/data/processed/preprocessors.pkl'
    if os.path.exists(preprocessor_path):
        print(f"Loading preprocessors from: {preprocessor_path}")
        with open(preprocessor_path, 'rb') as f:
            preprocessors = pickle.load(f)
        label_encoders = preprocessors['label_encoders']
        scaler = preprocessors['scaler']
    else:
        print("Warning: No preprocessors found, predictions may be incorrect")
        label_encoders = {}
        scaler = None

    # Load test data
    print(f"Loading test data from: {test_data_path}")
    test_df = pd.read_parquet(test_data_path)

    # Store IDs if they exist
    if 'ID' in test_df.columns:
        test_ids = test_df['ID'].values
    else:
        test_ids = [f"TEST_{i:07d}" for i in range(len(test_df))]

    print(f"Test data shape: {test_df.shape}")

    # Preprocess test data (same as training)
    test_df = test_df.fillna(0)

    # Get categorical columns (object dtype)
    categorical_cols = [col for col in test_df.columns if test_df[col].dtype == 'object']
    print(f"Categorical columns found: {categorical_cols}")

    # Encode categorical features if encoders exist, otherwise do label encoding

    for col in categorical_cols:
        if col in label_encoders:
            # Use existing encoder with handling of unseen labels (optimized with dict mapping)
            le = label_encoders[col]
            mapping = {label: idx for idx, label in enumerate(le.classes_)}
            test_df[col] = test_df[col].astype(str).map(mapping).fillna(0).astype(int)
        else:
            # Create new encoder for this column
            le = LabelEncoder()
            test_df[col] = le.fit_transform(test_df[col].astype(str))

    # Normalize dense features
    if scaler is not None:
        config = ModelConfig()
        existing_dense_cols = [col for col in config.dense_feats if col in test_df.columns]
        if existing_dense_cols:
            test_df[existing_dense_cols] = scaler.transform(test_df[existing_dense_cols])

    # Load model
    print(f"Loading model from: {model_path}")
    config = ModelConfig()
    model = MultiTaskTransformer(config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create dataset and dataloader
    test_dataset = TestDataset(test_df, config)
    test_loader = DataLoader(test_dataset, batch_size=16384, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Generate predictions
    print("Generating predictions...")
    predictions = []

    from tqdm import tqdm

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Predicting")):
            batch_input = {k: v.to(device) for k, v in batch.items() if k != 'index'}

            logits = model(batch_input)
            preds = torch.sigmoid(logits)

            predictions.extend(preds.squeeze().cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f"\nProcessed {min((i+1) * test_loader.batch_size, len(test_dataset))} / {len(test_dataset)} samples...")

    # Create submission file
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'clicked': predictions
    })

    # Save to output path
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    print(f"Shape: {submission_df.shape}")
    print(f"Sample predictions:")
    print(submission_df.head(10))
    print(f"\nPrediction statistics:")
    print(f"Mean: {submission_df['clicked'].mean():.6f}")
    print(f"Std: {submission_df['clicked'].std():.6f}")
    print(f"Min: {submission_df['clicked'].min():.6f}")
    print(f"Max: {submission_df['clicked'].max():.6f}")

    return submission_df

if __name__ == "__main__":
    # Best model configuration
    best_exp = "exp_20250923_094110"
    model_path = f"/home/lim/project/ctr/experiments/{best_exp}/models/best_model.pth"
    exp_dir = f"/home/lim/project/ctr/experiments/{best_exp}"
    test_data_path = "/home/lim/project/data/test.parquet"
    output_path = "/home/lim/project/data/sample_submission.csv"

    print(f"Generating submission using best model: {best_exp}")
    print(f"Model AUC: 0.7382, LogLoss: 0.0866")
    print("="*60)

    submission_df = generate_test_predictions(model_path, test_data_path, output_path, exp_dir)
    print("\n✅ Done!")
