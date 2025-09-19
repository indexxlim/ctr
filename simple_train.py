import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder

class SimpleCTRModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

class CTRDataset(Dataset):
    def __init__(self, df, feature_cols):
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(df['clicked'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_model():
    # Load and preprocess data
    print("Loading data...")
    df = pd.read_parquet('/home/klcube/lim/train/ctr/data/train.parquet')
    df = df.sample(n=50000, random_state=42)

    # Handle NaN values
    df = df.fillna(0)

    # Encode categorical features
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    for col in categorical_cols:
        if col != 'clicked':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Prepare features
    feature_cols = [col for col in df.columns if col != 'clicked']

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Create datasets
    train_dataset = CTRDataset(train_df, feature_cols)
    val_dataset = CTRDataset(val_df, feature_cols)
    test_dataset = CTRDataset(test_df, feature_cols)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Create model
    model = SimpleCTRModel(len(feature_cols))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    epochs = 10
    best_val_auc = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            preds = model(features).squeeze()
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0.0

        with torch.no_grad():
            for features, labels in val_loader:
                logits = model(features).squeeze()
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = torch.sigmoid(logits)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_auc = roc_auc_score(val_labels, val_preds)
        val_logloss = log_loss(val_labels, val_preds)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'  Val AUC: {val_auc:.4f}')
        print(f'  Val LogLoss: {val_logloss:.4f}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_simple_ctr_model.pth')
            print(f'  New best model saved!')
        print()

    # Test evaluation
    model.load_state_dict(torch.load('best_simple_ctr_model.pth'))
    model.eval()

    test_preds = []
    test_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            logits = model(features).squeeze()
            preds = torch.sigmoid(logits)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_auc = roc_auc_score(test_labels, test_preds)
    test_logloss = log_loss(test_labels, test_preds)

    print("="*50)
    print("TEST RESULTS:")
    print(f"CTR AUC: {test_auc:.4f}")
    print(f"CTR LogLoss: {test_logloss:.4f}")

    return model

if __name__ == "__main__":
    model = train_model()