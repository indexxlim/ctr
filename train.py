import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
import pickle

from model import MultiTaskTransformer, ModelConfig, multitask_loss

class CTRDataset(Dataset):
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
            sparse_features[feat_name] = torch.tensor(row[feat_name], dtype=torch.long)
        
        # Dense features
        dense_features = torch.tensor([
            row[feat] for feat in self.config.dense_feats
        ], dtype=torch.float32)
        
        # Labels
        labels = {
            'ctr': torch.tensor(row['clicked'], dtype=torch.long),
            'clicked': torch.tensor(row['clicked'], dtype=torch.long),
            'converted': torch.tensor(row['converted'], dtype=torch.long)
        }
        
        return {
            **sparse_features,
            'dense_features': dense_features,
            'labels': labels
        }

def train_model():
    # Load data
    df = pd.read_csv('ctr_data.csv')
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create config and model
    config = ModelConfig()
    model = MultiTaskTransformer(config)
    
    # Create datasets and dataloaders
    train_dataset = CTRDataset(train_df, config)
    val_dataset = CTRDataset(val_df, config)
    test_dataset = CTRDataset(test_df, config)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    epochs = 20
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Prepare batch
            batch_input = {k: v for k, v in batch.items() if k != 'labels'}
            labels = batch['labels']
            
            # Forward pass
            preds = model(batch_input)
            loss = multitask_loss(preds, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds_ctr = []
        val_labels_ctr = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch_input = {k: v for k, v in batch.items() if k != 'labels'}
                labels = batch['labels']
                
                preds = model(batch_input)
                loss = multitask_loss(preds, labels)
                val_loss += loss.item()
                
                val_preds_ctr.extend(preds['p_ctr'].squeeze().cpu().numpy())
                val_labels_ctr.extend(labels['ctr'].cpu().numpy())
        
        # Calculate metrics
        val_auc = roc_auc_score(val_labels_ctr, val_preds_ctr)
        val_logloss = log_loss(val_labels_ctr, val_preds_ctr)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'  Val AUC: {val_auc:.4f}')
        print(f'  Val LogLoss: {val_logloss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_ctr_model.pth')
            print(f'  New best model saved!')
        
        scheduler.step()
        print()
    
    # Test evaluation
    model.load_state_dict(torch.load('best_ctr_model.pth'))
    model.eval()
    
    test_preds_ctr = []
    test_labels_ctr = []
    test_preds_cvr = []
    test_labels_cvr = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch_input = {k: v for k, v in batch.items() if k != 'labels'}
            labels = batch['labels']
            
            preds = model(batch_input)
            
            test_preds_ctr.extend(preds['p_ctr'].squeeze().cpu().numpy())
            test_labels_ctr.extend(labels['ctr'].cpu().numpy())
            
            # CVR only for clicked samples
            clicked_mask = labels['clicked'] == 1
            if clicked_mask.sum() > 0:
                test_preds_cvr.extend(preds['p_cvr'][clicked_mask].squeeze().cpu().numpy())
                test_labels_cvr.extend(labels['converted'][clicked_mask].cpu().numpy())
    
    # Test metrics
    test_ctr_auc = roc_auc_score(test_labels_ctr, test_preds_ctr)
    test_ctr_logloss = log_loss(test_labels_ctr, test_preds_ctr)
    
    print("="*50)
    print("TEST RESULTS:")
    print(f"CTR AUC: {test_ctr_auc:.4f}")
    print(f"CTR LogLoss: {test_ctr_logloss:.4f}")
    
    if len(test_labels_cvr) > 0:
        test_cvr_auc = roc_auc_score(test_labels_cvr, test_preds_cvr)
        print(f"CVR AUC: {test_cvr_auc:.4f}")
    
    return model

if __name__ == "__main__":
    model = train_model()