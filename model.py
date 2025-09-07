import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.nn.parameter import Parameter

class EmbeddingTable(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, x):
        return self.embedding(x)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, output_dim=1, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mlp(x)

class MultiTaskTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # embedding tables for sparse features
        self.embeddings = nn.ModuleDict()
        for feat, vocab_size in config.sparse_feats.items():
            self.embeddings[feat] = EmbeddingTable(vocab_size, config.embed_dim)
        
        # dense encoder projection
        self.dense_proj = nn.Linear(len(config.dense_feats), config.d_model)
        
        # shared MLP
        total_embed_dim = len(config.sparse_feats) * config.embed_dim + config.d_model
        self.shared_mlp = MLP(input_dim=total_embed_dim, 
                             hidden_dims=config.hidden_dims)
        
        # task heads
        self.ctr_head = MLP(input_dim=config.hidden_dims[-1], output_dim=1)
        self.cvr_head = MLP(input_dim=config.hidden_dims[-1], output_dim=1)
        self.ctcvr_head = MLP(input_dim=config.hidden_dims[-1], output_dim=1)
        
        # optional uncertainty head
        self.uncertainty_head = MLP(input_dim=config.hidden_dims[-1], output_dim=1)

    def forward(self, batch):
        # embed sparse features
        emb_list = []
        for feat_name in self.config.sparse_feats.keys():
            if feat_name in batch:
                emb = self.embeddings[feat_name](batch[feat_name])
                emb_list.append(emb)
        
        # process dense features
        if 'dense_features' in batch:
            dense = self.dense_proj(batch['dense_features'])
        else:
            dense = torch.zeros(batch[list(batch.keys())[0]].size(0), self.config.d_model)
            
        # concatenate all features
        x = torch.cat(emb_list + [dense], dim=-1)  # shape [B, D]
        
        # shared representation
        shared = self.shared_mlp(x)
        
        # task predictions
        p_ctr = torch.sigmoid(self.ctr_head(shared))
        p_cvr = torch.sigmoid(self.cvr_head(shared))
        p_ctcvr = torch.sigmoid(self.ctcvr_head(shared))
        
        return {"p_ctr": p_ctr, "p_cvr": p_cvr, "p_ctcvr": p_ctcvr}

class ModelConfig:
    def __init__(self):
        self.sparse_feats = {
            'user_id_encoded': 5000,
            'item_id_encoded': 1000,
            'category_encoded': 5,
            'device_encoded': 3,
            'time_of_day_encoded': 4
        }
        self.dense_feats = ['user_age', 'price', 'item_rating', 'user_historical_ctr', 
                           'position', 'page_views', 'session_duration']
        self.embed_dim = 32
        self.d_model = 128
        self.hidden_dims = [256, 128, 64]

# Loss, training
def multitask_loss(preds, labels, weights=None):
    if weights is None:
        weights = {'ctr': 1.0, 'cvr': 0.5, 'ctcvr': 0.3}
    
    # CTR loss
    loss_ctr = F.binary_cross_entropy(preds['p_ctr'].squeeze(), labels['ctr'].float())
    
    # CVR loss - only on clicked samples
    clicked_mask = labels['clicked'] == 1
    if clicked_mask.sum() > 0:
        loss_cvr = F.binary_cross_entropy(
            preds['p_cvr'][clicked_mask].squeeze(), 
            labels['converted'][clicked_mask].float()
        )
    else:
        loss_cvr = torch.tensor(0.0)
    
    # CTCVR loss
    loss_ctcvr = F.binary_cross_entropy(preds['p_ctcvr'].squeeze(), labels['converted'].float())
    
    return weights['ctr'] * loss_ctr + weights['cvr'] * loss_cvr + weights['ctcvr'] * loss_ctcvr

# Training loop
for epoch in range(E):
    for batch in dataloader:
        preds = model(batch)
        loss = multitask_loss(preds, batch.labels, config.loss_weights)
        loss.backward()
        optimizer.step()
        scheduler.step()

# Calibration: Temperature Scaling (post-hoc on validation)
# Find temperature T minimizing NLL on val set:
def find_temperature(model, val_loader):
    T = Parameter(torch.ones(1))
    optimizer = Adam([T], lr=0.01)
    for i in range(1000):
        nll = 0
        for batch in val_loader:
            logits = logit(model(batch)['p_ctr'])  # inverse sigmoid
            scaled = logits / T
            nll += binary_cross_entropy_with_logits(scaled, batch.labels['ctr'])
        nll.backward(); optimizer.step(); optimizer.zero_grad()
    return float(T.detach())

# Save model + temp to ModelRegistry

