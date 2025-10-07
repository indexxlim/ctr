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
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.embedding(x)


class MultiTaskTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # embedding tables for sparse features (Adaptive Embedding)
        self.embeddings = nn.ModuleDict()
        for feat, vocab_size in config.sparse_feats.items():
            embed_dim = config.embed_dims.get(feat, config.embed_dim)  # Use adaptive dim if available
            self.embeddings[feat] = EmbeddingTable(vocab_size, embed_dim)

        # dense encoder projection
        self.dense_proj = nn.Linear(len(config.dense_feats), config.d_model)

        # shared MLP
        # Calculate total embedding dimension (sum of all adaptive dimensions)
        total_embed_dim = sum(self.embeddings[feat].embed_dim for feat in config.sparse_feats.keys()) + config.d_model
        self.shared_mlp = nn.Sequential(
            nn.Linear(total_embed_dim, config.hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # task heads
        self.ctr_head = nn.Linear(config.hidden_dims[-1], 1)

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
        
        # task predictions (only CTR) - return logits
        logits = self.ctr_head(shared)

        return logits

# Import from centralized config
from config import ModelConfig

# Loss function for CTR prediction
def ctr_loss(logits, labels, label_smoothing=0.0):
    """CTR loss with optional label smoothing"""
    if label_smoothing > 0:
        # Label smoothing: convert hard labels to soft labels
        # 0 -> label_smoothing, 1 -> 1 - label_smoothing
        labels = labels.float()
        labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(logits.squeeze(), labels.float())

# Focal Loss for class imbalance
def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    """
    Focal Loss for addressing class imbalance in CTR prediction

    Args:
        logits: model predictions (before sigmoid)
        alpha: balancing factor (default: 0.25 for minority class)
        gamma: focusing parameter (default: 2.0)
    """
    bce_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels.float(), reduction='none')
    probs = torch.sigmoid(logits.squeeze())

    # Calculate focal weight
    labels_float = labels.float()
    pt = torch.where(labels_float == 1, probs, 1 - probs)
    focal_weight = (1 - pt) ** gamma

    # Apply alpha balancing
    alpha_t = torch.where(labels_float == 1, alpha, 1 - alpha)

    loss = alpha_t * focal_weight * bce_loss
    return loss.mean()


