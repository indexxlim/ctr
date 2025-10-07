import torch
from torch import nn
import math
from xpos_relative_position import XPOS


class RESORT(nn.Module):
    """RE-SORT model adapted for CTR prediction"""
    def __init__(self, config):
        super(RESORT, self).__init__()
        self.config = config

        # Embedding layers
        self.embeddings = nn.ModuleDict()
        for feat, vocab_size in config.sparse_feats.items():
            embed_dim = config.embed_dims.get(feat, config.embed_dim)
            self.embeddings[feat] = nn.Embedding(vocab_size, embed_dim)

        # Dense feature projection
        self.dense_proj = nn.Linear(len(config.dense_feats), config.embed_dim)

        # Calculate total feature dimension
        num_fields = len(config.sparse_feats) + 1  # +1 for dense features
        feature_dim = config.embed_dim * num_fields

        # MSR parameters
        MSR_layers = config.msr_layers
        MSR_dim = config.msr_dim
        num_heads = config.num_heads

        # Gamma decay for each layer
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), MSR_layers))).detach().cpu().tolist()

        # Two parallel MSR streams (Deep and Shallow)
        # Input: [batch_size, num_fields, embed_dim] -> need to transform per field
        self.self_MSR1 = nn.Sequential(
            *[MultiHeadSelfMSR(config.embed_dim if i == 0 else MSR_dim,
                              MSR_dim=MSR_dim,
                              num_heads=num_heads,
                              dropout_rate=config.dropout,
                              use_residual=True,
                              use_scale=config.use_scale,
                              group_norm=config.group_norm,
                              gamma=self.gammas[i])
              for i in range(MSR_layers)])

        self.self_MSR2 = nn.Sequential(
            *[MultiHeadSelfMSR(config.embed_dim if i == 0 else MSR_dim,
                              MSR_dim=MSR_dim,
                              num_heads=num_heads,
                              dropout_rate=config.dropout,
                              use_residual=True,
                              use_scale=config.use_scale,
                              group_norm=config.group_norm,
                              gamma=self.gammas[i])
              for i in range(MSR_layers)])

        # Feature Selection module
        if config.use_fs:
            self.fs_module = FeatureSelection(num_fields, config.embed_dim, config.fs_hidden_units)
        else:
            self.fs_module = None

        # Interaction Aggregation module
        self.fusion_module = InteractionAggregation(num_fields * MSR_dim, num_fields * MSR_dim, output_dim=1, num_heads=num_heads)

    def forward(self, batch):
        # Embed sparse features
        emb_list = []
        for feat_name in self.config.sparse_feats.keys():
            if feat_name in batch:
                emb = self.embeddings[feat_name](batch[feat_name])
                emb_list.append(emb)

        # Process dense features
        if 'dense_features' in batch:
            dense = self.dense_proj(batch['dense_features'])
            emb_list.append(dense)

        # Stack embeddings: [batch_size, num_fields, embed_dim]
        flat_emb = torch.stack(emb_list, dim=1)

        # Feature selection
        if self.fs_module is not None:
            feat1, feat2 = self.fs_module(flat_emb)
        else:
            feat1, feat2 = flat_emb, flat_emb

        # MSR processing
        msr1_out = self.self_MSR1(feat1)  # [batch_size, num_fields, msr_dim]
        msr2_out = self.self_MSR2(feat2)  # [batch_size, num_fields, msr_dim]

        # Flatten for fusion
        msr1_flat = msr1_out.flatten(start_dim=1)  # [batch_size, num_fields * msr_dim]
        msr2_flat = msr2_out.flatten(start_dim=1)  # [batch_size, num_fields * msr_dim]

        # Fusion and prediction
        y_pred = self.fusion_module(msr1_flat, msr2_flat)

        return y_pred


class FeatureSelection(nn.Module):
    """Feature Selection module with gating mechanism"""
    def __init__(self, num_fields, embedding_dim, fs_hidden_units=[64]):
        super(FeatureSelection, self).__init__()

        self.num_fields = num_fields
        self.embedding_dim = embedding_dim

        # Context bias (learnable)
        self.fs1_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        self.fs2_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))

        # Gate networks - output per-field gates
        self.fs1_gate = nn.Sequential(
            nn.Linear(embedding_dim, fs_hidden_units[0]),
            nn.ReLU(),
            nn.Linear(fs_hidden_units[0], num_fields),
            nn.Sigmoid()
        )

        self.fs2_gate = nn.Sequential(
            nn.Linear(embedding_dim, fs_hidden_units[0]),
            nn.ReLU(),
            nn.Linear(fs_hidden_units[0], num_fields),
            nn.Sigmoid()
        )

    def forward(self, flat_emb):
        # flat_emb: [batch_size, num_fields, embedding_dim]
        batch_size = flat_emb.size(0)

        # Gate 1
        fs1_input = self.fs1_ctx_bias.repeat(batch_size, 1)
        gt1 = self.fs1_gate(fs1_input) * 2  # [batch_size, num_fields]
        gt1 = gt1.unsqueeze(-1)  # [batch_size, num_fields, 1]
        feature1 = flat_emb * gt1

        # Gate 2
        fs2_input = self.fs2_ctx_bias.repeat(batch_size, 1)
        gt2 = self.fs2_gate(fs2_input) * 2  # [batch_size, num_fields]
        gt2 = gt2.unsqueeze(-1)  # [batch_size, num_fields, 1]
        feature2 = flat_emb * gt2

        return feature1, feature2


class InteractionAggregation(nn.Module):
    """Chunk-based interaction aggregation module"""
    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super(InteractionAggregation, self).__init__()
        assert x_dim % num_heads == 0 and y_dim % num_heads == 0, \
            "Input dim must be divisible by num_heads!"
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_x_dim = x_dim // num_heads
        self.head_y_dim = y_dim // num_heads
        self.w_x = nn.Linear(x_dim, output_dim)
        self.w_y = nn.Linear(y_dim, output_dim)
        self.w_xy = nn.Parameter(torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim,
                                              output_dim))
        nn.init.xavier_normal_(self.w_xy)

    def forward(self, x, y):
        output = self.w_x(x) + self.w_y(y)
        head_x = x.view(-1, self.num_heads, self.head_x_dim)
        head_y = y.view(-1, self.num_heads, self.head_y_dim)
        xy = torch.matmul(torch.matmul(head_x.unsqueeze(2),
                                       self.w_xy.view(self.num_heads, self.head_x_dim, -1)) \
                               .view(-1, self.num_heads, self.output_dim, self.head_y_dim),
                          head_y.unsqueeze(-1)).squeeze(-1)
        output += xy.sum(dim=1)
        return output


class RetDotProductMSR(nn.Module):
    """Retention-based dot product attention with decay"""
    def __init__(self, dropout_rate=0.):
        super(RetDotProductMSR, self).__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, Q, K, V, D, scale=None, mask=None):
        device = Q.device
        scores = torch.matmul(Q, K.transpose(-1, -2))
        if scale:
            scores = scores / scale
        if mask is not None:
            mask = mask.view_as(scores)
            scores = scores.masked_fill_(mask.float() == 0, -1.e9)

        # Apply decay matrix
        MSR = scores * D.unsqueeze(0).to(device)
        if self.dropout is not None:
            MSR = self.dropout(MSR)
        output = torch.matmul(MSR, V)
        return output, MSR


class MultiHeadSelfMSR(nn.Module):
    """Multi-head Multi-scale Retention (MSR) module"""
    def __init__(self, input_dim, MSR_dim=None, num_heads=1, dropout_rate=0.,
                 use_residual=True, use_scale=False, group_norm=True, gamma=0.9):
        super(MultiHeadSelfMSR, self).__init__()
        if MSR_dim is None:
            MSR_dim = input_dim
        assert MSR_dim % num_heads == 0, \
            "MSR_dim={} is not divisible by num_heads={}".format(MSR_dim, num_heads)
        self.head_dim = MSR_dim // num_heads
        self.MSR_dim = MSR_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, MSR_dim, bias=False)
        self.W_k = nn.Linear(input_dim, MSR_dim, bias=False)
        self.W_v = nn.Linear(input_dim, MSR_dim, bias=False)
        self.xpos = XPOS(MSR_dim)
        self.gamma = gamma
        self.swish = lambda x: x * torch.sigmoid(x)

        if self.use_residual and input_dim != MSR_dim:
            self.W_res = nn.Linear(input_dim, MSR_dim, bias=False)
        else:
            self.W_res = None
        self.dot_MSR = RetDotProductMSR(dropout_rate)
        self.group_norm_flag = group_norm
        self.group_norm = nn.GroupNorm(num_heads, MSR_dim) if group_norm else None

    def _get_D(self, sequence_length):
        """Compute decay matrix D"""
        device = next(self.parameters()).device
        n = torch.arange(sequence_length, device=device).unsqueeze(1)
        m = torch.arange(sequence_length, device=device).unsqueeze(0)
        D = (self.gamma ** (n - m)) * (n >= m).float()
        D[D != D] = 0  # Replace NaN with 0
        return D

    def forward(self, X):
        residual = X

        # Linear projection
        query = self.W_q(X)
        key = self.W_k(X)
        query = self.xpos(query)
        key = self.xpos(key)

        if self.group_norm_flag and self.group_norm is not None:
            original_shape = key.shape
            key = self.group_norm(key.flatten(start_dim=0, end_dim=-2).unsqueeze(0).transpose(1, 2)).transpose(1, 2).squeeze(0).view(original_shape)

        value = self.W_v(X)

        sequence_length = X.shape[1]
        D = self._get_D(sequence_length)

        # Split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot product MSR
        output, MSR = self.dot_MSR(self.swish(query), key, value, D, scale=self.scale)

        # Concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        if self.W_res is not None:
            residual = self.W_res(residual)
        if self.use_residual:
            output += residual
        output = output.relu()
        return output


# Import from centralized config
from config import ResortConfig
