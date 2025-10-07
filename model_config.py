"""
Configuration file for MultiTaskTransformer model
"""

class BaseConfig:
    """Base configuration with common features"""

    # Feature definitions
    SPARSE_FEATURES = {
        'gender': 5,
        'age_group': 15,
        'inventory_id': 50,
        'day_of_week': 10,
        'hour': 30
    }

    DENSE_FEATURES = (
        ['seq'] +
        [f'l_feat_{i}' for i in range(1, 28)] +
        [f'feat_e_{i}' for i in range(1, 11)] +
        [f'feat_d_{i}' for i in range(1, 7)] +
        [f'feat_c_{i}' for i in range(1, 9)] +
        [f'feat_b_{i}' for i in range(1, 7)] +
        [f'feat_a_{i}' for i in range(1, 19)] +
        [f'history_a_{i}' for i in range(1, 8)] +
        [f'history_b_{i}' for i in range(1, 31)]
    )

    # Adaptive Embedding Dimensions
    ADAPTIVE_EMBED_DIMS = {
        'gender': 8,
        'age_group': 8,
        'inventory_id': 16,
        'day_of_week': 8,
        'hour': 16
    }


class ModelConfig(BaseConfig):
    """Configuration for MultiTaskTransformer model"""

    def __init__(self, use_adaptive_embedding=True):
        # Features
        self.sparse_feats = self.SPARSE_FEATURES
        self.dense_feats = self.DENSE_FEATURES

        # Embedding
        self.use_adaptive_embedding = use_adaptive_embedding
        self.embed_dims = self.ADAPTIVE_EMBED_DIMS if use_adaptive_embedding else {}
        self.embed_dim = 16  # default fallback

        # Model architecture
        self.d_model = 64
        self.hidden_dims = [128, 64]
