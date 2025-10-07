"""
Configuration file for CTR prediction models
"""

class BaseConfig:
    """Base configuration with common features and settings"""

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

    # Adaptive Embedding Dimensions (data-driven)
    ADAPTIVE_EMBED_DIMS = {
        'gender': 8,           # vocab=2, very small
        'age_group': 8,        # vocab=8, small
        'inventory_id': 16,    # vocab=18, medium
        'day_of_week': 8,      # vocab=7, small
        'hour': 16             # vocab=24, medium
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


class ResortConfig(BaseConfig):
    """Configuration for RESORT model with enhanced dense feature processing"""

    def __init__(self, use_dense_features=True, dense_proj_type='mlp'):
        # Features
        self.sparse_feats = self.SPARSE_FEATURES
        self.dense_feats = self.DENSE_FEATURES if use_dense_features else []

        # Embedding dimensions
        self.embed_dim = 16
        self.embed_dims = self.ADAPTIVE_EMBED_DIMS  # Use adaptive embedding

        # Dense feature processing (enhanced for better utilization)
        self.dense_proj_type = dense_proj_type  # 'linear', 'mlp', 'deep_mlp'
        self.dense_hidden_dims = [128, 64]  # Hidden dimensions for MLP projection
        self.dense_use_bn = True  # Batch normalization
        self.dense_dropout = 0.2  # Dropout rate for dense projection

        # MSR parameters
        self.msr_layers = 2
        self.msr_dim = 32
        self.num_heads = 2
        self.dropout = 0.1
        self.use_scale = False
        self.group_norm = True

        # Feature selection
        self.use_fs = True
        self.fs_hidden_units = [64]


class TrainingConfig:
    """Training hyperparameters"""

    # Data
    TRAIN_BATCH_SIZE = 20480
    VAL_BATCH_SIZE = 8192
    NUM_WORKERS = 4

    # Optimization
    LEARNING_RATE = 0.003
    WEIGHT_DECAY = 1e-5
    EPOCHS = 20

    # Scheduler (CosineAnnealingWarmRestarts)
    T_0 = 5
    T_MULT = 2
    ETA_MIN = 1e-6

    # Regularization
    GRADIENT_CLIP_NORM = 1.0
    DROPOUT = 0.1

    # Early Stopping
    PATIENCE = 5

    # SWA (Stochastic Weight Averaging)
    SWA_START_RATIO = 0.7  # Start SWA at 70% of total epochs
    SWA_LR = 0.005

    # Loss
    LABEL_SMOOTHING = 0.0
    USE_FOCAL_LOSS = False
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # Data paths
    DATA_DIR = './data/processed'
    TRAIN_DATA_PATH = '/home/lim/project/data/train.parquet'
    TEST_DATA_PATH = '/home/lim/project/data/test.parquet'
