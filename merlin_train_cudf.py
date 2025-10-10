#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

print("✅ Environment configured")

# Core imports
import gc
import time
from datetime import datetime
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import psutil

# GPU libraries
import cudf
import cupy as cp

# ML libraries
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

print("✅ All libraries imported successfully")

# Configuration
TRAIN_PATH = '/home/lim/project/data/train.parquet'
TEST_PATH = '/home/lim/project/data/test.parquet'
OUTPUT_DIR = '/home/lim/project/data/cudf_processed'
MODEL_NAME = 'xgboost_cudf'
DATE_STR = datetime.now().strftime('%Y%m%d_%H%M%S')
SUBMISSION_PATH = f'submission_{MODEL_NAME}_{DATE_STR}.csv'
N_FOLDS = 5
SKIP_CV = False  # Set to True to skip cross-validation and only generate submission

print(f"📋 Configuration:")
print(f"   Train: {TRAIN_PATH}")
print(f"   Test: {TEST_PATH}")
print(f"   Output: {OUTPUT_DIR}")
print(f"   Submission: {SUBMISSION_PATH}")
print(f"   Folds: {N_FOLDS}")
print(f"   Skip CV: {SKIP_CV}")

# Memory management functions
def print_memory():
    """Print current memory usage"""
    mem = psutil.virtual_memory()

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_used = gpu_info.used / 1024**3
        gpu_total = gpu_info.total / 1024**3
    except:
        gpu_used = 0
        gpu_total = 0

    print(f"💾 CPU: {mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB ({mem.percent:.1f}%)")
    print(f"💾 GPU: {gpu_used:.1f}GB/{gpu_total:.1f}GB")
    return mem.percent

def clear_gpu_memory():
    """Clear GPU memory"""
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    print("🧹 GPU memory cleared")

# Metric functions
def calculate_weighted_logloss(y_true, y_pred, eps=1e-15):
    """Calculate Weighted LogLoss with 50:50 class weights"""
    y_pred = np.clip(y_pred, eps, 1 - eps)

    mask_0 = (y_true == 0)
    mask_1 = (y_true == 1)

    ll_0 = -np.mean(np.log(1 - y_pred[mask_0])) if mask_0.sum() > 0 else 0
    ll_1 = -np.mean(np.log(y_pred[mask_1])) if mask_1.sum() > 0 else 0

    return 0.5 * ll_0 + 0.5 * ll_1

def calculate_competition_score(y_true, y_pred):
    """Calculate competition score: 0.5*AP + 0.5*(1/(1+WLL))"""
    ap = average_precision_score(y_true, y_pred)
    wll = calculate_weighted_logloss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return score, ap, wll

print("✅ Metric functions defined")

# Data processing with cuDF
def process_data_cudf():
    """Process data with cuDF (without nvtabular)"""
    print("\n" + "="*70)
    print("🚀 cuDF Data Processing")
    print("="*70)

    start_time = time.time()
    initial_mem = print_memory()

    # Load data
    print("\n📦 Loading data...")
    # Read columns (exclude 'seq')
    pf = pq.ParquetFile(TRAIN_PATH)
    cols = [c for c in pf.schema.names if c != 'seq']
    print(f"   Total columns: {len(pf.schema.names)}")
    print(f"   Using columns: {len(cols)} (excluded 'seq')")

    # Load with cuDF
    gdf = cudf.read_parquet(TRAIN_PATH, columns=cols)
    print(f"   Loaded {len(gdf):,} rows")
    print_memory()

    # Define categorical and continuous columns
    categorical = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    continuous = (
        [f'feat_a_{i}' for i in range(1, 19)] +
        [f'feat_b_{i}' for i in range(1, 7)] +
        [f'feat_c_{i}' for i in range(1, 9)] +
        [f'feat_d_{i}' for i in range(1, 7)] +
        [f'feat_e_{i}' for i in range(1, 11)] +
        [f'history_a_{i}' for i in range(1, 8)] +
        [f'history_b_{i}' for i in range(1, 31)] +
        [f'l_feat_{i}' for i in range(1, 28)]
    )

    print(f"\n📊 Feature types:")
    print(f"   Categorical: {len(categorical)} columns")
    print(f"   Continuous: {len(continuous)} columns")

    # Process categorical features (label encoding)
    print("\n🔧 Processing categorical features...")
    for col in categorical:
        if col in gdf.columns:
            gdf[col] = gdf[col].astype('category').cat.codes.astype('float32')

    # Process continuous features (fill missing)
    print("🔧 Processing continuous features...")
    for col in continuous:
        if col in gdf.columns:
            gdf[col] = gdf[col].fillna(0).astype('float32')

    # Ensure target is correct type
    gdf['clicked'] = gdf['clicked'].astype('int32')

    elapsed = time.time() - start_time
    final_mem = print_memory()

    print(f"\n✅ Processing complete!")
    print(f"   Time: {elapsed:.1f}s")
    print(f"   Memory increase: +{final_mem - initial_mem:.1f}%")

    return gdf

def process_test_data_cudf():
    """Process test data with cuDF (same as train processing)"""
    print("\n" + "="*70)
    print("🚀 cuDF Test Data Processing")
    print("="*70)

    start_time = time.time()
    initial_mem = print_memory()

    # Load data
    print("\n📦 Loading test data...")
    pf = pq.ParquetFile(TEST_PATH)
    cols = [c for c in pf.schema.names if c not in ['seq', 'clicked']]
    print(f"   Total columns: {len(pf.schema.names)}")
    print(f"   Using columns: {len(cols)}")

    # Load with cuDF
    gdf = cudf.read_parquet(TEST_PATH, columns=cols)
    print(f"   Loaded {len(gdf):,} rows")

    # Keep id column for submission
    test_ids = gdf['id'].to_pandas()
    print_memory()

    # Define categorical and continuous columns
    categorical = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    continuous = (
        [f'feat_a_{i}' for i in range(1, 19)] +
        [f'feat_b_{i}' for i in range(1, 7)] +
        [f'feat_c_{i}' for i in range(1, 9)] +
        [f'feat_d_{i}' for i in range(1, 7)] +
        [f'feat_e_{i}' for i in range(1, 11)] +
        [f'history_a_{i}' for i in range(1, 8)] +
        [f'history_b_{i}' for i in range(1, 31)] +
        [f'l_feat_{i}' for i in range(1, 28)]
    )

    # Process categorical features (label encoding)
    print("\n🔧 Processing categorical features...")
    for col in categorical:
        if col in gdf.columns:
            gdf[col] = gdf[col].astype('category').cat.codes.astype('float32')

    # Process continuous features (fill missing)
    print("🔧 Processing continuous features...")
    for col in continuous:
        if col in gdf.columns:
            gdf[col] = gdf[col].fillna(0).astype('float32')

    # Drop id column for model input
    gdf = gdf.drop('id', axis=1)

    elapsed = time.time() - start_time
    final_mem = print_memory()

    print(f"\n✅ Test data processing complete!")
    print(f"   Time: {elapsed:.1f}s")
    print(f"   Memory increase: +{final_mem - initial_mem:.1f}%")

    return gdf, test_ids

# Cross-validation
def run_cv(gdf, n_folds=5):
    """Run stratified cross-validation"""
    print("\n" + "="*70)
    print("🔄 Stratified KFold Cross-Validation")
    print("="*70)

    print_memory()

    # Prepare data
    print("\n📊 Preparing data for XGBoost...")
    y = gdf['clicked'].to_numpy()
    X = gdf.drop('clicked', axis=1)
    X_np = X.to_numpy()

    print(f"   Shape: {X_np.shape}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]:,}")

    # Class distribution
    pos_ratio = y.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio
    print(f"\n📊 Class distribution:")
    print(f"   Positive ratio: {pos_ratio:.4f}")
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")

    del X, gdf
    clear_gpu_memory()

    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'gpu_id': 0,
        'verbosity': 1,
        'seed': 42
    }

    # Cross-validation
    print("\n🔄 Starting cross-validation...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_scores = []
    cv_ap = []
    cv_wll = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y), 1):
        print(f"\n📍 Fold {fold}/{n_folds}")
        fold_start = time.time()

        # Create DMatrix
        print(f"   Train: {len(train_idx):,} | Val: {len(val_idx):,}")
        dtrain = xgb.DMatrix(X_np[train_idx], label=y[train_idx])
        dval = xgb.DMatrix(X_np[val_idx], label=y[val_idx])

        # Train
        print("   Training...")
        model = xgb.train(
            params, dtrain,
            num_boost_round=200,
            evals=[(dval, 'val')],
            early_stopping_rounds=20,
            verbose_eval=50
        )

        # Evaluate
        y_pred = model.predict(dval)
        score, ap, wll = calculate_competition_score(y[val_idx], y_pred)

        cv_scores.append(score)
        cv_ap.append(ap)
        cv_wll.append(wll)

        print(f"   📊 Results:")
        print(f"      Score: {score:.6f}")
        print(f"      AP: {ap:.6f}")
        print(f"      WLL: {wll:.6f}")
        print(f"      Best iteration: {model.best_iteration}")
        print(f"   ⏱️ Time: {time.time() - fold_start:.1f}s")

        # Cleanup
        del dtrain, dval, model
        clear_gpu_memory()

    # Final results
    print("\n" + "="*70)
    print("📊 Final Cross-Validation Results")
    print("="*70)

    print(f"\n🏆 Competition Score: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
    print(f"📈 Average Precision: {np.mean(cv_ap):.6f} ± {np.std(cv_ap):.6f}")
    print(f"📉 Weighted LogLoss: {np.mean(cv_wll):.6f} ± {np.std(cv_wll):.6f}")

    print(f"\nAll fold scores: {[f'{s:.6f}' for s in cv_scores]}")

    return cv_scores

def train_full_and_predict(gdf_train, gdf_test, test_ids, best_iterations=200):
    """Train on full dataset and predict on test"""
    print("\n" + "="*70)
    print("🎯 Training on Full Dataset & Predicting on Test")
    print("="*70)

    print_memory()

    # Prepare train data
    print("\n📊 Preparing training data...")
    y_train = gdf_train['clicked'].to_numpy()
    X_train = gdf_train.drop('clicked', axis=1)
    X_train_np = X_train.to_numpy()

    print(f"   Train shape: {X_train_np.shape}")
    print(f"   Features: {X_train.shape[1]}")

    # Prepare test data
    print("\n📊 Preparing test data...")
    X_test_np = gdf_test.to_numpy()
    print(f"   Test shape: {X_test_np.shape}")

    # Class distribution
    pos_ratio = y_train.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio
    print(f"\n📊 Class distribution:")
    print(f"   Positive ratio: {pos_ratio:.4f}")
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")

    del X_train, gdf_train, gdf_test
    clear_gpu_memory()

    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'gpu_id': 0,
        'verbosity': 1,
        'seed': 42
    }

    # Train on full dataset
    print("\n🚀 Training on full dataset...")
    train_start = time.time()

    dtrain = xgb.DMatrix(X_train_np, label=y_train)
    dtest = xgb.DMatrix(X_test_np)

    print(f"   Training for {best_iterations} rounds...")
    model = xgb.train(
        params, dtrain,
        num_boost_round=best_iterations,
        verbose_eval=50
    )

    print(f"   ⏱️ Training time: {time.time() - train_start:.1f}s")

    # Predict on test
    print("\n🔮 Predicting on test data...")
    pred_start = time.time()
    y_pred = model.predict(dtest)
    print(f"   ⏱️ Prediction time: {time.time() - pred_start:.1f}s")
    print(f"   Predictions: min={y_pred.min():.6f}, max={y_pred.max():.6f}, mean={y_pred.mean():.6f}")

    # Create submission
    print("\n💾 Creating submission file...")
    submission = pd.DataFrame({
        'id': test_ids,
        'clicked': y_pred
    })

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"   ✅ Saved to: {SUBMISSION_PATH}")
    print(f"   Rows: {len(submission):,}")

    # Cleanup
    del dtrain, dtest, model
    clear_gpu_memory()

    return submission

# Main execution
if __name__ == "__main__":
    # Process train data
    print("\n" + "="*70)
    print("📂 STEP 1: Process Training Data")
    print("="*70)
    gdf_train = process_data_cudf()

    # Run cross-validation (optional)
    cv_scores = None
    if not SKIP_CV:
        print("\n" + "="*70)
        print("📂 STEP 2: Cross-Validation")
        print("="*70)
        cv_scores = run_cv(gdf_train, N_FOLDS)

        if cv_scores:
            print(f"\n✅ CV Score: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")

        # Reload train data (it was modified during CV)
        gdf_train = process_data_cudf()
    else:
        print("\n⏭️  Skipping cross-validation (SKIP_CV=True)")

    # Process test data
    step_num = 3 if not SKIP_CV else 2
    print("\n" + "="*70)
    print(f"📂 STEP {step_num}: Process Test Data")
    print("="*70)
    gdf_test, test_ids = process_test_data_cudf()

    # Train on full dataset and predict
    step_num = 4 if not SKIP_CV else 3
    print("\n" + "="*70)
    print(f"📂 STEP {step_num}: Train Full Model & Generate Submission")
    print("="*70)
    submission = train_full_and_predict(gdf_train, gdf_test, test_ids, best_iterations=200)

    # Final summary
    print("\n" + "🎉"*35)
    print("COMPLETE!")
    print("🎉"*35)
    if cv_scores:
        print(f"\n✅ Final CV Score: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
    print(f"✅ Submission saved to: {SUBMISSION_PATH}")
    print(f"✅ Total predictions: {len(submission):,}")
    print("✅ Full dataset processed")
    print("✅ cuDF-based preprocessing (CUDA 12 compatible)")
    print("="*70)

    # Final cleanup
    clear_gpu_memory()
    print("\n🧹 Final cleanup complete")
    print_memory()
