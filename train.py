import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import pickle
import os
import json
import logging
from datetime import datetime
import shutil
import time

from model import MultiTaskTransformer, ModelConfig, ctr_loss, focal_loss

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
        
        # Labels (only CTR prediction for this dataset)
        labels = torch.tensor(row['clicked'], dtype=torch.long)
        
        return {
            **sparse_features,
            'dense_features': dense_features,
            'labels': labels
        }

def prepare_data():
    """데이터를 로드하고 전처리한 후 train/val/test로 분리하여 저장"""
    print("Loading and preprocessing data...")

    # Load data
    df = pd.read_parquet('/home/lim/project/data/train.parquet')
    #df = df.sample(n=50000, random_state=42)  # 5만 샘플로 빠른 테스트

    # Handle NaN values
    df = df.fillna(0)

    # 범주형 데이터 인코딩
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    label_encoders = {}

    for col in categorical_cols:
        if col != 'clicked':  # 라벨 제외
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Dense features 정규화
    config = ModelConfig()
    dense_cols = config.dense_feats
    scaler = StandardScaler()

    # Dense features가 존재하는 경우에만 정규화 적용
    existing_dense_cols = [col for col in dense_cols if col in df.columns]
    if existing_dense_cols:
        df[existing_dense_cols] = scaler.fit_transform(df[existing_dense_cols])
        print(f"Normalized {len(existing_dense_cols)} dense features: {existing_dense_cols}")
    else:
        scaler = None
        print("No dense features found for normalization")

    # Split data (train/val only, test.parquet is for final submission)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Create data directory if not exists
    data_dir = './data/processed'
    os.makedirs(data_dir, exist_ok=True)

    # Save datasets
    print("Saving train/val datasets...")
    train_df.to_parquet(f'{data_dir}/train_data.parquet', index=False)
    val_df.to_parquet(f'{data_dir}/val_data.parquet', index=False)

    # Save preprocessors
    preprocessors = {
        'label_encoders': label_encoders,
        'scaler': scaler
    }
    with open(f'{data_dir}/preprocessors.pkl', 'wb') as f:
        pickle.dump(preprocessors, f)

    print(f"Data saved to {data_dir}/")
    print(f"Preprocessors saved: label_encoders, scaler={'available' if scaler else 'none'}")
    return train_df, val_df, label_encoders, scaler

def load_prepared_data():
    """저장된 train/val 데이터와 전처리기 로드"""
    data_dir = './data/processed'

    if not os.path.exists(f'{data_dir}/train_data.parquet'):
        print("Prepared data not found. Preparing data first...")
        return prepare_data()

    print("Loading prepared data...")
    train_df = pd.read_parquet(f'{data_dir}/train_data.parquet')
    val_df = pd.read_parquet(f'{data_dir}/val_data.parquet')

    # Load preprocessors
    with open(f'{data_dir}/preprocessors.pkl', 'rb') as f:
        preprocessors = pickle.load(f)

    label_encoders = preprocessors['label_encoders']
    scaler = preprocessors['scaler']

    print(f"Loaded - Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"Preprocessors loaded: {len(label_encoders)} label encoders, scaler={'available' if scaler else 'none'}")

    return train_df, val_df, label_encoders, scaler

def setup_logging_and_save_dir():
    """로깅 설정 및 저장 디렉토리 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./experiments/exp_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)
    os.makedirs(f"{save_dir}/models", exist_ok=True)
    # Create dedicated models directory in root
    os.makedirs("./saved_models", exist_ok=True)

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{save_dir}/logs/training.log"),
            logging.StreamHandler()
        ]
    )

    return save_dir

def evaluate_model(model, test_loader, device, model_path=None):
    """모델 평가 함수"""
    logging.info("Starting model evaluation...")

    # 저장된 모델 로드 (경로가 주어진 경우)
    if model_path and os.path.exists(model_path):
        logging.info(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path))

    model.eval()

    test_preds_ctr = []
    test_labels_ctr = []

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Evaluating')
        for batch in test_bar:
            batch_input = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            logits = model(batch_input)
            preds = torch.sigmoid(logits)

            test_preds_ctr.extend(preds.squeeze().cpu().numpy())
            test_labels_ctr.extend(labels.cpu().numpy())

    # Test metrics
    test_ctr_auc = roc_auc_score(test_labels_ctr, test_preds_ctr)
    test_ctr_logloss = log_loss(test_labels_ctr, test_preds_ctr)
    test_brier_score = brier_score_loss(test_labels_ctr, test_preds_ctr)

    # Calibration metrics
    # Expected Calibration Error (ECE)
    prob_true, prob_pred = calibration_curve(test_labels_ctr, test_preds_ctr, n_bins=10, strategy='uniform')
    ece = np.mean(np.abs(prob_true - prob_pred))

    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(prob_true - prob_pred))

    # Mean absolute calibration error per bin
    mean_predicted_prob = np.mean(test_preds_ctr)
    mean_actual_prob = np.mean(test_labels_ctr)
    calibration_gap = abs(mean_predicted_prob - mean_actual_prob)

    test_results = {
        'ctr_auc': test_ctr_auc,
        'ctr_logloss': test_ctr_logloss,
        'brier_score': test_brier_score,
        'ece': ece,
        'mce': mce,
        'calibration_gap': calibration_gap,
        'mean_predicted_prob': mean_predicted_prob,
        'mean_actual_prob': mean_actual_prob,
        'num_test_samples': len(test_labels_ctr),
        'calibration_curve': {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist()
        }
    }

    # Logging
    logging.info("="*50)
    logging.info("EVALUATION RESULTS:")
    logging.info(f"CTR AUC: {test_ctr_auc:.4f}")
    logging.info(f"CTR LogLoss: {test_ctr_logloss:.4f}")
    logging.info(f"Brier Score: {test_brier_score:.4f}")
    logging.info(f"Expected Calibration Error (ECE): {ece:.4f}")
    logging.info(f"Maximum Calibration Error (MCE): {mce:.4f}")
    logging.info(f"Calibration Gap: {calibration_gap:.4f}")
    logging.info(f"Mean Predicted Prob: {mean_predicted_prob:.4f}")
    logging.info(f"Mean Actual Prob: {mean_actual_prob:.4f}")
    logging.info(f"Test Samples: {len(test_labels_ctr)}")
    logging.info("="*50)

    return test_results

def save_training_artifacts(save_dir, model, config, train_history, test_results):
    """학습 관련 모든 파일들을 저장"""

    # 성능 지표를 포함한 파일명 생성
    test_auc = test_results['ctr_auc']
    test_logloss = test_results['ctr_logloss']
    best_epoch = train_history['best_epoch']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 성능 기반 파일명 형식: auc{auc:.4f}_logloss{logloss:.4f}_epoch{epoch}_{timestamp}
    perf_suffix = f"auc{test_auc:.4f}_logloss{test_logloss:.4f}_epoch{best_epoch}_{timestamp}"

    # 1. 실험 폴더 내 모델 저장 (.pth)
    torch.save(model.state_dict(), f"{save_dir}/models/best_model_{perf_suffix}.pth")
    torch.save(model, f"{save_dir}/models/full_model_{perf_suffix}.pth")

    # 2. model.py 파일 복사
    if os.path.exists("model.py"):
        shutil.copy2("model.py", f"{save_dir}/model.py")

    # 3. train.py 파일 복사
    shutil.copy2(__file__, f"{save_dir}/train.py")

    # 4. 설정 저장
    config_dict = {
        'sparse_feats': config.sparse_feats,
        'dense_feats': config.dense_feats,
        'embed_dim': config.embed_dim,
        'd_model': config.d_model,
        'hidden_dims': config.hidden_dims
    }
    with open(f"{save_dir}/config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    # 5. 학습 히스토리 저장
    with open(f"{save_dir}/training_history.json", 'w') as f:
        json.dump(train_history, f, indent=2)

    # 6. 테스트 결과 저장
    with open(f"{save_dir}/test_results.json", 'w') as f:
        json.dump(test_results, f, indent=2)

    # 7. 요약 정보 저장
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'MultiTaskTransformer',
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        'final_test_auc': test_results['ctr_auc'],
        'final_test_logloss': test_results['ctr_logloss'],
        'best_epoch': train_history['best_epoch'],
        'best_val_loss': train_history['best_val_loss'],
        'model_filename': f"model_{perf_suffix}.pth"
    }

    with open(f"{save_dir}/experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"All training artifacts saved to: {save_dir}")
    logging.info(f"Performance-based model saved to: ./saved_models/model_{perf_suffix}.pth")
    return save_dir

def train_model(use_focal_loss=False, label_smoothing=0.0):
    # 실험 디렉토리 및 로깅 설정
    save_dir = setup_logging_and_save_dir()
    logging.info("Starting CTR model training...")
    logging.info(f"Loss configuration: Focal Loss={use_focal_loss}, Label Smoothing={label_smoothing}")

    # Load or prepare data
    train_df, val_df, label_encoders, scaler = load_prepared_data()
    logging.info(f"Data loaded - Train: {len(train_df)}, Val: {len(val_df)}")

    # Create config and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    config = ModelConfig()
    model = MultiTaskTransformer(config).to(device)

    # 모델 컴파일 최적화 (PyTorch 2.0+)
    try:
        model = torch.compile(model)
        logging.info("Model compiled with torch.compile for faster training")
    except:
        logging.info("torch.compile not available, using standard model")

    logging.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create datasets and dataloaders
    train_dataset = CTRDataset(train_df, config)
    val_dataset = CTRDataset(val_df, config)

    # 최적화된 DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True,
                             num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=8192, shuffle=False,
                           num_workers=2, pin_memory=True, persistent_workers=True)

    # Optimizer and scheduler (Warmup 포함된 CosineAnnealingWarmRestarts)
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    # Mixed Precision Training 설정
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    use_amp = device.type == 'cuda'
    logging.info(f"Mixed Precision Training: {'Enabled' if use_amp else 'Disabled'}")

    # SWA (Stochastic Weight Averaging) 설정
    swa_model = AveragedModel(model)
    swa_start = int(epochs * 0.7)  # 전체 epochs의 70%부터 SWA 시작
    swa_scheduler = SWALR(optimizer, swa_lr=0.005)
    logging.info(f"SWA will start from epoch {swa_start + 1}/{epochs}")

    # Training history
    train_history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'val_logloss': [],
        'learning_rates': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }

    # Training loop
    epochs = 20
    best_val_loss = float('inf')

    # Early Stopping parameters
    patience = 5
    patience_counter = 0

    # 전체 학습 시간 측정 시작
    total_start_time = time.time()
    logging.info("Starting training loop...")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        # Training
        model.train()
        train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} Training')
        for batch in train_bar:
            optimizer.zero_grad()

            # Prepare batch and move to device
            batch_input = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            # Mixed Precision Forward pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    preds = model(batch_input)
                    if use_focal_loss:
                        loss = focal_loss(preds, labels)
                    else:
                        loss = ctr_loss(preds, labels, label_smoothing=label_smoothing)
                # Backward pass with scaling
                scaler.scale(loss).backward()
                # Gradient Clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass
                preds = model(batch_input)
                if use_focal_loss:
                    loss = focal_loss(preds, labels)
                else:
                    loss = ctr_loss(preds, labels, label_smoothing=label_smoothing)
                loss.backward()
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()

            # Update progress bar with current loss
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{train_loss/(train_bar.n+1):.4f}'
            })

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds_ctr = []
        val_labels_ctr = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} Validation')
            for batch in val_bar:
                batch_input = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)

                logits = model(batch_input)
                loss = ctr_loss(logits, labels)
                val_loss += loss.item()

                preds = torch.sigmoid(logits)
                val_preds_ctr.extend(preds.squeeze().cpu().numpy())
                val_labels_ctr.extend(labels.cpu().numpy())

                # Update validation progress bar
                val_bar.set_postfix({
                    'Val Loss': f'{loss.item():.4f}',
                    'Avg Val Loss': f'{val_loss/(val_bar.n+1):.4f}'
                })
        
        # Calculate metrics
        val_auc = roc_auc_score(val_labels_ctr, val_preds_ctr)
        val_logloss = log_loss(val_labels_ctr, val_preds_ctr)

        # Record history
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        train_history['train_loss'].append(avg_train_loss)
        train_history['val_loss'].append(avg_val_loss)
        train_history['val_auc'].append(val_auc)
        train_history['val_logloss'].append(val_logloss)
        train_history['learning_rates'].append(current_lr)

        # 에포크 시간 측정
        epoch_time = time.time() - epoch_start_time

        # Logging
        logging.info(f'Epoch {epoch+1}/{epochs}:')
        logging.info(f'  Train Loss: {avg_train_loss:.4f}')
        logging.info(f'  Val Loss: {avg_val_loss:.4f}')
        logging.info(f'  Val AUC: {val_auc:.4f}')
        logging.info(f'  Val LogLoss: {val_logloss:.4f}')
        logging.info(f'  Learning Rate: {current_lr:.6f}')
        logging.info(f'  Epoch Time: {epoch_time:.2f}s')

        # ETA 계산
        avg_epoch_time = (time.time() - total_start_time) / (epoch + 1)
        remaining_epochs = epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_minutes = eta_seconds / 60
        logging.info(f'  ETA: {eta_minutes:.1f} minutes ({eta_seconds:.0f}s)')

        # Save best model with performance info
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            train_history['best_epoch'] = epoch + 1
            train_history['best_val_loss'] = best_val_loss
            patience_counter = 0  # Reset patience counter

            # 성능 정보를 포함한 체크포인트 파일명
            checkpoint_name = f"checkpoint_epoch{epoch+1}_auc{val_auc:.4f}_logloss{val_logloss:.4f}.pth"
            torch.save(model.state_dict(), f'{save_dir}/models/{checkpoint_name}')

            # 기본 체크포인트도 저장 (호환성 유지)
            torch.save(model.state_dict(), f'{save_dir}/models/best_model_checkpoint.pth')

            logging.info(f'  New best model saved at epoch {epoch+1}!')
            logging.info(f'  Checkpoint: {checkpoint_name}')
        else:
            patience_counter += 1
            logging.info(f'  No improvement. Patience: {patience_counter}/{patience}')

        # Early Stopping check
        if patience_counter >= patience:
            logging.info(f'Early stopping triggered after {epoch+1} epochs!')
            logging.info(f'Best validation loss: {best_val_loss:.4f} at epoch {train_history["best_epoch"]}')
            break

        # SWA Update
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            logging.info(f'  SWA model updated (epoch {epoch+1 - swa_start}/{epochs - swa_start})')
        else:
            scheduler.step()

    # SWA 최종 처리: BatchNorm statistics 업데이트
    if epochs > swa_start:
        logging.info("Updating SWA BatchNorm statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

        # SWA 모델 평가
        logging.info("Evaluating SWA model on validation set...")
        swa_model.eval()
        swa_val_loss = 0.0
        swa_val_preds = []
        swa_val_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='SWA Validation'):
                batch_input = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)

                logits = swa_model(batch_input)
                loss = ctr_loss(logits, labels)
                swa_val_loss += loss.item()

                preds = torch.sigmoid(logits)
                swa_val_preds.extend(preds.squeeze().cpu().numpy())
                swa_val_labels.extend(labels.cpu().numpy())

        swa_val_auc = roc_auc_score(swa_val_labels, swa_val_preds)
        swa_val_logloss = log_loss(swa_val_labels, swa_val_preds)
        swa_avg_val_loss = swa_val_loss / len(val_loader)

        logging.info(f'SWA Model Performance:')
        logging.info(f'  Val Loss: {swa_avg_val_loss:.4f}')
        logging.info(f'  Val AUC: {swa_val_auc:.4f}')
        logging.info(f'  Val LogLoss: {swa_val_logloss:.4f}')

        # SWA 모델이 더 좋으면 저장
        if swa_avg_val_loss < best_val_loss:
            logging.info(f'SWA model outperforms best checkpoint! Saving SWA model...')
            torch.save(swa_model.state_dict(), f'{save_dir}/models/swa_model.pth')
            model = swa_model  # 테스트에 SWA 모델 사용
            train_history['swa_val_loss'] = swa_avg_val_loss
            train_history['swa_val_auc'] = swa_val_auc
            train_history['swa_val_logloss'] = swa_val_logloss
        else:
            logging.info(f'Best checkpoint still better than SWA model.')
            train_history['swa_val_loss'] = swa_avg_val_loss
            train_history['swa_val_auc'] = swa_val_auc
            train_history['swa_val_logloss'] = swa_val_logloss
    
    # Validation evaluation (using val_loader for final metrics)
    test_results = evaluate_model(model, val_loader, device, f'{save_dir}/models/best_model_checkpoint.pth')

    # Save all training artifacts
    final_save_dir = save_training_artifacts(save_dir, model, config, train_history, test_results)

    logging.info("Training completed successfully!")
    logging.info(f"All results saved to: {final_save_dir}")

    return model, final_save_dir

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CTR Model Training')
    parser.add_argument('--prepare-data', action='store_true',
                       help='Force data preparation even if processed data exists')
    parser.add_argument('--focal-loss', action='store_true',
                       help='Use Focal Loss instead of BCE')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                       help='Label smoothing factor (0.0 to 0.5)')

    args = parser.parse_args()

    if args.prepare_data:
        print("Preparing data...")
        prepare_data()

    model, save_dir = train_model(use_focal_loss=args.focal_loss, label_smoothing=args.label_smoothing)
    print(f"\n🎉 Training completed! All artifacts saved to: {save_dir}")

def evaluate_saved_model(model_path, data_path=None):
    """저장된 모델을 평가하는 독립 함수"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터 로드
    if data_path:
        val_df = pd.read_parquet(data_path)
    else:
        _, val_df, _, _ = load_prepared_data()

    # 모델 설정 및 로드
    config = ModelConfig()
    model = MultiTaskTransformer(config).to(device)

    # 데이터셋 생성
    val_dataset = CTRDataset(val_df, config)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

    # 평가 실행
    results = evaluate_model(model, val_loader, device, model_path)

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CTR Model Training and Evaluation')
    parser.add_argument('--prepare-data', action='store_true',
                       help='Force data preparation even if processed data exists')
    parser.add_argument('--eval-only', type=str,
                       help='Evaluate saved model only. Provide path to .pth file')
    parser.add_argument('--eval-data', type=str,
                       help='Path to test data for evaluation (optional)')
    parser.add_argument('--focal-loss', action='store_true',
                       help='Use Focal Loss instead of BCE')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                       help='Label smoothing factor (0.0 to 0.5)')

    args = parser.parse_args()

    if args.eval_only:
        print(f"Evaluating model: {args.eval_only}")
        results = evaluate_saved_model(args.eval_only, args.eval_data)
        print(f"Results: AUC={results['ctr_auc']:.4f}, LogLoss={results['ctr_logloss']:.4f}")
    else:
        if args.prepare_data:
            print("Preparing data...")
            prepare_data()

        model, save_dir = train_model(use_focal_loss=args.focal_loss, label_smoothing=args.label_smoothing)
        print(f"\n🎉 Training completed! All artifacts saved to: {save_dir}")