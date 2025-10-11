# SOTAS Models Training Guide

## Data Setup

The training data has been split into:
- **train.parquet**: 9,633,761 rows (90%)
- **valid.parquet**: 1,070,418 rows (10%)
- **test.parquet**: test data

All model configs have been updated to use these data files.

## Quick Setup (빠른 설정)

### 첫 실행 전 필수 설정

```bash
cd /home/lim/project/RE-SORT

# 1. Python 경로 설정 (필수!)
export PYTHONPATH="/home/lim/project/RE-SORT:$PYTHONPATH"

# 2. 스크립트 수정
cd SOTAS
./fix_scripts.sh
```

### 일반적인 에러

| 에러 | 원인 | 해결 |
|------|------|------|
| "실행할 수 없음: 필요한 파일이 없습니다" | CRLF 줄바꿈 | `./fix_scripts.sh` |
| "Permission denied" | 실행 권한 없음 | `chmod +x *.sh` |
| "No module named 're_sortctr'" | Python 경로 미설정 | `export PYTHONPATH="...":$PYTHONPATH` |

**자세한 설정:** [SETUP.md](SETUP.md) 참고

---

## Running Models

### Option 1: Run All Models

Run all models sequentially with logging and GPU availability checking:

```bash
cd /home/lim/project/RE-SORT/SOTAS
./run_all_models.sh [gpu_id] [gpu_threshold] [max_wait_time]
```

Examples:
```bash
./run_all_models.sh              # Run on CPU
./run_all_models.sh 0            # Run on GPU 0 (waits if busy)
./run_all_models.sh 0 20         # Wait until GPU memory usage < 20%
./run_all_models.sh 0 10 7200    # Wait up to 2 hours for GPU
```

**GPU Checking:**
- Automatically checks GPU availability before starting
- Waits for GPU if memory usage is above threshold (default: 10%)
- Max wait time: 3600s (1 hour) by default
- Checks every 60 seconds

**Output:**
- Results displayed in terminal (with `tee`)
- `results/summary_YYYYMMDD_HHMMSS.txt` - Summary of all runs
- `results/detailed_log_YYYYMMDD_HHMMSS.log` - Full combined log
- `results/ModelName_YYYYMMDD_HHMMSS.log` - Individual model logs

### Option 2: Run Single Model

Run a specific model with logging and GPU availability checking:

```bash
cd /home/lim/project/RE-SORT/SOTAS
./run_single_model.sh <model_name> [gpu_id] [expid] [gpu_threshold] [max_wait_time]
```

Examples:
```bash
./run_single_model.sh Baseline                # Run Baseline on CPU
./run_single_model.sh Baseline 0              # Run Baseline on GPU 0 (waits if busy)
./run_single_model.sh Baseline 0 DeepFM_test # Run with specific experiment ID
./run_single_model.sh Baseline 0 "" 20        # Wait until GPU memory < 20%
```

**GPU Checking:**
- Same behavior as run_all_models.sh
- Automatically waits for GPU availability

**Output:**
- Results displayed in terminal (with `tee`)
- `results/ModelName_YYYYMMDD_HHMMSS.log` - Full training log
- `results/ModelName_YYYYMMDD_HHMMSS_summary.txt` - Quick summary with metrics

### Option 3: Check GPU Status

Check GPU availability manually:

```bash
cd /home/lim/project/RE-SORT/SOTAS
./check_gpu.sh [gpu_id] [threshold]
```

Examples:
```bash
./check_gpu.sh 0        # Check GPU 0 (default threshold: 10%)
./check_gpu.sh 0 20     # Check if GPU 0 memory usage < 20%
```

**Output example:**
```
GPU 0 Status:
  Memory: 1234MB / 24576MB (5%)
  Utilization: 12%
  Status: Available ✓
```

### Option 4: Manual Run

Navigate to a model directory and run directly:

```bash
cd /home/lim/project/RE-SORT/SOTAS/Baseline
python run_expid.py --config ./config/ --gpu 0
```

## Analyzing Results

After running models, analyze and compare results:

```bash
cd /home/lim/project/RE-SORT/SOTAS
python analyze_results.py
```

This will:
- Extract metrics (logloss, AUC) from all log files
- Display a comparison table of all models
- Identify the best performing model
- Save results to `results/model_comparison.csv`

**Example output:**
```
MODEL COMPARISON RESULTS
================================================================================

model                    val_logloss  val_auc   test_logloss  test_auc
Baseline                 0.123456     0.876543  0.124567      0.875432
Baseline_DoubleStream    0.120000     0.880000  0.121000      0.879000
...

BEST MODEL
================================================================================
Model: Baseline_DoubleStream
Validation AUC: 0.880000
Validation Logloss: 0.120000
...
```

## Available Models

- **Baseline**: Basic model
- **Baseline_DoubleStream**: Double stream baseline
- **Baseline_SA**: Self-attention baseline
- **DCNv2**: Deep & Cross Network v2
- **FM**: Factorization Machine
- **FinalMLP**: Final MLP architecture
- **FinalNet**: Final network architecture
- **LR**: Logistic Regression
- **MaskNet**: Mask network
- **xDeepFM**: eXtreme Deep FM

## Output Files

All outputs are saved in the `results/` directory:

```
results/
├── summary_20231006_120000.txt              # Summary of batch run
├── detailed_log_20231006_120000.log         # Full log of batch run
├── Baseline_20231006_120000.log             # Individual model logs
├── Baseline_20231006_120000_summary.txt     # Individual model summaries
├── ...
└── model_comparison.csv                     # Comparison table
```

## Configuration

All models use the `train_parquet` dataset configuration from their respective `config/dataset_config.yaml` files.

To modify training parameters, edit the model config files in each model's `config/` directory.

## Troubleshooting

### GPU Issues

**Problem: "nvidia-smi not found"**
```
ERROR: nvidia-smi not found. GPU monitoring not available.
Please install NVIDIA drivers or use CPU mode (gpu_id=-1)
```
**Solution:**
- Install NVIDIA drivers or use CPU mode by setting `gpu_id=-1`
- Run in CPU mode: `./run_all_models.sh -1` or `./run_single_model.sh Baseline`

**Problem: "nvidia-smi found but cannot access GPU"**
```
ERROR: nvidia-smi found but cannot access GPU.
Possible causes:
  - NVIDIA driver not loaded
  - Permission issues
  - No NVIDIA GPU available
```
**Solution:**
1. Run `nvidia-smi` manually to see detailed error
2. Check if driver is loaded: `lsmod | grep nvidia`
3. Reload driver: `sudo modprobe nvidia`
4. Or use CPU mode: `./run_all_models.sh -1`

**Problem: "GPU X not found"**
```
ERROR: GPU 10 not found.

Available GPUs:
GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-...)
```
**Solution:**
- Use an available GPU ID (check output above)
- Example: `./run_all_models.sh 0`

**Problem: GPU always busy**
```
Status: Busy (memory usage 91% >= 10%)
Waiting for GPU 0 to be available...
```
**Solution:**
1. Increase threshold: `./run_all_models.sh 0 20` (wait until < 20%)
2. Increase wait time: `./run_all_models.sh 0 10 7200` (wait up to 2 hours)
3. Use a different GPU: `./run_all_models.sh 1`
4. Use CPU mode: `./run_all_models.sh -1`

### Other Issues

**Problem: Permission denied when running scripts**
```
bash: ./run_all_models.sh: Permission denied
```
**Solution:**
```bash
chmod +x run_all_models.sh run_single_model.sh check_gpu.sh
```

**Problem: No results in output files**
```
Results saved to: results/summary_*.txt
```
**Solution:**
- Check if training completed successfully
- Look for errors in detailed log: `cat results/detailed_log_*.log`
- Check individual model logs: `cat results/ModelName_*.log`
