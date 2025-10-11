#!/usr/bin/env python3
"""
Script to analyze and compare results from all models
"""

import os
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict

def extract_metrics_from_log(log_file):
    """Extract validation and test metrics from log file"""
    metrics = {}

    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # Look for validation and test results
        # Pattern: logloss: 0.1234, AUC: 0.9876
        patterns = {
            'val_logloss': r'Validation.*?logloss[:\s]+([0-9.]+)',
            'val_auc': r'Validation.*?AUC[:\s]+([0-9.]+)',
            'test_logloss': r'Test.*?logloss[:\s]+([0-9.]+)',
            'test_auc': r'Test.*?AUC[:\s]+([0-9.]+)',
        }

        for key, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # Take the last occurrence
                metrics[key] = float(matches[-1])

    except Exception as e:
        print(f"Error reading {log_file}: {e}")

    return metrics

def main():
    results_dir = Path("/home/lim/project/RE-SORT/SOTAS/results")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Find all log files
    log_files = list(results_dir.glob("*_*.log"))

    if not log_files:
        print("No log files found in results directory")
        return

    print(f"Found {len(log_files)} log files")
    print()

    # Extract model name and metrics
    results = []

    for log_file in log_files:
        # Extract model name from filename (e.g., "Baseline_20231006_120000.log")
        filename = log_file.stem
        parts = filename.split('_')

        if len(parts) >= 2:
            # Model name might contain underscores, so join all parts except last 2 (date and time)
            model_name = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]

            metrics = extract_metrics_from_log(log_file)

            if metrics:
                result = {'model': model_name, **metrics}
                results.append(result)
                print(f"✓ Extracted metrics from {model_name}")
            else:
                print(f"✗ No metrics found in {model_name}")

    if not results:
        print("\nNo results extracted!")
        return

    # Create DataFrame
    df = pd.DataFrame(results)

    # Group by model and take the most recent run
    df_latest = df.groupby('model').last().reset_index()

    # Sort by validation AUC (descending)
    if 'val_auc' in df_latest.columns:
        df_latest = df_latest.sort_values('val_auc', ascending=False)

    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print()

    # Print formatted table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.6f}'.format)

    print(df_latest.to_string(index=False))
    print()

    # Save to CSV
    output_csv = results_dir / "model_comparison.csv"
    df_latest.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

    # Find best model
    if 'val_auc' in df_latest.columns:
        best_idx = df_latest['val_auc'].idxmax()
        best_model = df_latest.loc[best_idx]

        print()
        print("="*80)
        print("BEST MODEL")
        print("="*80)
        print(f"Model: {best_model['model']}")
        if 'val_auc' in best_model:
            print(f"Validation AUC: {best_model['val_auc']:.6f}")
        if 'val_logloss' in best_model:
            print(f"Validation Logloss: {best_model['val_logloss']:.6f}")
        if 'test_auc' in best_model:
            print(f"Test AUC: {best_model['test_auc']:.6f}")
        if 'test_logloss' in best_model:
            print(f"Test Logloss: {best_model['test_logloss']:.6f}")
        print("="*80)

if __name__ == '__main__':
    main()
