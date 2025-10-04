import pandas as pd
import numpy as np

# Load data sample
print('Analyzing sparse features for Adaptive Embedding...\n')
df = pd.read_parquet('/home/lim/project/data/train.parquet')
df = df.sample(n=min(200000, len(df)), random_state=42)

# Sparse features from model.py
sparse_features = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']

print('='*70)
print('SPARSE FEATURE ANALYSIS FOR ADAPTIVE EMBEDDING')
print('='*70)

results = []
for feat in sparse_features:
    n_unique = df[feat].nunique()
    top_10_ratio = df[feat].value_counts(normalize=True).head(10).sum()
    top_1_ratio = df[feat].value_counts(normalize=True).iloc[0]
    
    # Recommend embedding dimension based on cardinality
    if n_unique <= 10:
        recommended_dim = 8
    elif n_unique <= 50:
        recommended_dim = 16
    elif n_unique <= 200:
        recommended_dim = 32
    elif n_unique <= 1000:
        recommended_dim = 48
    else:
        recommended_dim = 64
    
    results.append({
        'feature': feat,
        'vocab_size': n_unique,
        'top_1_freq': f'{top_1_ratio:.2%}',
        'top_10_freq': f'{top_10_ratio:.2%}',
        'recommended_dim': recommended_dim
    })
    
    print(f'\n{feat}:')
    print(f'  Vocabulary Size: {n_unique:,}')
    print(f'  Top 1 Frequency: {top_1_ratio:.2%}')
    print(f'  Top 10 Frequency: {top_10_ratio:.2%}')
    print(f'  Recommended Embedding Dim: {recommended_dim}')
    print(f'  Top 5 values:')
    for val, count in df[feat].value_counts().head(5).items():
        print(f'    {val}: {count:,} ({count/len(df):.2%})')

print('\n' + '='*70)
print('SUMMARY - ADAPTIVE EMBEDDING CONFIGURATION')
print('='*70)

result_df = pd.DataFrame(results)
print(result_df.to_string(index=False))

print('\n' + '='*70)
print('TOTAL PARAMETERS COMPARISON:')
print('='*70)

# Current fixed embedding (all 16 dim)
current_params = sum([r['vocab_size'] * 16 for r in results])
# Adaptive embedding
adaptive_params = sum([r['vocab_size'] * r['recommended_dim'] for r in results])

print(f"Current (fixed 16-dim): {current_params:,} parameters")
print(f"Adaptive embedding:     {adaptive_params:,} parameters")
print(f"Reduction:              {current_params - adaptive_params:,} ({(1 - adaptive_params/current_params)*100:.1f}%)")

