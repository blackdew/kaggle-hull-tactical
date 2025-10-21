#!/usr/bin/env python3
"""
EXP-016 Phase 1: 원본 Features 분석

목표: 1 row에서 계산 가능한 원본 features 선택
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("="*80)
print("EXP-016 Phase 1: 원본 Features 분석")
print("="*80)
print()

# Load data
print("[1] Loading data...")
train = pd.read_csv("data/train.csv")
print(f"Train shape: {train.shape}")
print()

# Identify original features
print("[2] Identifying original features...")
exclude_cols = [
    'date_id', 'forward_returns', 'risk_free_rate',
    'market_forward_excess_returns', 'is_scored',
    'lagged_forward_returns', 'lagged_risk_free_rate',
    'lagged_market_forward_excess_returns'
]

all_features = [c for c in train.columns if c not in exclude_cols]
print(f"Total features: {len(all_features)}")

# Filter to original features only (no lag/rolling/ema)
original_features = []
for feat in all_features:
    # Skip derived features
    if any(pattern in feat.lower() for pattern in ['_lag', '_rolling', '_ema', '_vol_', '_trend', '_return', '_rank', '_zscore', '_quantile', '_regime']):
        continue
    original_features.append(feat)

print(f"Original features (1-row calculable): {len(original_features)}")
print()

# Group by prefix
print("[3] Feature categories:")
prefixes = {}
for feat in original_features:
    prefix = feat.split('_')[0] if '_' in feat else feat[0]
    if prefix not in prefixes:
        prefixes[prefix] = []
    prefixes[prefix].append(feat)

for prefix in sorted(prefixes.keys()):
    print(f"  {prefix}: {len(prefixes[prefix])} features")
print()

# Prepare data
print("[4] Preparing data for importance analysis...")
X = train[original_features].copy()
y = train['market_forward_excess_returns'].copy()

# Handle missing/inf values
X = X.fillna(X.median()).replace([np.inf, -np.inf], np.nan).fillna(0)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print()

# Feature importance using RandomForest
print("[5] Computing feature importance (RandomForest)...")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X, y)

importance_df = pd.DataFrame({
    'feature': original_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 30 features by importance:")
print(importance_df.head(30).to_string(index=False))
print()

# Correlation with target
print("[6] Computing correlation with target...")
correlations = X.corrwith(y).abs().sort_values(ascending=False)
correlation_df = pd.DataFrame({
    'feature': correlations.index,
    'abs_correlation': correlations.values
})

print("Top 30 features by correlation:")
print(correlation_df.head(30).to_string(index=False))
print()

# Combined ranking
print("[7] Combined ranking (importance + correlation)...")
importance_rank = importance_df.set_index('feature')['importance'].rank(ascending=False)
corr_rank = correlations.rank(ascending=False)

combined_rank = (importance_rank + corr_rank) / 2
combined_df = pd.DataFrame({
    'feature': combined_rank.index,
    'combined_rank': combined_rank.values,
    'importance': importance_df.set_index('feature')['importance'],
    'correlation': correlations
}).sort_values('combined_rank')

print("Top 30 features (combined ranking):")
print(combined_df.head(30).to_string(index=False))
print()

# Select Top 20
top_20_features = combined_df.head(20)['feature'].tolist()
print(f"[8] Selected Top 20 features:")
for i, feat in enumerate(top_20_features, 1):
    imp = importance_df[importance_df['feature'] == feat]['importance'].values[0]
    corr = correlations[feat]
    print(f"  {i:2d}. {feat:20s}  (imp: {imp:.4f}, corr: {corr:.4f})")
print()

# Baseline test
print("[9] Baseline test with Top 20...")
X_top20 = X[top_20_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_top20)

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(rf, X_scaled, y, cv=tscv, scoring='r2', n_jobs=-1)

print(f"5-fold CV R²: {scores.mean():.4f} ± {scores.std():.4f}")
print(f"Scores: {scores}")
print()

# Save results
import os
os.makedirs('experiments/016/results', exist_ok=True)

combined_df.to_csv('experiments/016/results/feature_ranking.csv', index=False)
pd.DataFrame({'feature': top_20_features}).to_csv('experiments/016/results/top_20_features.csv', index=False)

print("="*80)
print("Phase 1 Complete!")
print("="*80)
print()
print("Next: Phase 2 - Feature Engineering (interactions)")
