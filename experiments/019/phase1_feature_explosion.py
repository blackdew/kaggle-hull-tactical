#!/usr/bin/env python3
"""
EXP-019 Phase 1: Feature Explosion for 10+ Public Score

Goal: Generate 150+ features (3-way, 4-way, meta) and select top 100
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from itertools import combinations

print("="*80)
print("EXP-019 Phase 1: Feature Explosion (Target: 10+ Public Score)")
print("="*80)
print()

# Load data
print("[1] Loading data...")
train = pd.read_csv("data/train.csv")
print(f"Train data shape: {train.shape}")

# Base features
top_20_base = [
    'M4', 'V13', 'V7', 'P8', 'S2', 'I2', 'E19', 'S5', 'P5', 'P7',
    'M2', 'V9', 'M3', 'P12', 'P10', 'V10', 'E12', 'P11', 'M12', 'S8'
]

X_base = train[top_20_base].fillna(train[top_20_base].median()).replace([np.inf, -np.inf], 0)
y = train['market_forward_excess_returns'].to_numpy()

print(f"Base features: {X_base.shape}")
print()

# Feature storage
all_features = {}
eps = 1e-8

print("[2] Creating interaction features...")

# Get top 10 and top 5 for interactions
top_10 = top_20_base[:10]
top_5 = top_20_base[:5]

# 2-way interactions (baseline from EXP-016)
print("  - 2-way interactions...")
count_2way = 0

# Multiplication
for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i+1:]:
        name = f'{feat1}*{feat2}'
        all_features[name] = X_base[feat1] * X_base[feat2]
        count_2way += 1

# Division
for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i+1:]:
        name = f'{feat1}/{feat2}'
        all_features[name] = X_base[feat1] / (X_base[feat2].abs() + eps)
        count_2way += 1

# Polynomial
for feat in top_5:
    all_features[f'{feat}²'] = X_base[feat] ** 2
    all_features[f'{feat}³'] = X_base[feat] ** 3
    all_features[f'{feat}⁴'] = X_base[feat] ** 4
    count_2way += 3

print(f"    Created {count_2way} 2-way features")

# 3-way interactions (NEW - aggressive)
print("  - 3-way interactions...")
count_3way = 0

# Top 8 features for 3-way (C(8,3) = 56)
top_8 = top_20_base[:8]
for feat1, feat2, feat3 in combinations(top_8, 3):
    # Multiplication
    name = f'{feat1}*{feat2}*{feat3}'
    all_features[name] = X_base[feat1] * X_base[feat2] * X_base[feat3]
    count_3way += 1

    # Mixed operations (select a few high-value combinations)
    if feat1 in top_5 and feat2 in top_5:
        # (feat1*feat2)/feat3
        name = f'({feat1}*{feat2})/{feat3}'
        all_features[name] = (X_base[feat1] * X_base[feat2]) / (X_base[feat3].abs() + eps)
        count_3way += 1

print(f"    Created {count_3way} 3-way features")

# 4-way interactions (NEW - most aggressive)
print("  - 4-way interactions...")
count_4way = 0

# Top 6 features for 4-way (C(6,4) = 15)
top_6 = top_20_base[:6]
for feat1, feat2, feat3, feat4 in combinations(top_6, 4):
    name = f'{feat1}*{feat2}*{feat3}*{feat4}'
    all_features[name] = X_base[feat1] * X_base[feat2] * X_base[feat3] * X_base[feat4]
    count_4way += 1

print(f"    Created {count_4way} 4-way features")

# Meta-features (NEW)
print("  - Meta-features (category statistics)...")
count_meta = 0

# Category groups
M_features = ['M4', 'M2', 'M3', 'M12']
V_features = ['V13', 'V7', 'V9', 'V10']
P_features = ['P8', 'P7', 'P5', 'P12', 'P10', 'P11']
S_features = ['S2', 'S5', 'S8']
I_features = ['I2']
E_features = ['E19', 'E12']

# Statistics for each category
for cat_name, cat_feats in [('M', M_features), ('V', V_features), ('P', P_features), ('S', S_features)]:
    cat_data = X_base[cat_feats]

    # Mean, Std, Min, Max
    all_features[f'{cat_name}_mean'] = cat_data.mean(axis=1)
    all_features[f'{cat_name}_std'] = cat_data.std(axis=1)
    all_features[f'{cat_name}_min'] = cat_data.min(axis=1)
    all_features[f'{cat_name}_max'] = cat_data.max(axis=1)
    all_features[f'{cat_name}_range'] = all_features[f'{cat_name}_max'] - all_features[f'{cat_name}_min']

    # Skewness and Kurtosis
    all_features[f'{cat_name}_skew'] = cat_data.apply(lambda x: skew(x, nan_policy='omit'), axis=1)
    all_features[f'{cat_name}_kurt'] = cat_data.apply(lambda x: kurtosis(x, nan_policy='omit'), axis=1)

    # Coefficient of Variation
    all_features[f'{cat_name}_cv'] = all_features[f'{cat_name}_std'] / (all_features[f'{cat_name}_mean'].abs() + eps)

    count_meta += 8

print(f"    Created {count_meta} meta-features")

# Cross-category features (NEW)
print("  - Cross-category features...")
count_cross = 0

# Market strength indicators
all_features['market_strength'] = all_features['M_mean'] / (all_features['V_mean'] + eps)
all_features['risk_adjusted_return'] = all_features['P_mean'] / (all_features['V_mean'] + eps)
all_features['sentiment_to_volatility'] = all_features['S_mean'] / (all_features['V_mean'] + eps)
all_features['market_to_price'] = all_features['M_mean'] / (all_features['P_mean'] + eps)

# Composite indicators
all_features['total_volatility'] = np.sqrt(all_features['V_mean']**2 + all_features['M_std']**2 + eps)
all_features['market_turbulence'] = all_features['M_std'] * all_features['V_std']
all_features['price_momentum'] = all_features['P_max'] - all_features['P_min']

# Normalized composites
all_features['normalized_market'] = all_features['M_mean'] / (all_features['total_volatility'] + eps)
all_features['normalized_sentiment'] = all_features['S_mean'] / (all_features['total_volatility'] + eps)

count_cross += 9

print(f"    Created {count_cross} cross-category features")

# Volatility features (from EXP-018)
print("  - Volatility features...")
count_vol = 0

all_features['vol_mean_V'] = (abs(X_base['V13']) + abs(X_base['V7']) + abs(X_base['V9'])) / 3
all_features['vol_composite_V_all'] = np.sqrt(X_base['V13']**2 + X_base['V7']**2 + X_base['V9']**2 + X_base['V10']**2 + eps)
all_features['vol_composite_V'] = np.sqrt(X_base['V13']**2 + X_base['V7']**2 + X_base['V9']**2 + eps)
all_features['cross_vol_MV'] = np.sqrt(X_base['M4']**2 + X_base['V13']**2 + eps)
all_features['sentiment_dispersion'] = abs(X_base['S2'] - X_base['S5'])
all_features['market_vol_composite'] = np.sqrt(X_base['M4']**2 + X_base['M2']**2 + X_base['M3']**2 + eps)
all_features['price_vol'] = np.sqrt((X_base['P8'] - X_base['P7'])**2 + (X_base['P7'] - X_base['P5'])**2 + eps)

count_vol += 7

print(f"    Created {count_vol} volatility features")

# Advanced ratio features
print("  - Advanced ratio features...")
count_ratio = 0

# Top feature ratios
for feat1 in top_5:
    for feat2 in top_5:
        if feat1 != feat2:
            name = f'ratio_{feat1}_to_{feat2}'
            all_features[name] = X_base[feat1] / (X_base[feat2].abs() + eps)
            count_ratio += 1

print(f"    Created {count_ratio} ratio features")

# Convert to DataFrame
print()
print("[3] Converting to DataFrame...")
X_all = pd.DataFrame(all_features)
X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Total features created: {X_all.shape[1]}")
print()

# Feature selection using RandomForest
print("[4] Feature selection with RandomForest...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1, max_features='sqrt')
print("  Training RandomForest...")
rf.fit(X_scaled, y)

importances = rf.feature_importances_
feature_names = list(X_all.columns)

# Calculate correlation
print("  Calculating correlations...")
correlations = []
for col in X_all.columns:
    corr = np.corrcoef(X_all[col], y)[0, 1]
    correlations.append(abs(corr) if not np.isnan(corr) else 0.0)

# Combine scores
results = pd.DataFrame({
    'feature': feature_names,
    'importance': importances,
    'abs_correlation': correlations,
    'combined_score': importances * 0.7 + np.array(correlations) * 0.3
})

results = results.sort_values('combined_score', ascending=False).reset_index(drop=True)

print()
print("Top 30 Features:")
print(results.head(30).to_string(index=False))
print()

# Save results
print("[5] Saving results...")
results.to_csv("experiments/019/results/all_features_ranking.csv", index=False)

# Save top 100 features
top_100 = results.head(100)
top_100.to_csv("experiments/019/results/top_100_features.csv", index=False)

print(f"Saved to experiments/019/results/")
print()

# Summary
print("="*80)
print("Feature Creation Summary")
print("="*80)
print(f"2-way interactions:    {count_2way}")
print(f"3-way interactions:    {count_3way}")
print(f"4-way interactions:    {count_4way}")
print(f"Meta-features:         {count_meta}")
print(f"Cross-category:        {count_cross}")
print(f"Volatility features:   {count_vol}")
print(f"Ratio features:        {count_ratio}")
print(f"{'='*40}")
print(f"TOTAL:                 {X_all.shape[1]}")
print()
print(f"Selected TOP 100 features for modeling")
print(f"Average importance: {top_100['importance'].mean():.6f}")
print(f"Average correlation: {top_100['abs_correlation'].mean():.6f}")
print()

# Feature type distribution in top 100
print("Top 100 Feature Type Distribution:")
for feature_type in ['*', '/', '²', '³', '⁴', '_mean', '_std', 'vol_', 'ratio_']:
    count = sum(1 for f in top_100['feature'] if feature_type in f)
    print(f"  {feature_type:12s}: {count}")

print()
print("="*80)
print("Phase 1 Complete!")
print("="*80)
