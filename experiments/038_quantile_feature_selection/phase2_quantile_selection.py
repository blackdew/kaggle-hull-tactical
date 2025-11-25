#!/usr/bin/env python3
"""
EXP-038 Phase 2: Quantile Feature Selection
Goal: Select features that are most important for Quantile Regression (q10, q90).
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import os

print("="*80)
print("EXP-038 Phase 2: Quantile Feature Selection")
print("="*80)

# 1. Load Data
print("\n[1] Loading data...")
train = pd.read_csv("data/train.csv")
y = train['market_forward_excess_returns'].values

# 2. Load Candidate Features
print("\n[2] Loading candidate features...")
candidate_df = pd.read_csv("experiments/038_quantile_feature_selection/results/candidate_features.csv")
candidates = candidate_df['feature'].tolist()
print(f"   Candidates: {len(candidates)}")

# 3. Create Feature Matrix
print("\n[3] Creating feature matrix...")
# Load base features first
top_20 = [
    'M4', 'V13', 'V7', 'P8', 'S2', 'I2', 'E19', 'S5', 'P5', 'P7',
    'M2', 'V9', 'M3', 'P12', 'P10', 'V10', 'E12', 'P11', 'M12', 'S8'
]
X_base = train[top_20].fillna(train[top_20].median()).replace([np.inf, -np.inf], np.nan).fillna(0)

# Generate interactions on the fly (same logic as Phase 1)
interactions = []
feature_names = top_20.copy()
top_10 = top_20[:10]
eps = 1e-8

# Multiplication
for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i+1:]:
        interactions.append(X_base[feat1] * X_base[feat2])
        feature_names.append(f'{feat1}*{feat2}')

# Division
for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i+1:]:
        interactions.append(X_base[feat1] / (X_base[feat2].abs() + eps))
        feature_names.append(f'{feat1}/{feat2}')

# Polynomial
top_5 = top_20[:5]
for feat in top_5:
    interactions.append(X_base[feat] ** 2)
    feature_names.append(f'{feat}²')
    interactions.append(X_base[feat] ** 3)
    feature_names.append(f'{feat}³')

X_all = pd.concat([X_base] + [pd.Series(feat, name=name) for feat, name in zip(interactions, feature_names[len(top_20):])], axis=1)
X_all.columns = feature_names
X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"   Feature Matrix: {X_all.shape}")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# 4. Quantile Feature Importance
print("\n[4] Calculating Quantile Feature Importance...")

def get_importance(alpha, name):
    print(f"   Training {name} (alpha={alpha})...")
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='reg:quantileerror',
        quantile_alpha=alpha,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_scaled, y)
    return model.feature_importances_

# q10 importance
imp_q10 = get_importance(0.1, "q10")

# q90 importance
imp_q90 = get_importance(0.9, "q90")

# q50 importance (for reference)
imp_q50 = get_importance(0.5, "q50")

# 5. Aggregate Importance
print("\n[5] Aggregating Importance...")
importance_df = pd.DataFrame({
    'feature': feature_names,
    'imp_q10': imp_q10,
    'imp_q90': imp_q90,
    'imp_q50': imp_q50
})

# Combined score: Average of q10 and q90 (focus on tails)
importance_df['imp_tail'] = (importance_df['imp_q10'] + importance_df['imp_q90']) / 2
importance_df = importance_df.sort_values('imp_tail', ascending=False)

print("\nTop 10 Quantile Features:")
print(importance_df[['feature', 'imp_tail', 'imp_q10', 'imp_q90']].head(10))

# 6. Select Top 30
top_30 = importance_df.head(30)['feature'].tolist()
print(f"\n[6] Selected Top 30 Features: {top_30}")

# 7. Save Results
output_path = "experiments/038_quantile_feature_selection/results/top_30_quantile_features.csv"
pd.DataFrame({'feature': top_30}).to_csv(output_path, index=False)
importance_df.to_csv("experiments/038_quantile_feature_selection/results/quantile_importance.csv", index=False)

print(f"\nResults saved to: {output_path}")

print("\n" + "="*80)
print("Phase 2 Complete!")
print("="*80)
