#!/usr/bin/env python3
"""
EXP-016 Phase 2: Feature Engineering (Interaction Features)

목표: Top 20 features로 1-row 계산 가능한 interaction features 생성
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EXP-016 Phase 2: Feature Engineering")
print("="*80)
print()

# Load data
print("[1] Loading data...")
train = pd.read_csv("data/train.csv")
y = train['market_forward_excess_returns'].copy()

# Load Top 20 features
top_20_df = pd.read_csv("experiments/016/results/top_20_features.csv")
top_20 = top_20_df['feature'].tolist()
print(f"Top 20 features loaded: {len(top_20)}")
print()

# Create interaction features
print("[2] Creating interaction features...")
X_base = train[top_20].fillna(train[top_20].median()).replace([np.inf, -np.inf], np.nan).fillna(0)

interactions = []
feature_names = top_20.copy()

# Select top 10 for interactions (to avoid explosion)
top_10 = top_20[:10]
print(f"Using top 10 features for interactions: {top_10}")
print()

# Multiplication
print("  Creating multiplication features...")
for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i+1:]:
        new_feat = X_base[feat1] * X_base[feat2]
        interactions.append(new_feat)
        feature_names.append(f'{feat1}*{feat2}')

# Division (with epsilon to avoid division by zero)
print("  Creating division features...")
eps = 1e-8
for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i+1:]:
        new_feat = X_base[feat1] / (X_base[feat2].abs() + eps)
        interactions.append(new_feat)
        feature_names.append(f'{feat1}/{feat2}')

# Polynomial (square and cube for top 5)
print("  Creating polynomial features...")
top_5 = top_20[:5]
for feat in top_5:
    interactions.append(X_base[feat] ** 2)
    feature_names.append(f'{feat}²')
    interactions.append(X_base[feat] ** 3)
    feature_names.append(f'{feat}³')

print(f"Total interaction features created: {len(interactions)}")
print(f"Total features: {len(top_20)} (base) + {len(interactions)} (interactions) = {len(top_20) + len(interactions)}")
print()

# Combine
X_all = pd.concat([X_base] + [pd.Series(feat, name=name) for feat, name in zip(interactions, feature_names[len(top_20):])], axis=1)
X_all.columns = feature_names

# Handle inf/nan again
X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)
print(f"Combined feature matrix: {X_all.shape}")
print()

# Feature selection using XGBoost
print("[3] Feature importance analysis...")
xgb = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

xgb.fit(X_scaled, y)

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 30 features by XGBoost importance:")
print(importance_df.head(30).to_string(index=False))
print()

# Select top features
top_50 = importance_df.head(50)['feature'].tolist()
top_30 = importance_df.head(30)['feature'].tolist()

print(f"[4] Evaluating different feature sets...")

# Test with XGBoost
def evaluate_features(X, y, name):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    xgb = XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        n_jobs=-1
    )

    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(xgb, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    sharpe_approx = -np.mean(scores) ** -0.5  # Rough approximation

    print(f"{name:30s}: MSE = {-scores.mean():.6f} ± {scores.std():.6f}")
    return -scores.mean()

# Baseline
mse_top20 = evaluate_features(X_all[top_20], y, "Top 20 (base)")
mse_top30 = evaluate_features(X_all[top_30], y, "Top 30 (with interactions)")
mse_top50 = evaluate_features(X_all[top_50], y, "Top 50 (with interactions)")
mse_all = evaluate_features(X_all, y, f"All {len(feature_names)} features")
print()

# Save results
print("[5] Saving results...")
importance_df.to_csv('experiments/016/results/feature_importance_with_interactions.csv', index=False)
pd.DataFrame({'feature': top_30}).to_csv('experiments/016/results/top_30_with_interactions.csv', index=False)
pd.DataFrame({'feature': top_50}).to_csv('experiments/016/results/top_50_with_interactions.csv', index=False)

# Save summary
summary = pd.DataFrame({
    'feature_set': ['Top 20 (base)', 'Top 30 (interactions)', 'Top 50 (interactions)', f'All {len(feature_names)}'],
    'n_features': [20, 30, 50, len(feature_names)],
    'mse': [mse_top20, mse_top30, mse_top50, mse_all]
})
summary.to_csv('experiments/016/results/phase2_summary.csv', index=False)

print("Results saved to experiments/016/results/")
print()

print("="*80)
print("Phase 2 Complete!")
print("="*80)
print()
print("Best feature set:")
best_idx = summary['mse'].idxmin()
best = summary.iloc[best_idx]
print(f"  {best['feature_set']}: MSE = {best['mse']:.6f}")
print()
print("Next: Phase 3 - Model Training & Hyperparameter Tuning")
