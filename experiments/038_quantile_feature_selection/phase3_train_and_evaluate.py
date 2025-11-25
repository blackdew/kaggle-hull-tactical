#!/usr/bin/env python3
"""
EXP-038 Phase 3: Train and Evaluate
Goal: Train EXP-020 strategy using the new Quantile-Optimized Feature Set.
Compare performance with original EXP-020 (CV Sharpe 0.637).
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import os

print("="*80)
print("EXP-038 Phase 3: Train and Evaluate")
print("="*80)

# 1. Load Data
print("\n[1] Loading data...")
train = pd.read_csv("data/train.csv")
y_excess = train['market_forward_excess_returns'].values
fwd_returns = train['forward_returns'].values
risk_free = train['risk_free_rate'].values

# 2. Load Selected Features
print("\n[2] Loading selected features...")
top_30_df = pd.read_csv("experiments/038_quantile_feature_selection/results/top_30_quantile_features.csv")
top_30 = top_30_df['feature'].tolist()
print(f"   Selected Features ({len(top_30)}):")
print(f"   {top_30}")

# 3. Create Feature Matrix
print("\n[3] Creating feature matrix...")
# Load base features
top_20 = [
    'M4', 'V13', 'V7', 'P8', 'S2', 'I2', 'E19', 'S5', 'P5', 'P7',
    'M2', 'V9', 'M3', 'P12', 'P10', 'V10', 'E12', 'P11', 'M12', 'S8'
]
X_base = train[top_20].fillna(train[top_20].median()).replace([np.inf, -np.inf], np.nan).fillna(0)

# Generate interactions (same logic)
interactions = []
feature_names = top_20.copy()
top_10 = top_20[:10]
eps = 1e-8

for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i+1:]:
        interactions.append(X_base[feat1] * X_base[feat2])
        feature_names.append(f'{feat1}*{feat2}')

for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i+1:]:
        interactions.append(X_base[feat1] / (X_base[feat2].abs() + eps))
        feature_names.append(f'{feat1}/{feat2}')

top_5 = top_20[:5]
for feat in top_5:
    interactions.append(X_base[feat] ** 2)
    feature_names.append(f'{feat}²')
    interactions.append(X_base[feat] ** 3)
    feature_names.append(f'{feat}³')

X_all = pd.concat([X_base] + [pd.Series(feat, name=name) for feat, name in zip(interactions, feature_names[len(top_20):])], axis=1)
X_all.columns = feature_names
X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)

# Select Top 30
X = X_all[top_30].copy()
print(f"   Feature Matrix: {X.shape}")

# 4. Train and Evaluate (CV)
print("\n[4] Evaluating Strategy (5-fold CV)...")

def calculate_sharpe(positions, fwd_returns, risk_free):
    positions = np.clip(positions, 0.0, 2.0)
    strategy_returns = risk_free * (1.0 - positions) + fwd_returns * positions
    excess_returns = strategy_returns - risk_free
    if np.std(strategy_returns) > 0:
        sharpe = (np.mean(excess_returns) / np.std(strategy_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0
    return sharpe

sharpes = []
tscv = TimeSeriesSplit(n_splits=5)
K = 250  # Fixed K from EXP-016/020

for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
    print(f"\n   Fold {fold_idx}/5...")
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr = y_excess[tr_idx]
    
    # Scale
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    
    # Train q10, q50, q90
    models = {}
    for alpha, name in [(0.1, 'q10'), (0.5, 'q50'), (0.9, 'q90')]:
        model = XGBRegressor(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.025,
            subsample=1.0,
            colsample_bytree=0.6,
            reg_lambda=0.5,
            objective='reg:quantileerror',
            quantile_alpha=alpha,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_tr_scaled, y_tr)
        models[name] = model
    
    # Predict
    q10_pred = models['q10'].predict(X_va_scaled)
    q50_pred = models['q50'].predict(X_va_scaled)
    q90_pred = models['q90'].predict(X_va_scaled)
    
    # Strategy: CI-based (scaled x5) - Best from EXP-020
    ci_width = q90_pred - q10_pred
    confidence = 1.0 / (np.abs(ci_width) + 0.001)
    confidence = np.clip(confidence, 0.5, 5.0)
    positions = 1.0 + q50_pred * K * confidence
    
    # Evaluate
    va_fwd = fwd_returns[va_idx]
    va_rf = risk_free[va_idx]
    sharpe = calculate_sharpe(positions, va_fwd, va_rf)
    sharpes.append(sharpe)
    
    print(f"      Sharpe: {sharpe:.4f}")

# 5. Results
avg_sharpe = np.mean(sharpes)
std_sharpe = np.std(sharpes)

print("\n" + "="*80)
print("RESULTS:")
print("="*80)
print(f"EXP-038 CV Sharpe: {avg_sharpe:.4f} ± {std_sharpe:.4f}")
print(f"EXP-020 CV Sharpe: 0.6368 (Baseline)")

improvement = ((avg_sharpe - 0.6368) / 0.6368) * 100
print(f"Improvement: {improvement:+.2f}%")

if avg_sharpe > 0.6368:
    print("✅ SUCCESS: New feature set improved performance!")
else:
    print("❌ FAILURE: No improvement.")

# Save results
results_df = pd.DataFrame({
    'fold': range(1, 6),
    'sharpe': sharpes
})
results_df.to_csv("experiments/038_quantile_feature_selection/results/cv_results.csv", index=False)
print("\nResults saved to experiments/038_quantile_feature_selection/results/")
