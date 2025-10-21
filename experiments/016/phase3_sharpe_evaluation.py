#!/usr/bin/env python3
"""
EXP-016 Phase 3: Sharpe Ratio Evaluation

목표: 실제 Sharpe ratio 계산 및 최적 K 값 찾기
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

def calculate_sharpe(y_pred, y_true, fwd_returns, risk_free, k=200.0):
    """Calculate Sharpe ratio from predictions"""
    # Convert excess return predictions to positions
    positions = np.clip(1.0 + y_pred * k, 0.0, 2.0)

    # Calculate strategy returns
    strategy_returns = risk_free * (1.0 - positions) + fwd_returns * positions
    excess_returns = strategy_returns - risk_free

    # Calculate Sharpe
    if np.std(strategy_returns) > 0:
        sharpe = (np.mean(excess_returns) / np.std(strategy_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    return sharpe, positions.mean(), positions.std()

print("="*80)
print("EXP-016 Phase 3: Sharpe Ratio Evaluation")
print("="*80)
print()

# Load data
print("[1] Loading data...")
train = pd.read_csv("data/train.csv")
y = train['market_forward_excess_returns'].copy()
fwd_returns = train['forward_returns'].copy()
risk_free = train['risk_free_rate'].copy()

# Load features
top_30_df = pd.read_csv("experiments/016/results/top_30_with_interactions.csv")
top_30 = top_30_df['feature'].tolist()

# Recreate features
print("[2] Recreating features...")
top_20 = pd.read_csv("experiments/016/results/top_20_features.csv")['feature'].tolist()
X_base = train[top_20].fillna(train[top_20].median()).replace([np.inf, -np.inf], np.nan).fillna(0)

# Create interactions
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

# Select Top 30
X = X_all[top_30].copy()
print(f"Features: {X.shape}")
print()

# Test different K values
print("[3] Testing different K values...")
K_values = [50, 100, 150, 200, 250, 300]

best_sharpe = -np.inf
best_k = None

for k in K_values:
    sharpes = []

    tscv = TimeSeriesSplit(n_splits=5)

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]

        # Scale
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)

        # Train
        xgb = XGBRegressor(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.025,
            subsample=1.0,
            colsample_bytree=0.6,
            reg_lambda=0.5,
            random_state=42,
            n_jobs=-1
        )
        xgb.fit(X_tr_scaled, y_tr)

        # Predict
        y_pred = xgb.predict(X_va_scaled)

        # Evaluate
        va_fwd = fwd_returns.iloc[va_idx].values
        va_rf = risk_free.iloc[va_idx].values

        sharpe, pos_mean, pos_std = calculate_sharpe(y_pred, y_va.values, va_fwd, va_rf, k=k)
        sharpes.append(sharpe)

    avg_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes)

    print(f"K={k:3d}: Sharpe = {avg_sharpe:.4f} ± {std_sharpe:.4f}  (folds: {[f'{s:.3f}' for s in sharpes]})")

    if avg_sharpe > best_sharpe:
        best_sharpe = avg_sharpe
        best_k = k

print()
print(f"[4] Best K = {best_k}, Sharpe = {best_sharpe:.4f}")
print()

# Final evaluation with best K
print(f"[5] Final 5-fold CV with K={best_k}...")
sharpes = []
positions_stats = []

tscv = TimeSeriesSplit(n_splits=5)

for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
    X_tr = X.iloc[tr_idx]
    y_tr = y.iloc[tr_idx]
    X_va = X.iloc[va_idx]
    y_va = y.iloc[va_idx]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)

    xgb = XGBRegressor(
        n_estimators=150,
        max_depth=7,
        learning_rate=0.025,
        subsample=1.0,
        colsample_bytree=0.6,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_tr_scaled, y_tr)

    y_pred = xgb.predict(X_va_scaled)

    va_fwd = fwd_returns.iloc[va_idx].values
    va_rf = risk_free.iloc[va_idx].values

    sharpe, pos_mean, pos_std = calculate_sharpe(y_pred, y_va.values, va_fwd, va_rf, k=best_k)
    sharpes.append(sharpe)
    positions_stats.append((pos_mean, pos_std))

    print(f"  Fold {fold_idx}: Sharpe = {sharpe:.4f}, Position mean = {pos_mean:.4f}, std = {pos_std:.4f}")

avg_sharpe = np.mean(sharpes)
std_sharpe = np.std(sharpes)

print()
print("="*80)
print(f"FINAL RESULT: Sharpe = {avg_sharpe:.4f} ± {std_sharpe:.4f}")
print("="*80)
print()

# Save results
results_df = pd.DataFrame({
    'fold': range(1, 6),
    'sharpe': sharpes,
    'position_mean': [ps[0] for ps in positions_stats],
    'position_std': [ps[1] for ps in positions_stats]
})
results_df.to_csv('experiments/016/results/final_cv_results.csv', index=False)

config_df = pd.DataFrame({
    'parameter': ['best_k', 'avg_sharpe', 'std_sharpe', 'n_features'],
    'value': [best_k, avg_sharpe, std_sharpe, len(top_30)]
})
config_df.to_csv('experiments/016/results/final_config.csv', index=False)

print("Results saved!")
print()
print("Next: InferenceServer implementation")
