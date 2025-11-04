#!/usr/bin/env python3
"""
EXP-020 Phase 1: Volatility Prediction Model

목표:
- EXP-016의 Top 30 features를 사용하여 realized volatility 예측
- Volatility-aware position sizing의 기반 구축

Target: Forward 20-day realized volatility (std of future returns)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 80)
print("EXP-020 Phase 1: Volatility Prediction Model")
print("=" * 80)
print()

# Load data
print("[1] Loading data...")
train = pd.read_csv("data/train.csv")

# Create realized volatility target (forward 20-day rolling std)
print("[2] Creating realized volatility target...")
print("   Forward 20-day realized volatility (std of forward returns)")

# Calculate forward returns
returns = train['forward_returns'].values
n_samples = len(returns)

# Forward rolling window volatility
vol_window = 20
vol_target = np.zeros(n_samples)

for i in range(n_samples - vol_window):
    future_returns = returns[i : i + vol_window]
    vol_target[i] = np.std(future_returns)

# For last window samples, use the last valid volatility
vol_target[n_samples - vol_window :] = vol_target[n_samples - vol_window - 1]

print(f"   Volatility target: mean={np.mean(vol_target):.6f}, std={np.std(vol_target):.6f}")
print()

# Load EXP-016 features
print("[3] Loading EXP-016 Top 30 features...")
top_30_df = pd.read_csv("experiments/016/results/top_30_with_interactions.csv")
top_30 = top_30_df['feature'].tolist()
print(f"   Features: {len(top_30)}")

# Recreate features (same as EXP-016)
print("[4] Recreating features...")
top_20 = pd.read_csv("experiments/016/results/top_20_features.csv")['feature'].tolist()
X_base = train[top_20].fillna(train[top_20].median()).replace([np.inf, -np.inf], np.nan).fillna(0)

# Create interactions
interactions = []
feature_names = top_20.copy()

top_10 = top_20[:10]
eps = 1e-8

# Multiplication
for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i + 1 :]:
        interactions.append(X_base[feat1] * X_base[feat2])
        feature_names.append(f'{feat1}*{feat2}')

# Division
for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i + 1 :]:
        interactions.append(X_base[feat1] / (X_base[feat2].abs() + eps))
        feature_names.append(f'{feat1}/{feat2}')

# Polynomial
top_5 = top_20[:5]
for feat in top_5:
    interactions.append(X_base[feat] ** 2)
    feature_names.append(f'{feat}²')
    interactions.append(X_base[feat] ** 3)
    feature_names.append(f'{feat}³')

X_all = pd.concat(
    [X_base] + [pd.Series(feat, name=name) for feat, name in zip(interactions, feature_names[len(top_20) :])],
    axis=1
)
X_all.columns = feature_names
X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)

# Select Top 30
X = X_all[top_30].copy()
y_vol = pd.Series(vol_target, index=X.index)

print(f"   X shape: {X.shape}")
print(f"   y_vol shape: {y_vol.shape}")
print()

# Train volatility prediction model
print("[5] Training volatility prediction model (5-fold CV)...")
tscv = TimeSeriesSplit(n_splits=5)

r2_scores = []
mae_scores = []
rmse_scores = []

for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
    X_tr = X.iloc[tr_idx]
    y_tr = y_vol.iloc[tr_idx]
    X_va = X.iloc[va_idx]
    y_va = y_vol.iloc[va_idx]

    # Scale
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)

    # Train XGBoost (same hyperparameters as EXP-016)
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
    r2 = r2_score(y_va, y_pred)
    mae = mean_absolute_error(y_va, y_pred)
    rmse = np.sqrt(mean_squared_error(y_va, y_pred))

    r2_scores.append(r2)
    mae_scores.append(mae)
    rmse_scores.append(rmse)

    print(f"  Fold {fold_idx}: R²={r2:.4f}, MAE={mae:.6f}, RMSE={rmse:.6f}")

avg_r2 = np.mean(r2_scores)
avg_mae = np.mean(mae_scores)
avg_rmse = np.mean(rmse_scores)

print()
print("=" * 80)
print(f"FINAL RESULT (Volatility Prediction):")
print(f"  R² = {avg_r2:.4f} ± {np.std(r2_scores):.4f}")
print(f"  MAE = {avg_mae:.6f} ± {np.std(mae_scores):.6f}")
print(f"  RMSE = {avg_rmse:.6f} ± {np.std(rmse_scores):.6f}")
print("=" * 80)
print()

# Save results
results_df = pd.DataFrame({
    'fold': range(1, 6),
    'r2': r2_scores,
    'mae': mae_scores,
    'rmse': rmse_scores
})
results_df.to_csv('experiments/020/results/phase1_volatility_prediction.csv', index=False)

config_df = pd.DataFrame({
    'metric': ['avg_r2', 'avg_mae', 'avg_rmse', 'n_features'],
    'value': [avg_r2, avg_mae, avg_rmse, len(top_30)]
})
config_df.to_csv('experiments/020/results/phase1_config.csv', index=False)

print("Results saved to experiments/020/results/")
print()
print("Next: Phase 2 - Volatility-Scaled Strategy")
