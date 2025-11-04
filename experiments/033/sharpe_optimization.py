"""
EXP-033: Direct Sharpe Ratio Optimization

Current problem: Train with MSE, evaluate with Sharpe → objective mismatch
Solution: Custom loss function that directly optimizes Sharpe ratio

Challenge: Sharpe ratio is not differentiable in standard form
Approach: Use differentiable approximation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def calculate_sharpe(positions, fwd_returns, risk_free):
    """Calculate Sharpe ratio"""
    positions = np.clip(positions, 0.0, 2.0)
    strategy_returns = risk_free * (1.0 - positions) + fwd_returns * positions
    excess_returns = strategy_returns - risk_free

    if np.std(strategy_returns) > 0:
        sharpe = (np.mean(excess_returns) / np.std(strategy_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    return sharpe

def sharpe_loss(y_pred, dtrain):
    """
    Custom XGBoost objective: Negative Sharpe ratio

    We want to maximize Sharpe, so minimize negative Sharpe.

    Challenge: Sharpe = mean(returns) / std(returns)
    This is not directly differentiable for gradient boosting.

    Approximation: Use mean squared prediction error weighted by return sign
    Better predictions → better positions → higher Sharpe
    """
    y_true = dtrain.get_label()

    # Gradient: derivative of loss w.r.t. prediction
    # For now, use MSE-like gradient but we'll try different approaches
    grad = 2 * (y_pred - y_true)

    # Hessian: second derivative
    hess = 2 * np.ones_like(y_pred)

    return grad, hess

def sharpe_weighted_loss(y_pred, dtrain, K=250):
    """
    Custom loss that weights errors by their impact on Sharpe

    Intuition:
    - Errors that lead to wrong position direction are more costly
    - Errors on high-magnitude returns matter more
    """
    y_true = dtrain.get_label()

    # Weight by absolute value of true return (high returns matter more)
    weights = np.abs(y_true) + 0.001  # Add small constant to avoid zero weights
    weights = weights / np.mean(weights)  # Normalize

    # Direction penalty: penalize getting the sign wrong
    sign_penalty = np.where(np.sign(y_pred) != np.sign(y_true), 2.0, 1.0)

    # Combined weights
    total_weights = weights * sign_penalty

    # Weighted MSE
    residual = y_pred - y_true
    grad = 2 * total_weights * residual
    hess = 2 * total_weights

    return grad, hess

def rank_loss(y_pred, dtrain):
    """
    Ranking-based loss: focus on getting the order right

    For Sharpe optimization, we care more about ranking than exact values.
    This is similar to learning-to-rank objectives.
    """
    y_true = dtrain.get_label()
    n = len(y_true)

    # Pairwise ranking objective
    # For each pair (i, j), if y_true[i] > y_true[j], we want y_pred[i] > y_pred[j]

    # Simplified approach: use rank correlation loss
    # This is an approximation - we use smooth ranking

    # Standard MSE for now (XGBoost expects (grad, hess) return)
    grad = 2 * (y_pred - y_true)
    hess = 2 * np.ones_like(y_pred)

    return grad, hess

print("="*80)
print("EXP-033: Direct Sharpe Ratio Optimization")
print("="*80)

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n[1] Loading data...")
df = pd.read_csv("data/train.csv")

# Targets
y_excess = df['market_forward_excess_returns'].values
fwd_returns = df['forward_returns'].values
risk_free = df['risk_free_rate'].values

# Load EXP-021 features
print("   Loading EXP-021 features...")
top_30_df = pd.read_csv("experiments/016/results/top_30_with_interactions.csv")
top_30 = top_30_df['feature'].tolist()
top_20 = pd.read_csv("experiments/016/results/top_20_features.csv")['feature'].tolist()

# Create base features
X_base = df[top_20].copy()
feature_names = top_20.copy()

# Add interactions (from top 10 base features)
interactions = []
top_10 = top_20[:10]
for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i+1:]:
        interactions.append(df[feat1] * df[feat2])
        feature_names.append(f'{feat1}*{feat2}')

for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i+1:]:
        interactions.append(df[feat1] / (df[feat2].abs() + 1e-8))
        feature_names.append(f'{feat1}/{feat2}')

# Add polynomials (from top 5 base features)
top_5 = top_20[:5]
for feat in top_5:
    interactions.append(df[feat] ** 2)
    feature_names.append(f'{feat}²')
    interactions.append(df[feat] ** 3)
    feature_names.append(f'{feat}³')

X_all_features = pd.concat(
    [X_base] + [pd.Series(feat, name=name) for feat, name in zip(interactions, feature_names[len(top_20):])],
    axis=1
)
X_all_features.columns = feature_names

# Select only top_30 features
X = X_all_features[top_30].copy()
X = X.fillna(0).replace([np.inf, -np.inf], 0)

print(f"   Feature matrix: {X.shape}")

# ============================================================================
# 2. Baseline (Standard MSE loss)
# ============================================================================
print("\n[2] Baseline: Standard MSE Loss")

K = 250
xgb_params_base = {
    'n_estimators': 150,
    'max_depth': 5,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
}

tscv = TimeSeriesSplit(n_splits=5)
baseline_results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    print(f"   Fold {fold_idx}/5...", end=" ")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_excess[train_idx], y_excess[val_idx]
    fwd_ret_val = fwd_returns[val_idx]
    rf_val = risk_free[val_idx]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train with standard objective
    model = xgb.XGBRegressor(**xgb_params_base, objective='reg:squarederror')
    model.fit(X_train_scaled, y_train, verbose=False)

    # Predict
    y_pred = model.predict(X_val_scaled)

    # Positions
    positions = np.clip(1.0 + y_pred * K, 0.0, 2.0)

    # Sharpe
    sharpe = calculate_sharpe(positions, fwd_ret_val, rf_val)

    print(f"Sharpe: {sharpe:.3f}")

    baseline_results.append({
        'fold': fold_idx,
        'sharpe': sharpe,
    })

df_baseline = pd.DataFrame(baseline_results)
baseline_sharpe = df_baseline['sharpe'].mean()
print(f"\n   Baseline CV Sharpe: {baseline_sharpe:.3f} ± {df_baseline['sharpe'].std():.3f}")

# ============================================================================
# 3. Strategy 1: Weighted Loss (emphasize high returns)
# ============================================================================
print("\n[3] Strategy 1: Sharpe-Weighted Loss")

strategy1_results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    print(f"   Fold {fold_idx}/5...", end=" ")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_excess[train_idx], y_excess[val_idx]
    fwd_ret_val = fwd_returns[val_idx]
    rf_val = risk_free[val_idx]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train with custom objective
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dval = xgb.DMatrix(X_val_scaled, label=y_val)

    params = {
        'max_depth': 5,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'alpha': 0.1,
        'lambda': 1.0,
        'seed': 42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=150,
        obj=sharpe_weighted_loss,
        verbose_eval=False
    )

    # Predict
    y_pred = model.predict(dval)

    # Positions
    positions = np.clip(1.0 + y_pred * K, 0.0, 2.0)

    # Sharpe
    sharpe = calculate_sharpe(positions, fwd_ret_val, rf_val)

    print(f"Sharpe: {sharpe:.3f}")

    strategy1_results.append({
        'fold': fold_idx,
        'sharpe': sharpe,
    })

df_s1 = pd.DataFrame(strategy1_results)
s1_sharpe = df_s1['sharpe'].mean()
s1_improvement = (s1_sharpe - baseline_sharpe) / baseline_sharpe * 100
s1_symbol = "✅" if s1_sharpe > baseline_sharpe else "❌"
print(f"\n   Strategy 1 CV Sharpe: {s1_sharpe:.3f} ± {df_s1['sharpe'].std():.3f} ({s1_improvement:+.1f}%) {s1_symbol}")

# ============================================================================
# 4. Strategy 2: Quantile Regression (predict median, less sensitive to outliers)
# ============================================================================
print("\n[4] Strategy 2: Quantile Regression (alpha=0.5)")

strategy2_results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    print(f"   Fold {fold_idx}/5...", end=" ")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_excess[train_idx], y_excess[val_idx]
    fwd_ret_val = fwd_returns[val_idx]
    rf_val = risk_free[val_idx]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train with quantile objective (median)
    model = xgb.XGBRegressor(**xgb_params_base, objective='reg:quantileerror', quantile_alpha=0.5)
    model.fit(X_train_scaled, y_train, verbose=False)

    # Predict
    y_pred = model.predict(X_val_scaled)

    # Positions
    positions = np.clip(1.0 + y_pred * K, 0.0, 2.0)

    # Sharpe
    sharpe = calculate_sharpe(positions, fwd_ret_val, rf_val)

    print(f"Sharpe: {sharpe:.3f}")

    strategy2_results.append({
        'fold': fold_idx,
        'sharpe': sharpe,
    })

df_s2 = pd.DataFrame(strategy2_results)
s2_sharpe = df_s2['sharpe'].mean()
s2_improvement = (s2_sharpe - baseline_sharpe) / baseline_sharpe * 100
s2_symbol = "✅" if s2_sharpe > baseline_sharpe else "❌"
print(f"\n   Strategy 2 CV Sharpe: {s2_sharpe:.3f} ± {df_s2['sharpe'].std():.3f} ({s2_improvement:+.1f}%) {s2_symbol}")

# ============================================================================
# 5. Strategy 3: Huber Loss (robust to outliers)
# ============================================================================
print("\n[5] Strategy 3: Huber Loss (robust regression)")

strategy3_results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    print(f"   Fold {fold_idx}/5...", end=" ")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_excess[train_idx], y_excess[val_idx]
    fwd_ret_val = fwd_returns[val_idx]
    rf_val = risk_free[val_idx]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train with Huber objective
    model = xgb.XGBRegressor(**xgb_params_base, objective='reg:pseudohubererror')
    model.fit(X_train_scaled, y_train, verbose=False)

    # Predict
    y_pred = model.predict(X_val_scaled)

    # Positions
    positions = np.clip(1.0 + y_pred * K, 0.0, 2.0)

    # Sharpe
    sharpe = calculate_sharpe(positions, fwd_ret_val, rf_val)

    print(f"Sharpe: {sharpe:.3f}")

    strategy3_results.append({
        'fold': fold_idx,
        'sharpe': sharpe,
    })

df_s3 = pd.DataFrame(strategy3_results)
s3_sharpe = df_s3['sharpe'].mean()
s3_improvement = (s3_sharpe - baseline_sharpe) / baseline_sharpe * 100
s3_symbol = "✅" if s3_sharpe > baseline_sharpe else "❌"
print(f"\n   Strategy 3 CV Sharpe: {s3_sharpe:.3f} ± {df_s3['sharpe'].std():.3f} ({s3_improvement:+.1f}%) {s3_symbol}")

# ============================================================================
# 6. Strategy 4: MAE Loss (L1, more robust than MSE)
# ============================================================================
print("\n[6] Strategy 4: MAE Loss (L1 regression)")

strategy4_results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    print(f"   Fold {fold_idx}/5...", end=" ")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_excess[train_idx], y_excess[val_idx]
    fwd_ret_val = fwd_returns[val_idx]
    rf_val = risk_free[val_idx]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train with MAE objective
    model = xgb.XGBRegressor(**xgb_params_base, objective='reg:absoluteerror')
    model.fit(X_train_scaled, y_train, verbose=False)

    # Predict
    y_pred = model.predict(X_val_scaled)

    # Positions
    positions = np.clip(1.0 + y_pred * K, 0.0, 2.0)

    # Sharpe
    sharpe = calculate_sharpe(positions, fwd_ret_val, rf_val)

    print(f"Sharpe: {sharpe:.3f}")

    strategy4_results.append({
        'fold': fold_idx,
        'sharpe': sharpe,
    })

df_s4 = pd.DataFrame(strategy4_results)
s4_sharpe = df_s4['sharpe'].mean()
s4_improvement = (s4_sharpe - baseline_sharpe) / baseline_sharpe * 100
s4_symbol = "✅" if s4_sharpe > baseline_sharpe else "❌"
print(f"\n   Strategy 4 CV Sharpe: {s4_sharpe:.3f} ± {df_s4['sharpe'].std():.3f} ({s4_improvement:+.1f}%) {s4_symbol}")

# ============================================================================
# 7. Results Comparison
# ============================================================================
print("\n[7] Results Comparison")
print("="*80)

results_summary = {
    'Baseline (MSE)': baseline_sharpe,
    'Strategy 1 (Sharpe-Weighted)': s1_sharpe,
    'Strategy 2 (Quantile α=0.5)': s2_sharpe,
    'Strategy 3 (Huber)': s3_sharpe,
    'Strategy 4 (MAE/L1)': s4_sharpe,
}

print(f"\n📊 All Strategies:")
best_sharpe = baseline_sharpe
best_strategy = 'Baseline'

for strategy, sharpe in results_summary.items():
    improvement = (sharpe - baseline_sharpe) / baseline_sharpe * 100 if strategy != 'Baseline (MSE)' else 0
    symbol = "✅" if sharpe > baseline_sharpe else "❌" if strategy != 'Baseline (MSE)' else "🔵"
    print(f"   {strategy:30s}: {sharpe:.3f} ({improvement:+.1f}%) {symbol}")

    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_strategy = strategy

print(f"\n🏆 Best Strategy: {best_strategy}")
print(f"   CV Sharpe: {best_sharpe:.3f}")
print(f"   vs EXP-021 (0.637): {(best_sharpe - 0.637) / 0.637 * 100:+.1f}%")
print(f"   Expected Public (8.6x): {best_sharpe * 8.6:.2f}")

# ============================================================================
# 8. Save Results
# ============================================================================
print(f"\n[8] Saving results...")

Path("experiments/033/results").mkdir(parents=True, exist_ok=True)

# Save all results
df_baseline.to_csv("experiments/033/results/baseline_mse.csv", index=False)
df_s1.to_csv("experiments/033/results/strategy1_sharpe_weighted.csv", index=False)
df_s2.to_csv("experiments/033/results/strategy2_quantile.csv", index=False)
df_s3.to_csv("experiments/033/results/strategy3_huber.csv", index=False)
df_s4.to_csv("experiments/033/results/strategy4_mae.csv", index=False)

# Summary
summary_data = []
for strategy, sharpe in results_summary.items():
    improvement = (sharpe - baseline_sharpe) / baseline_sharpe * 100 if strategy != 'Baseline (MSE)' else 0
    summary_data.append({
        'strategy': strategy,
        'cv_sharpe': sharpe,
        'improvement_pct': improvement,
        'expected_public': sharpe * 8.6,
    })

summary = pd.DataFrame(summary_data)
summary.to_csv("experiments/033/results/summary.csv", index=False)
print(f"   Results saved to: experiments/033/results/")

print("\n" + "="*80)
print("EXP-033 Complete!")
print("="*80)
