#!/usr/bin/env python3
"""
EXP-020 Phase 4: Multi-Objective Optimization

목표:
- Return과 Risk를 동시에 고려하는 position sizing
- Sharpe ratio 최대화 전략

Objectives:
1. Maximize return (excess return prediction)
2. Minimize risk (volatility prediction)
3. Maximize Sharpe (return / risk)

Strategies:
- Weighted combination
- Return/Risk ratio
- Sharpe-focused
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

def calculate_sharpe(positions, fwd_returns, risk_free):
    """Calculate Sharpe ratio from positions"""
    positions = np.clip(positions, 0.0, 2.0)

    strategy_returns = risk_free * (1.0 - positions) + fwd_returns * positions
    excess_returns = strategy_returns - risk_free

    if np.std(strategy_returns) > 0:
        sharpe = (np.mean(excess_returns) / np.std(strategy_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    return sharpe

print("=" * 80)
print("EXP-020 Phase 4: Multi-Objective Optimization")
print("=" * 80)
print()

# Load data
print("[1] Loading data...")
train = pd.read_csv("data/train.csv")
y_excess = train['market_forward_excess_returns'].copy()
fwd_returns = train['forward_returns'].copy()
risk_free = train['risk_free_rate'].copy()

# Create volatility target
print("[2] Creating volatility target...")
returns = train['forward_returns'].values
n_samples = len(returns)
vol_window = 20
vol_target = np.zeros(n_samples)

for i in range(n_samples - vol_window):
    future_returns = returns[i : i + vol_window]
    vol_target[i] = np.std(future_returns)

vol_target[n_samples - vol_window :] = vol_target[n_samples - vol_window - 1]
y_vol = pd.Series(vol_target, index=train.index)

# Load features
print("[3] Loading features...")
top_30_df = pd.read_csv("experiments/016/results/top_30_with_interactions.csv")
top_30 = top_30_df['feature'].tolist()
top_20 = pd.read_csv("experiments/016/results/top_20_features.csv")['feature'].tolist()

X_base = train[top_20].fillna(train[top_20].median()).replace([np.inf, -np.inf], np.nan).fillna(0)

# Create interactions
interactions = []
feature_names = top_20.copy()
top_10 = top_20[:10]
eps = 1e-8

for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i + 1 :]:
        interactions.append(X_base[feat1] * X_base[feat2])
        feature_names.append(f'{feat1}*{feat2}')

for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i + 1 :]:
        interactions.append(X_base[feat1] / (X_base[feat2].abs() + eps))
        feature_names.append(f'{feat1}/{feat2}')

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

X = X_all[top_30].copy()
print(f"   Features: {X.shape}")
print()

# Test multi-objective strategies
print("[4] Testing multi-objective strategies...")
print()

K = 250  # From EXP-016

# Multi-objective strategies
strategies = {
    'Baseline (return only)': lambda ret, vol, K: 1.0 + ret * K,
    'Return/Risk ratio': lambda ret, vol, K: 1.0 + (ret / (vol + 1e-6)) * (K / 10),
    'Return/Risk (scaled x20)': lambda ret, vol, K: 1.0 + (ret / (vol + 1e-6)) * (K / 5),
    'Weighted (ret=0.7, risk=0.3)': lambda ret, vol, K: 1.0 + ret * K - vol * K * 0.3,
    'Weighted (ret=0.5, risk=0.5)': lambda ret, vol, K: 1.0 + ret * K - vol * K * 0.5,
    'Sharpe-focused': lambda ret, vol, K: 1.0 + (ret / (vol + 1e-6)) * (K / 2.5),
}

results = {}

for strategy_name, position_func in strategies.items():
    print(f"Strategy: {strategy_name}")

    sharpes = []
    tscv = TimeSeriesSplit(n_splits=5)

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        X_tr = X.iloc[tr_idx]
        y_tr_excess = y_excess.iloc[tr_idx]
        y_tr_vol = y_vol.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_va_excess = y_excess.iloc[va_idx]
        y_va_vol = y_vol.iloc[va_idx]

        # Scale
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)

        # Train return prediction model
        xgb_return = XGBRegressor(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.025,
            subsample=1.0,
            colsample_bytree=0.6,
            reg_lambda=0.5,
            random_state=42,
            n_jobs=-1
        )
        xgb_return.fit(X_tr_scaled, y_tr_excess)

        # Train volatility prediction model
        xgb_vol = XGBRegressor(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.025,
            subsample=1.0,
            colsample_bytree=0.6,
            reg_lambda=0.5,
            random_state=42,
            n_jobs=-1
        )
        xgb_vol.fit(X_tr_scaled, y_tr_vol)

        # Predict
        excess_pred = xgb_return.predict(X_va_scaled)
        vol_pred = xgb_vol.predict(X_va_scaled)

        # Calculate positions
        positions = position_func(excess_pred, vol_pred, K)

        # Evaluate
        va_fwd = fwd_returns.iloc[va_idx].values
        va_rf = risk_free.iloc[va_idx].values

        sharpe = calculate_sharpe(positions, va_fwd, va_rf)
        sharpes.append(sharpe)

    avg_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes)

    results[strategy_name] = {
        'avg_sharpe': avg_sharpe,
        'std_sharpe': std_sharpe,
        'sharpes': sharpes
    }

    print(f"  Sharpe = {avg_sharpe:.4f} ± {std_sharpe:.4f}")
    print(f"  Folds: {[f'{s:.3f}' for s in sharpes]}")
    print()

# Summary
print("=" * 80)
print("STRATEGY COMPARISON:")
print("=" * 80)

sorted_strategies = sorted(results.items(), key=lambda x: x[1]['avg_sharpe'], reverse=True)

for rank, (name, result) in enumerate(sorted_strategies, 1):
    avg = result['avg_sharpe']
    std = result['std_sharpe']
    print(f"{rank}. {name:35s}: Sharpe = {avg:.4f} ± {std:.4f}")

print("=" * 80)
print()

# Save results
results_df = pd.DataFrame([
    {
        'strategy': name,
        'avg_sharpe': result['avg_sharpe'],
        'std_sharpe': result['std_sharpe'],
        'fold1': result['sharpes'][0],
        'fold2': result['sharpes'][1],
        'fold3': result['sharpes'][2],
        'fold4': result['sharpes'][3],
        'fold5': result['sharpes'][4]
    }
    for name, result in sorted_strategies
])
results_df.to_csv('experiments/020/results/phase4_multi_objective_comparison.csv', index=False)

print("Results saved to experiments/020/results/")
print()

# Best strategy
best_strategy = sorted_strategies[0][0]
best_sharpe = sorted_strategies[0][1]['avg_sharpe']

print(f"Best Strategy: {best_strategy}")
print(f"Best Sharpe: {best_sharpe:.4f}")
print()

# Compare with EXP-016 (CV Sharpe = 0.5590)
exp016_sharpe = 0.5590
improvement = ((best_sharpe - exp016_sharpe) / exp016_sharpe) * 100

print(f"EXP-016 Sharpe: {exp016_sharpe:.4f}")
print(f"Improvement: {improvement:+.2f}%")
print()

if best_sharpe > exp016_sharpe:
    print("✅ Multi-objective strategy IMPROVED performance!")
else:
    print("❌ Multi-objective strategy did NOT improve performance")

print()
print("Next: Phase 5 - Best Strategy Selection & InferenceServer")
