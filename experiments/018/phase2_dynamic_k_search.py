#!/usr/bin/env python3
"""
EXP-018 Phase 2: Dynamic K Parameter Grid Search

Goal: Find optimal K parameters for volatility-adaptive strategy
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from itertools import product

print("="*80)
print("EXP-018 Phase 2: Dynamic K Parameter Grid Search")
print("="*80)
print()

# Load data
print("[1] Loading data...")
train = pd.read_csv("data/train.csv")
y = train['market_forward_excess_returns'].copy()
fwd_returns = train['forward_returns'].copy()
risk_free = train['risk_free_rate'].copy()

print(f"Train data shape: {train.shape}")

# Load top features from EXP-016
top_20_base = [
    'M4', 'V13', 'V7', 'P8', 'S2', 'I2', 'E19', 'S5', 'P5', 'P7',
    'M2', 'V9', 'M3', 'P12', 'P10', 'V10', 'E12', 'P11', 'M12', 'S8'
]

top_30_interactions = [
    'P8*S2', 'M4*V7', 'P8/P7', 'V7*P7', 'M4/S2',
    'S2*S5', 'S5/P7', 'M4*P8', 'M4²', 'V13²',
    'V7/P7', 'P8²', 'V7*I2', 'I2*E19', 'M4/P8',
    'S2/P5', 'V7*P5', 'P5', 'P5/P7', 'V7/P8',
    'M4/I2', 'M4/V7', 'M4/P5', 'P8/P5', 'V13/S2',
    'V13*I2', 'M4/E19', 'M4/P7', 'I2/S5', 'V13/P7'
]

# Load top volatility features
vol_features_df = pd.read_csv("experiments/018/results/top_10_volatility_features.csv")
top_vol_features = vol_features_df['feature'].tolist()[:5]  # Use top 5

print(f"Base features: {len(top_20_base)}")
print(f"Interaction features: {len(top_30_interactions)}")
print(f"Volatility features: {len(top_vol_features)}")
print(f"Top 5 volatility features: {top_vol_features}")
print()

# Create features
print("[2] Creating features...")
X_base = train[top_20_base].fillna(train[top_20_base].median()).replace([np.inf, -np.inf], 0)

# Create interaction features (same as EXP-016)
eps = 1e-8
interactions = []
feature_names = []

top_10 = top_20_base[:10]
top_5 = top_20_base[:5]

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
for feat in top_5:
    interactions.append(X_base[feat] ** 2)
    feature_names.append(f'{feat}²')
    interactions.append(X_base[feat] ** 3)
    feature_names.append(f'{feat}³')

X_interactions = pd.concat([pd.Series(feat, name=name) for feat, name in zip(interactions, feature_names)], axis=1)
X_interactions = X_interactions.replace([np.inf, -np.inf], np.nan).fillna(0)

# Create volatility features
M_features = ['M4', 'M2', 'M3', 'M12']
V_features = ['V13', 'V7', 'V9', 'V10']
P_features = ['P8', 'P7', 'P5', 'P12']
S_features = ['S2', 'S5', 'S8']

vol_feats = {}
vol_feats['vol_mean_V'] = (abs(X_base['V13']) + abs(X_base['V7']) + abs(X_base['V9'])) / 3
vol_feats['vol_composite_V_all'] = np.sqrt(X_base['V13']**2 + X_base['V7']**2 + X_base['V9']**2 + X_base['V10']**2 + eps)
vol_feats['vol_composite_V'] = np.sqrt(X_base['V13']**2 + X_base['V7']**2 + X_base['V9']**2 + eps)
vol_feats['cross_vol_MV'] = np.sqrt(X_base['M4']**2 + X_base['V13']**2 + eps)
vol_feats['sentiment_dispersion'] = abs(X_base['S2'] - X_base['S5'])

X_vol = pd.DataFrame(vol_feats)
X_vol = X_vol.replace([np.inf, -np.inf], np.nan).fillna(0)

# Combine: Top 30 interactions + Top 5 volatility
X_all = pd.concat([X_interactions, X_vol], axis=1)

# Select features
missing = [f for f in top_30_interactions if f not in X_all.columns]
for f in missing:
    X_all[f] = 0.0

X_final = X_all[top_30_interactions + top_vol_features].copy()
X_final = X_final.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Final feature matrix: {X_final.shape}")
print()

# Create volatility proxy for dynamic K (use vol_mean_V as main proxy)
vol_proxy = X_vol['vol_mean_V'].to_numpy()

# Dynamic K calculation function
def calculate_sharpe_dynamic_k(y_pred, fwd_returns, risk_free, vol_proxy,
                               k_base=250, vol_scale=1.0, conf_mult=1.0,
                               long_ratio=1.0, short_ratio=1.0):
    """
    Calculate Sharpe with dynamic K based on volatility and confidence

    Args:
        y_pred: Predicted excess returns
        fwd_returns: Forward returns
        risk_free: Risk-free rate
        vol_proxy: Volatility proxy values
        k_base: Base K parameter
        vol_scale: Volatility scaling factor
        conf_mult: Confidence multiplier
        long_ratio: Long position multiplier
        short_ratio: Short position multiplier
    """
    # Dynamic K based on volatility (inverse relationship)
    # High volatility -> lower K (more conservative)
    vol_adjustment = 1.0 / (1.0 + vol_scale * vol_proxy)
    k_dynamic = k_base * vol_adjustment

    # Confidence-based adjustment (higher confidence -> higher K)
    confidence = np.abs(y_pred)
    k_adjusted = k_dynamic * (1.0 + conf_mult * confidence)

    # Asymmetric long/short
    k_final = np.where(y_pred > 0, k_adjusted * long_ratio, k_adjusted * short_ratio)

    # Convert to positions
    positions = np.clip(1.0 + y_pred * k_final, 0.0, 2.0)

    # Calculate strategy returns
    strategy_returns = risk_free * (1.0 - positions) + fwd_returns * positions
    excess_returns = strategy_returns - risk_free

    # Calculate Sharpe
    if np.std(strategy_returns) > 1e-8:
        sharpe = (np.mean(excess_returns) / np.std(strategy_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    return sharpe, positions.mean(), positions.std()

# Grid search parameters
print("[3] Setting up grid search...")
param_grid = {
    'k_base': [150, 200, 250, 300, 350],
    'vol_scale': [0.3, 0.5, 0.7, 1.0, 1.5],
    'conf_mult': [0.0, 0.5, 1.0, 1.5],
    'long_ratio': [0.8, 1.0, 1.2],
    'short_ratio': [0.8, 1.0, 1.2]
}

# Sample combinations to reduce search space
# Full grid would be 5*5*4*3*3 = 900 combinations
# We'll use strategic sampling
strategic_params = []

# Strategy 1: Baseline (no adjustments)
for k in param_grid['k_base']:
    strategic_params.append({
        'k_base': k,
        'vol_scale': 0.0,
        'conf_mult': 0.0,
        'long_ratio': 1.0,
        'short_ratio': 1.0
    })

# Strategy 2: Volatility-only adjustment
for k in param_grid['k_base']:
    for vs in param_grid['vol_scale']:
        if vs > 0:
            strategic_params.append({
                'k_base': k,
                'vol_scale': vs,
                'conf_mult': 0.0,
                'long_ratio': 1.0,
                'short_ratio': 1.0
            })

# Strategy 3: Confidence-only adjustment
for k in [200, 250, 300]:
    for cm in param_grid['conf_mult']:
        if cm > 0:
            strategic_params.append({
                'k_base': k,
                'vol_scale': 0.0,
                'conf_mult': cm,
                'long_ratio': 1.0,
                'short_ratio': 1.0
            })

# Strategy 4: Asymmetric long/short
for k in [200, 250, 300]:
    for lr in param_grid['long_ratio']:
        for sr in param_grid['short_ratio']:
            if lr != 1.0 or sr != 1.0:
                strategic_params.append({
                    'k_base': k,
                    'vol_scale': 0.5,
                    'conf_mult': 0.0,
                    'long_ratio': lr,
                    'short_ratio': sr
                })

# Strategy 5: Combined adjustments (best from each strategy)
for k in [200, 250, 300]:
    for vs in [0.5, 1.0]:
        for cm in [0.5, 1.0]:
            for lr in [0.8, 1.0, 1.2]:
                strategic_params.append({
                    'k_base': k,
                    'vol_scale': vs,
                    'conf_mult': cm,
                    'long_ratio': lr,
                    'short_ratio': 1.0
                })

print(f"Total combinations to test: {len(strategic_params)}")
print()

# Model setup
model_params = {
    'n_estimators': 150,
    'max_depth': 7,
    'learning_rate': 0.025,
    'subsample': 1.0,
    'colsample_bytree': 0.6,
    'reg_lambda': 0.5,
    'random_state': 42,
    'n_jobs': -1
}

# Cross-validation
print("[4] Running grid search with 5-fold CV...")
tscv = TimeSeriesSplit(n_splits=5)

results = []
best_sharpe = -np.inf
best_params = None

for idx, params in enumerate(strategic_params):
    if (idx + 1) % 50 == 0:
        print(f"  Progress: {idx+1}/{len(strategic_params)} ({100*(idx+1)/len(strategic_params):.1f}%)")

    fold_sharpes = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X_final)):
        # Split data
        X_tr, X_va = X_final.iloc[tr_idx], X_final.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        fwd_tr, fwd_va = fwd_returns.iloc[tr_idx], fwd_returns.iloc[va_idx]
        rf_tr, rf_va = risk_free.iloc[tr_idx], risk_free.iloc[va_idx]
        vol_tr, vol_va = vol_proxy[tr_idx], vol_proxy[va_idx]

        # Scale
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)

        # Train
        model = XGBRegressor(**model_params)
        model.fit(X_tr_scaled, y_tr)

        # Predict
        y_va_pred = model.predict(X_va_scaled)

        # Calculate Sharpe with dynamic K
        sharpe, pos_mean, pos_std = calculate_sharpe_dynamic_k(
            y_va_pred, fwd_va, rf_va, vol_va, **params
        )

        fold_sharpes.append(sharpe)

    # Average across folds
    mean_sharpe = np.mean(fold_sharpes)
    std_sharpe = np.std(fold_sharpes)

    results.append({
        **params,
        'mean_sharpe': mean_sharpe,
        'std_sharpe': std_sharpe,
        'min_sharpe': np.min(fold_sharpes),
        'max_sharpe': np.max(fold_sharpes)
    })

    if mean_sharpe > best_sharpe:
        best_sharpe = mean_sharpe
        best_params = params
        print(f"\n  New best! Sharpe={mean_sharpe:.4f} ± {std_sharpe:.4f}")
        print(f"    Params: {params}")
        print()

print()
print("="*80)
print("Grid Search Complete!")
print("="*80)
print()

# Save results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('mean_sharpe', ascending=False).reset_index(drop=True)
results_df.to_csv("experiments/018/results/k_search_results.csv", index=False)

print(f"Total combinations tested: {len(results)}")
print(f"Best mean Sharpe: {best_sharpe:.4f}")
print(f"\nBest parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")
print()

# Top 10 configurations
print("Top 10 Configurations:")
print("-" * 80)
for idx, row in results_df.head(10).iterrows():
    print(f"{idx+1:2d}. Sharpe={row['mean_sharpe']:.4f} ± {row['std_sharpe']:.4f} | "
          f"K={row['k_base']:.0f}, vol_scale={row['vol_scale']:.1f}, "
          f"conf_mult={row['conf_mult']:.1f}, long={row['long_ratio']:.1f}, short={row['short_ratio']:.1f}")

print()
print("="*80)
print("Phase 2 Complete!")
print("="*80)
