#!/usr/bin/env python3
"""
EXP-018 Phase 3: Integrated Model with Dynamic K

Goal: Build final model with best parameters and evaluate
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

print("="*80)
print("EXP-018 Phase 3: Integrated Model Evaluation")
print("="*80)
print()

# Load best parameters
print("[1] Loading best parameters from Phase 2...")
results_df = pd.read_csv("experiments/018/results/k_search_results.csv")
best_config = results_df.iloc[0]

print(f"Best configuration:")
print(f"  K_base: {best_config['k_base']:.0f}")
print(f"  vol_scale: {best_config['vol_scale']:.2f}")
print(f"  conf_mult: {best_config['conf_mult']:.2f}")
print(f"  long_ratio: {best_config['long_ratio']:.2f}")
print(f"  short_ratio: {best_config['short_ratio']:.2f}")
print(f"  Expected Sharpe: {best_config['mean_sharpe']:.4f} ± {best_config['std_sharpe']:.4f}")
print()

# Load data
print("[2] Loading data...")
train = pd.read_csv("data/train.csv")
y = train['market_forward_excess_returns'].copy()
fwd_returns = train['forward_returns'].copy()
risk_free = train['risk_free_rate'].copy()

# Load features
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

top_5_vol = ['vol_mean_V', 'vol_composite_V_all', 'vol_composite_V', 'cross_vol_MV', 'sentiment_dispersion']

# Create features
print("[3] Creating features...")
X_base = train[top_20_base].fillna(train[top_20_base].median()).replace([np.inf, -np.inf], 0)

# Interaction features
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

# Volatility features
vol_feats = {}
vol_feats['vol_mean_V'] = (abs(X_base['V13']) + abs(X_base['V7']) + abs(X_base['V9'])) / 3
vol_feats['vol_composite_V_all'] = np.sqrt(X_base['V13']**2 + X_base['V7']**2 + X_base['V9']**2 + X_base['V10']**2 + eps)
vol_feats['vol_composite_V'] = np.sqrt(X_base['V13']**2 + X_base['V7']**2 + X_base['V9']**2 + eps)
vol_feats['cross_vol_MV'] = np.sqrt(X_base['M4']**2 + X_base['V13']**2 + eps)
vol_feats['sentiment_dispersion'] = abs(X_base['S2'] - X_base['S5'])

X_vol = pd.DataFrame(vol_feats)
X_vol = X_vol.replace([np.inf, -np.inf], np.nan).fillna(0)

# Combine features
X_all = pd.concat([X_interactions, X_vol], axis=1)

missing = [f for f in top_30_interactions if f not in X_all.columns]
for f in missing:
    X_all[f] = 0.0

X_final = X_all[top_30_interactions + top_5_vol].copy()
X_final = X_final.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Feature matrix shape: {X_final.shape}")
print(f"  Top 30 interactions: {len(top_30_interactions)}")
print(f"  Top 5 volatility: {len(top_5_vol)}")
print()

# Volatility proxy
vol_proxy = X_vol['vol_mean_V'].to_numpy()

# Model parameters
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

# Dynamic K function
def calculate_sharpe_dynamic(y_pred, fwd_returns, risk_free, vol_proxy, k_base, vol_scale):
    """Calculate Sharpe with dynamic K"""
    # Dynamic K based on volatility (inverse)
    vol_adjustment = 1.0 / (1.0 + vol_scale * vol_proxy)
    k_dynamic = k_base * vol_adjustment

    # Convert to positions
    positions = np.clip(1.0 + y_pred * k_dynamic, 0.0, 2.0)

    # Calculate strategy returns
    strategy_returns = risk_free * (1.0 - positions) + fwd_returns * positions
    excess_returns = strategy_returns - risk_free

    # Calculate Sharpe
    if np.std(strategy_returns) > 1e-8:
        sharpe = (np.mean(excess_returns) / np.std(strategy_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    return sharpe, positions.mean(), positions.std(), k_dynamic.mean()

# 5-fold CV evaluation
print("[4] Running 5-fold CV evaluation...")
tscv = TimeSeriesSplit(n_splits=5)

fold_results = []

for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X_final), 1):
    print(f"\nFold {fold_idx}/5:")
    print("-" * 40)

    # Split data
    X_tr, X_va = X_final.iloc[tr_idx], X_final.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    fwd_va = fwd_returns.iloc[va_idx]
    rf_va = risk_free.iloc[va_idx]
    vol_va = vol_proxy[va_idx]

    # Scale
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)

    # Train
    model = XGBRegressor(**model_params)
    model.fit(X_tr_scaled, y_tr)

    # Predict
    y_va_pred = model.predict(X_va_scaled)

    # Evaluate with dynamic K
    sharpe, pos_mean, pos_std, k_mean = calculate_sharpe_dynamic(
        y_va_pred, fwd_va, rf_va, vol_va,
        k_base=best_config['k_base'],
        vol_scale=best_config['vol_scale']
    )

    fold_results.append({
        'fold': fold_idx,
        'sharpe': sharpe,
        'position_mean': pos_mean,
        'position_std': pos_std,
        'k_mean': k_mean
    })

    print(f"  Sharpe: {sharpe:.4f}")
    print(f"  Position: {pos_mean:.4f} ± {pos_std:.4f}")
    print(f"  K_mean: {k_mean:.2f}")

print()
print("="*80)
print("Cross-Validation Results")
print("="*80)

fold_df = pd.DataFrame(fold_results)

print(f"\nSharpe by fold:")
for idx, row in fold_df.iterrows():
    print(f"  Fold {row['fold']}: {row['sharpe']:.4f}")

print(f"\nOverall Statistics:")
print(f"  Mean Sharpe: {fold_df['sharpe'].mean():.4f} ± {fold_df['sharpe'].std():.4f}")
print(f"  Min Sharpe:  {fold_df['sharpe'].min():.4f}")
print(f"  Max Sharpe:  {fold_df['sharpe'].max():.4f}")
print(f"  Mean Position: {fold_df['position_mean'].mean():.4f}")
print(f"  Mean K: {fold_df['k_mean'].mean():.2f}")
print()

# Compare with EXP-016
print("Comparison with EXP-016:")
print("-" * 40)
print(f"EXP-016 Sharpe: 0.559 ± 0.362 (K=250 fixed)")
print(f"EXP-018 Sharpe: {fold_df['sharpe'].mean():.4f} ± {fold_df['sharpe'].std():.4f} (K={best_config['k_base']:.0f} dynamic)")
improvement = (fold_df['sharpe'].mean() - 0.559) / 0.559 * 100
print(f"Improvement: {improvement:+.2f}%")
print()

# Save results
fold_df.to_csv("experiments/018/results/final_cv_results.csv", index=False)

# Save configuration
config_df = pd.DataFrame([{
    'k_base': best_config['k_base'],
    'vol_scale': best_config['vol_scale'],
    'conf_mult': best_config['conf_mult'],
    'long_ratio': best_config['long_ratio'],
    'short_ratio': best_config['short_ratio'],
    'mean_sharpe': fold_df['sharpe'].mean(),
    'std_sharpe': fold_df['sharpe'].std(),
    'n_features': X_final.shape[1]
}])
config_df.to_csv("experiments/018/results/final_config.csv", index=False)

print(f"Results saved to experiments/018/results/")
print()

# Feature importance analysis
print("[5] Feature Importance Analysis...")
print("-" * 80)

# Train on full dataset for feature importance
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X_final)

model_full = XGBRegressor(**model_params)
model_full.fit(X_scaled_full, y)

importances = model_full.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_final.columns,
    'importance': importances
}).sort_values('importance', ascending=False).reset_index(drop=True)

print("\nTop 15 Features:")
for idx, row in feature_importance_df.head(15).iterrows():
    feature_type = "VOL" if row['feature'] in top_5_vol else "INT"
    print(f"  {idx+1:2d}. [{feature_type}] {row['feature']:30s} {row['importance']:.6f}")

print(f"\nVolatility features in top 15: {sum(1 for f in feature_importance_df.head(15)['feature'] if f in top_5_vol)}")

feature_importance_df.to_csv("experiments/018/results/feature_importance.csv", index=False)

print()
print("="*80)
print("Phase 3 Complete!")
print("="*80)
print()
print(f"Expected Public Score range: {fold_df['sharpe'].mean() * 7.0:.1f} - {fold_df['sharpe'].mean() * 9.0:.1f}")
print("(Based on EXP-016 CV-to-Public ratio of ~7-8x)")
