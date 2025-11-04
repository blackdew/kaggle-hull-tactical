"""
EXP-035: Hyperparameter Tuning on Best Configuration

Best config from EXP-034: EXP-028 (105 features) + MAE + Rank
CV Sharpe: 0.669

Tune:
- learning_rate
- max_depth
- n_estimators
- subsample / colsample_bytree
- K (position sizing parameter)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from scipy.stats import rankdata
from itertools import product
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

print("="*80)
print("EXP-035: Hyperparameter Tuning")
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

# Load EXP-028 features (105)
print("   Loading EXP-028 features (105)...")
exp028_features = pd.read_csv("experiments/028/results/top105_features.csv")
features_028 = exp028_features['feature'].tolist()

engineered_data = pd.read_parquet("experiments/024/results/phase2_engineered_features.parquet")
phase2_list = pd.read_csv("experiments/024/results/phase2_feature_list.csv")
original_features = phase2_list[phase2_list['type'] == 'original']['feature'].tolist()
original_data = df[original_features].copy()
X_all = pd.concat([original_data, engineered_data], axis=1)
X = X_all[features_028].copy()
X = X.fillna(0).replace([np.inf, -np.inf], 0)

print(f"   Feature matrix: {X.shape}")

# ============================================================================
# 2. Baseline (EXP-034 best config)
# ============================================================================
print("\n[2] Baseline: EXP-034 Best Config")
print("   (105 features + MAE + Rank, lr=0.03, depth=5, K=250)")

K_baseline = 250
xgb_params_baseline = {
    'n_estimators': 150,
    'max_depth': 5,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'objective': 'reg:absoluteerror',
}

tscv = TimeSeriesSplit(n_splits=5)
baseline_results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_excess[train_idx], y_excess[val_idx]
    fwd_ret_val = fwd_returns[val_idx]
    rf_val = risk_free[val_idx]

    # Convert to rank
    y_train_rank = rankdata(y_train, method='average') / len(y_train)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train
    model = xgb.XGBRegressor(**xgb_params_baseline)
    model.fit(X_train_scaled, y_train_rank, verbose=False)

    # Predict
    y_pred_rank = model.predict(X_val_scaled)
    y_pred_rank = np.clip(y_pred_rank, 0.0, 1.0)

    # Convert back
    y_train_sorted = np.sort(y_train)
    indices = (y_pred_rank * (len(y_train_sorted) - 1)).astype(int)
    y_pred = y_train_sorted[indices]

    # Positions
    positions = np.clip(1.0 + y_pred * K_baseline, 0.0, 2.0)

    # Sharpe
    sharpe = calculate_sharpe(positions, fwd_ret_val, rf_val)

    baseline_results.append({'fold': fold_idx, 'sharpe': sharpe})

df_baseline = pd.DataFrame(baseline_results)
baseline_sharpe = df_baseline['sharpe'].mean()
print(f"   Baseline CV Sharpe: {baseline_sharpe:.3f} ± {df_baseline['sharpe'].std():.3f}")

# ============================================================================
# 3. Grid Search: Learning Rate & Max Depth
# ============================================================================
print("\n[3] Grid Search: Learning Rate & Max Depth")

param_grid = {
    'learning_rate': [0.01, 0.02, 0.03, 0.05],
    'max_depth': [4, 5, 6, 7],
}

print(f"   Testing {len(param_grid['learning_rate']) * len(param_grid['max_depth'])} configurations...")

grid_results = []

for lr, depth in product(param_grid['learning_rate'], param_grid['max_depth']):
    print(f"   lr={lr}, depth={depth}...", end=" ")

    xgb_params = xgb_params_baseline.copy()
    xgb_params['learning_rate'] = lr
    xgb_params['max_depth'] = depth

    fold_sharpes = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_excess[train_idx], y_excess[val_idx]
        fwd_ret_val = fwd_returns[val_idx]
        rf_val = risk_free[val_idx]

        # Rank target
        y_train_rank = rankdata(y_train, method='average') / len(y_train)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train_scaled, y_train_rank, verbose=False)

        # Predict
        y_pred_rank = model.predict(X_val_scaled)
        y_pred_rank = np.clip(y_pred_rank, 0.0, 1.0)

        # Convert back
        y_train_sorted = np.sort(y_train)
        indices = (y_pred_rank * (len(y_train_sorted) - 1)).astype(int)
        y_pred = y_train_sorted[indices]

        # Positions
        positions = np.clip(1.0 + y_pred * K_baseline, 0.0, 2.0)

        # Sharpe
        sharpe = calculate_sharpe(positions, fwd_ret_val, rf_val)
        fold_sharpes.append(sharpe)

    cv_sharpe = np.mean(fold_sharpes)
    print(f"CV Sharpe: {cv_sharpe:.3f}")

    grid_results.append({
        'learning_rate': lr,
        'max_depth': depth,
        'cv_sharpe': cv_sharpe,
        'cv_std': np.std(fold_sharpes),
    })

df_grid = pd.DataFrame(grid_results).sort_values('cv_sharpe', ascending=False)
best_lr = df_grid.iloc[0]['learning_rate']
best_depth = int(df_grid.iloc[0]['max_depth'])
best_grid_sharpe = df_grid.iloc[0]['cv_sharpe']

print(f"\n   Best: lr={best_lr}, depth={best_depth} → CV Sharpe: {best_grid_sharpe:.3f}")
print(f"   Improvement: {(best_grid_sharpe - baseline_sharpe) / baseline_sharpe * 100:+.1f}%")

# ============================================================================
# 4. Tune K (Position Sizing Parameter)
# ============================================================================
print("\n[4] Tuning K (Position Sizing)")

K_values = [150, 200, 250, 300, 350, 400]
print(f"   Testing K values: {K_values}")

# Use best hyperparameters from grid search
xgb_params_best = xgb_params_baseline.copy()
xgb_params_best['learning_rate'] = best_lr
xgb_params_best['max_depth'] = best_depth

k_results = []

for K in K_values:
    print(f"   K={K}...", end=" ")

    fold_sharpes = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_excess[train_idx], y_excess[val_idx]
        fwd_ret_val = fwd_returns[val_idx]
        rf_val = risk_free[val_idx]

        # Rank target
        y_train_rank = rankdata(y_train, method='average') / len(y_train)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train
        model = xgb.XGBRegressor(**xgb_params_best)
        model.fit(X_train_scaled, y_train_rank, verbose=False)

        # Predict
        y_pred_rank = model.predict(X_val_scaled)
        y_pred_rank = np.clip(y_pred_rank, 0.0, 1.0)

        # Convert back
        y_train_sorted = np.sort(y_train)
        indices = (y_pred_rank * (len(y_train_sorted) - 1)).astype(int)
        y_pred = y_train_sorted[indices]

        # Positions with current K
        positions = np.clip(1.0 + y_pred * K, 0.0, 2.0)

        # Sharpe
        sharpe = calculate_sharpe(positions, fwd_ret_val, rf_val)
        fold_sharpes.append(sharpe)

    cv_sharpe = np.mean(fold_sharpes)
    print(f"CV Sharpe: {cv_sharpe:.3f}")

    k_results.append({
        'K': K,
        'cv_sharpe': cv_sharpe,
        'cv_std': np.std(fold_sharpes),
    })

df_k = pd.DataFrame(k_results).sort_values('cv_sharpe', ascending=False)
best_K = int(df_k.iloc[0]['K'])
best_k_sharpe = df_k.iloc[0]['cv_sharpe']

print(f"\n   Best K: {best_K} → CV Sharpe: {best_k_sharpe:.3f}")
print(f"   Improvement: {(best_k_sharpe - best_grid_sharpe) / best_grid_sharpe * 100:+.1f}%")

# ============================================================================
# 5. Final Best Configuration
# ============================================================================
print("\n[5] Final Best Configuration")
print("="*80)

final_config = {
    'learning_rate': best_lr,
    'max_depth': best_depth,
    'K': best_K,
    'cv_sharpe': best_k_sharpe,
}

print(f"\n📊 Best Hyperparameters:")
print(f"   learning_rate: {best_lr}")
print(f"   max_depth: {best_depth}")
print(f"   K: {best_K}")
print(f"\n   CV Sharpe: {best_k_sharpe:.3f}")
print(f"   vs Baseline (0.669): {(best_k_sharpe - 0.669) / 0.669 * 100:+.1f}%")
print(f"   vs EXP-021 (0.637): {(best_k_sharpe - 0.637) / 0.637 * 100:+.1f}%")
print(f"   Expected Public (8.6x): {best_k_sharpe * 8.6:.2f}")

# ============================================================================
# 6. Save Results
# ============================================================================
print(f"\n[6] Saving results...")

Path("experiments/035/results").mkdir(parents=True, exist_ok=True)

# Save grid search results
df_grid.to_csv("experiments/035/results/grid_search_lr_depth.csv", index=False)

# Save K tuning results
df_k.to_csv("experiments/035/results/k_tuning.csv", index=False)

# Save best config
pd.DataFrame([final_config]).to_csv("experiments/035/results/best_config.csv", index=False)

# Summary
summary = {
    'baseline_sharpe': baseline_sharpe,
    'best_sharpe': best_k_sharpe,
    'improvement_pct': (best_k_sharpe - baseline_sharpe) / baseline_sharpe * 100,
    'best_lr': best_lr,
    'best_depth': best_depth,
    'best_K': best_K,
    'expected_public': best_k_sharpe * 8.6,
}

pd.DataFrame([summary]).to_csv("experiments/035/results/summary.csv", index=False)
print(f"   Results saved to: experiments/035/results/")

print("\n" + "="*80)
print("EXP-035 Complete!")
print("="*80)
