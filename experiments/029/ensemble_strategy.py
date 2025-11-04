"""
EXP-029: Ensemble Strategy

Multiple approaches:
1. Multi-seed ensemble with EXP-021 features
2. Feature-set ensemble (EXP-021 + EXP-028)
3. Model ensemble (XGBoost + LightGBM)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
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
print("EXP-029: Ensemble Strategy")
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

# Recreate EXP-021 features (from EXP-016)
print("   Loading EXP-016 features...")
top_30_df = pd.read_csv("experiments/016/results/top_30_with_interactions.csv")
top_30 = top_30_df['feature'].tolist()
top_20 = pd.read_csv("experiments/016/results/top_20_features.csv")['feature'].tolist()

# Create base features
X_base = df[top_20].copy()
feature_names = top_20.copy()

# Add interactions (from top 10 base features, matching EXP-021)
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

# Add polynomials (from top 5 base features, matching EXP-021)
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

# Select only top_30 features (matching EXP-021)
top_30 = top_30_df['feature'].tolist()
X_021_temp = X_all_features[top_30].copy()
features_021 = top_30

# Load EXP-028 features (105)
exp028_features = pd.read_csv("experiments/028/results/top105_features.csv")
features_028 = exp028_features['feature'].tolist()

print(f"   EXP-021 features: {len(features_021)}")
print(f"   EXP-028 features: {len(features_028)}")

# ============================================================================
# 2. Prepare Feature Matrices
# ============================================================================
print("\n[2] Preparing feature matrices...")

# For EXP-021 features (already created above)
X_021 = X_021_temp.fillna(0).replace([np.inf, -np.inf], 0)

# For EXP-028 features
engineered_data = pd.read_parquet("experiments/024/results/phase2_engineered_features.parquet")
phase2_list = pd.read_csv("experiments/024/results/phase2_feature_list.csv")
original_features = phase2_list[phase2_list['type'] == 'original']['feature'].tolist()
original_data = df[original_features].copy()
X_all = pd.concat([original_data, engineered_data], axis=1)
X_028 = X_all[features_028].copy()
X_028 = X_028.fillna(0).replace([np.inf, -np.inf], 0)

print(f"   X_021: {X_021.shape}")
print(f"   X_028: {X_028.shape}")

# ============================================================================
# 3. Strategy 1: Multi-Seed Ensemble with EXP-021 Features
# ============================================================================
print("\n[3] Strategy 1: Multi-Seed Ensemble (5 seeds) with EXP-021 features")

K = 250
xgb_params_base = {
    'n_estimators': 150,
    'max_depth': 5,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_jobs': -1,
}

seeds = [42, 123, 456, 789, 999]
tscv = TimeSeriesSplit(n_splits=5)
strategy1_results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_021), 1):
    print(f"\n   Fold {fold_idx}/5...")

    X_train, X_val = X_021.iloc[train_idx], X_021.iloc[val_idx]
    y_train, y_val = y_excess[train_idx], y_excess[val_idx]
    fwd_ret_val = fwd_returns[val_idx]
    rf_val = risk_free[val_idx]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train multiple models with different seeds
    seed_predictions = []
    for seed in seeds:
        xgb_params = xgb_params_base.copy()
        xgb_params['random_state'] = seed

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train_scaled, y_train, verbose=False)

        y_pred = model.predict(X_val_scaled)
        seed_predictions.append(y_pred)

    # Average predictions
    y_pred_avg = np.mean(seed_predictions, axis=0)

    # Positions
    positions = np.clip(1.0 + y_pred_avg * K, 0.0, 2.0)

    # Sharpe
    sharpe = calculate_sharpe(positions, fwd_ret_val, rf_val)

    pos_mean = np.mean(positions)
    pos_std = np.std(positions)
    pred_std = np.std(y_pred_avg)

    print(f"      Sharpe: {sharpe:.3f}")
    print(f"      Position: {pos_mean:.3f} ± {pos_std:.3f}")

    strategy1_results.append({
        'fold': fold_idx,
        'sharpe': sharpe,
        'pos_mean': pos_mean,
        'pos_std': pos_std,
    })

df_s1 = pd.DataFrame(strategy1_results)
s1_sharpe = df_s1['sharpe'].mean()
print(f"\n   Strategy 1 CV Sharpe: {s1_sharpe:.3f} ± {df_s1['sharpe'].std():.3f}")

# ============================================================================
# 4. Strategy 2: Feature-Set Ensemble (021 + 028)
# ============================================================================
print("\n[4] Strategy 2: Feature-Set Ensemble (EXP-021 + EXP-028)")

strategy2_results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_021), 1):
    print(f"\n   Fold {fold_idx}/5...")

    # EXP-021 model
    X_train_021, X_val_021 = X_021.iloc[train_idx], X_021.iloc[val_idx]
    y_train, y_val = y_excess[train_idx], y_excess[val_idx]
    fwd_ret_val = fwd_returns[val_idx]
    rf_val = risk_free[val_idx]

    scaler_021 = StandardScaler()
    X_train_021_scaled = scaler_021.fit_transform(X_train_021)
    X_val_021_scaled = scaler_021.transform(X_val_021)

    model_021 = xgb.XGBRegressor(**xgb_params_base, random_state=42)
    model_021.fit(X_train_021_scaled, y_train, verbose=False)
    y_pred_021 = model_021.predict(X_val_021_scaled)

    # EXP-028 model
    X_train_028, X_val_028 = X_028.iloc[train_idx], X_028.iloc[val_idx]

    scaler_028 = StandardScaler()
    X_train_028_scaled = scaler_028.fit_transform(X_train_028)
    X_val_028_scaled = scaler_028.transform(X_val_028)

    model_028 = xgb.XGBRegressor(**xgb_params_base, random_state=42)
    model_028.fit(X_train_028_scaled, y_train, verbose=False)
    y_pred_028 = model_028.predict(X_val_028_scaled)

    # Weighted average (EXP-021 gets more weight since it's better)
    y_pred_avg = 0.6 * y_pred_021 + 0.4 * y_pred_028

    # Positions
    positions = np.clip(1.0 + y_pred_avg * K, 0.0, 2.0)

    # Sharpe
    sharpe = calculate_sharpe(positions, fwd_ret_val, rf_val)

    pos_mean = np.mean(positions)
    pos_std = np.std(positions)

    print(f"      Sharpe: {sharpe:.3f}")
    print(f"      Position: {pos_mean:.3f} ± {pos_std:.3f}")

    strategy2_results.append({
        'fold': fold_idx,
        'sharpe': sharpe,
        'pos_mean': pos_mean,
        'pos_std': pos_std,
    })

df_s2 = pd.DataFrame(strategy2_results)
s2_sharpe = df_s2['sharpe'].mean()
print(f"\n   Strategy 2 CV Sharpe: {s2_sharpe:.3f} ± {df_s2['sharpe'].std():.3f}")

# ============================================================================
# 5. Strategy 3: Model Ensemble (XGBoost + LightGBM)
# ============================================================================
print("\n[5] Strategy 3: Model Ensemble (XGBoost + LightGBM)")

lgb_params = {
    'n_estimators': 150,
    'max_depth': 5,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'verbose': -1,
}

strategy3_results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_021), 1):
    print(f"\n   Fold {fold_idx}/5...")

    X_train, X_val = X_021.iloc[train_idx], X_021.iloc[val_idx]
    y_train, y_val = y_excess[train_idx], y_excess[val_idx]
    fwd_ret_val = fwd_returns[val_idx]
    rf_val = risk_free[val_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # XGBoost
    model_xgb = xgb.XGBRegressor(**xgb_params_base, random_state=42)
    model_xgb.fit(X_train_scaled, y_train, verbose=False)
    y_pred_xgb = model_xgb.predict(X_val_scaled)

    # LightGBM
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(X_train_scaled, y_train)
    y_pred_lgb = model_lgb.predict(X_val_scaled)

    # Average
    y_pred_avg = 0.5 * y_pred_xgb + 0.5 * y_pred_lgb

    # Positions
    positions = np.clip(1.0 + y_pred_avg * K, 0.0, 2.0)

    # Sharpe
    sharpe = calculate_sharpe(positions, fwd_ret_val, rf_val)

    pos_mean = np.mean(positions)
    pos_std = np.std(positions)

    print(f"      Sharpe: {sharpe:.3f}")
    print(f"      Position: {pos_mean:.3f} ± {pos_std:.3f}")

    strategy3_results.append({
        'fold': fold_idx,
        'sharpe': sharpe,
        'pos_mean': pos_mean,
        'pos_std': pos_std,
    })

df_s3 = pd.DataFrame(strategy3_results)
s3_sharpe = df_s3['sharpe'].mean()
print(f"\n   Strategy 3 CV Sharpe: {s3_sharpe:.3f} ± {df_s3['sharpe'].std():.3f}")

# ============================================================================
# 6. Results Comparison
# ============================================================================
print("\n[6] Results Comparison")
print("="*80)

results_summary = {
    'EXP-021 (Baseline)': 0.637,
    'Strategy 1 (Multi-Seed)': s1_sharpe,
    'Strategy 2 (Feature Ensemble)': s2_sharpe,
    'Strategy 3 (Model Ensemble)': s3_sharpe,
}

print(f"\n📊 All Strategies:")
best_strategy = None
best_sharpe = 0.637

for strategy, sharpe in results_summary.items():
    improvement = (sharpe - 0.637) / 0.637 * 100 if strategy != 'EXP-021 (Baseline)' else 0
    symbol = "✅" if sharpe > best_sharpe else "🟡" if sharpe > 0.637 else "❌"
    print(f"   {strategy:30s}: {sharpe:.3f} ({improvement:+.1f}%) {symbol}")

    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_strategy = strategy

print(f"\n🏆 Best Strategy: {best_strategy}")
print(f"   CV Sharpe: {best_sharpe:.3f}")
print(f"   Expected Public (8.6x): {best_sharpe * 8.6:.2f}")

# ============================================================================
# 7. Save Results
# ============================================================================
print(f"\n[7] Saving results...")

Path("experiments/029/results").mkdir(parents=True, exist_ok=True)

# Save all strategies
df_s1.to_csv("experiments/029/results/strategy1_multiseed.csv", index=False)
df_s2.to_csv("experiments/029/results/strategy2_feature_ensemble.csv", index=False)
df_s3.to_csv("experiments/029/results/strategy3_model_ensemble.csv", index=False)

# Summary
summary = pd.DataFrame([{
    'strategy': k,
    'cv_sharpe': v,
    'expected_public': v * 8.6,
} for k, v in results_summary.items()])

summary.to_csv("experiments/029/results/summary.csv", index=False)
print(f"   Results saved to: experiments/029/results/")

print("\n" + "="*80)
print("EXP-029 Complete!")
print("="*80)
