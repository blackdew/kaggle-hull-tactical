"""
EXP-034: MAE Loss + Combined Improvements

Combine best approaches:
1. MAE (L1) loss - proven +10.4% improvement
2. Different feature sets (EXP-021 30 vs EXP-028 105)
3. Rank-based target engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from scipy.stats import rankdata
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
print("EXP-034: MAE Loss + Combined Improvements")
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

# ============================================================================
# 2. Prepare Feature Sets
# ============================================================================
print("\n[2] Preparing feature sets...")

# Feature Set 1: EXP-021 (30 features)
print("   Loading EXP-021 features (30)...")
top_30_df = pd.read_csv("experiments/016/results/top_30_with_interactions.csv")
top_30 = top_30_df['feature'].tolist()
top_20 = pd.read_csv("experiments/016/results/top_20_features.csv")['feature'].tolist()

# Create base features
X_base = df[top_20].copy()
feature_names = top_20.copy()

# Add interactions
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

# Add polynomials
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

X_021 = X_all_features[top_30].copy()
X_021 = X_021.fillna(0).replace([np.inf, -np.inf], 0)

# Feature Set 2: EXP-028 (105 features)
print("   Loading EXP-028 features (105)...")
exp028_features = pd.read_csv("experiments/028/results/top105_features.csv")
features_028 = exp028_features['feature'].tolist()

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
# 3. Baseline: EXP-021 + MAE
# ============================================================================
print("\n[3] Baseline: EXP-021 (30 features) + MAE Loss")

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
    'objective': 'reg:absoluteerror',  # MAE
}

tscv = TimeSeriesSplit(n_splits=5)
baseline_results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_021), 1):
    print(f"   Fold {fold_idx}/5...", end=" ")

    X_train, X_val = X_021.iloc[train_idx], X_021.iloc[val_idx]
    y_train, y_val = y_excess[train_idx], y_excess[val_idx]
    fwd_ret_val = fwd_returns[val_idx]
    rf_val = risk_free[val_idx]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train
    model = xgb.XGBRegressor(**xgb_params_base)
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
# 4. Strategy 1: EXP-028 (105 features) + MAE
# ============================================================================
print("\n[4] Strategy 1: EXP-028 (105 features) + MAE Loss")

strategy1_results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_028), 1):
    print(f"   Fold {fold_idx}/5...", end=" ")

    X_train, X_val = X_028.iloc[train_idx], X_028.iloc[val_idx]
    y_train, y_val = y_excess[train_idx], y_excess[val_idx]
    fwd_ret_val = fwd_returns[val_idx]
    rf_val = risk_free[val_idx]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train
    model = xgb.XGBRegressor(**xgb_params_base)
    model.fit(X_train_scaled, y_train, verbose=False)

    # Predict
    y_pred = model.predict(X_val_scaled)

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
# 5. Strategy 2: EXP-021 + MAE + Rank Target
# ============================================================================
print("\n[5] Strategy 2: EXP-021 (30 features) + MAE + Rank-Based Target")

strategy2_results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_021), 1):
    print(f"   Fold {fold_idx}/5...", end=" ")

    X_train, X_val = X_021.iloc[train_idx], X_021.iloc[val_idx]
    y_train, y_val = y_excess[train_idx], y_excess[val_idx]
    fwd_ret_val = fwd_returns[val_idx]
    rf_val = risk_free[val_idx]

    # Convert to rank
    y_train_rank = rankdata(y_train, method='average') / len(y_train)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train on rank
    model = xgb.XGBRegressor(**xgb_params_base)
    model.fit(X_train_scaled, y_train_rank, verbose=False)

    # Predict rank
    y_pred_rank = model.predict(X_val_scaled)
    y_pred_rank = np.clip(y_pred_rank, 0.0, 1.0)

    # Convert back using quantile mapping
    y_train_sorted = np.sort(y_train)
    indices = (y_pred_rank * (len(y_train_sorted) - 1)).astype(int)
    y_pred = y_train_sorted[indices]

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
# 6. Strategy 3: EXP-028 + MAE + Rank Target
# ============================================================================
print("\n[6] Strategy 3: EXP-028 (105 features) + MAE + Rank-Based Target")

strategy3_results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_028), 1):
    print(f"   Fold {fold_idx}/5...", end=" ")

    X_train, X_val = X_028.iloc[train_idx], X_028.iloc[val_idx]
    y_train, y_val = y_excess[train_idx], y_excess[val_idx]
    fwd_ret_val = fwd_returns[val_idx]
    rf_val = risk_free[val_idx]

    # Convert to rank
    y_train_rank = rankdata(y_train, method='average') / len(y_train)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train on rank
    model = xgb.XGBRegressor(**xgb_params_base)
    model.fit(X_train_scaled, y_train_rank, verbose=False)

    # Predict rank
    y_pred_rank = model.predict(X_val_scaled)
    y_pred_rank = np.clip(y_pred_rank, 0.0, 1.0)

    # Convert back using quantile mapping
    y_train_sorted = np.sort(y_train)
    indices = (y_pred_rank * (len(y_train_sorted) - 1)).astype(int)
    y_pred = y_train_sorted[indices]

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
# 7. Results Comparison
# ============================================================================
print("\n[7] Results Comparison")
print("="*80)

results_summary = {
    'Baseline (021+MAE)': baseline_sharpe,
    'Strategy 1 (028+MAE)': s1_sharpe,
    'Strategy 2 (021+MAE+Rank)': s2_sharpe,
    'Strategy 3 (028+MAE+Rank)': s3_sharpe,
}

print(f"\n📊 All Strategies:")
best_sharpe = baseline_sharpe
best_strategy = 'Baseline'

for strategy, sharpe in results_summary.items():
    improvement = (sharpe - baseline_sharpe) / baseline_sharpe * 100 if strategy != 'Baseline (021+MAE)' else 0
    symbol = "✅" if sharpe > baseline_sharpe else "❌" if strategy != 'Baseline (021+MAE)' else "🔵"
    print(f"   {strategy:30s}: {sharpe:.3f} ({improvement:+.1f}%) {symbol}")

    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_strategy = strategy

print(f"\n🏆 Best Strategy: {best_strategy}")
print(f"   CV Sharpe: {best_sharpe:.3f}")
print(f"   vs EXP-021 (0.637): {(best_sharpe - 0.637) / 0.637 * 100:+.1f}%")
print(f"   vs EXP-033 MAE (0.650): {(best_sharpe - 0.650) / 0.650 * 100:+.1f}%")
print(f"   Expected Public (8.6x): {best_sharpe * 8.6:.2f}")

# ============================================================================
# 8. Save Results
# ============================================================================
print(f"\n[8] Saving results...")

Path("experiments/034/results").mkdir(parents=True, exist_ok=True)

# Save all results
df_baseline.to_csv("experiments/034/results/baseline_021_mae.csv", index=False)
df_s1.to_csv("experiments/034/results/strategy1_028_mae.csv", index=False)
df_s2.to_csv("experiments/034/results/strategy2_021_mae_rank.csv", index=False)
df_s3.to_csv("experiments/034/results/strategy3_028_mae_rank.csv", index=False)

# Summary
summary_data = []
for strategy, sharpe in results_summary.items():
    improvement = (sharpe - baseline_sharpe) / baseline_sharpe * 100 if strategy != 'Baseline (021+MAE)' else 0
    summary_data.append({
        'strategy': strategy,
        'cv_sharpe': sharpe,
        'improvement_pct': improvement,
        'vs_exp021': (sharpe - 0.637) / 0.637 * 100,
        'expected_public': sharpe * 8.6,
    })

summary = pd.DataFrame(summary_data)
summary.to_csv("experiments/034/results/summary.csv", index=False)
print(f"   Results saved to: experiments/034/results/")

print("\n" + "="*80)
print("EXP-034 Complete!")
print("="*80)
