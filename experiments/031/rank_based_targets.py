"""
EXP-031: Rank-Based Target Engineering

Current: Predict raw excess returns directly
New: Predict rank of returns, then convert back

Hypothesis: Rank prediction is easier and more stable than raw value prediction
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
print("EXP-031: Rank-Based Target Engineering")
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
print(f"   Target range: [{y_excess.min():.4f}, {y_excess.max():.4f}]")

# ============================================================================
# 2. Baseline (Raw value prediction)
# ============================================================================
print("\n[2] Baseline: Raw Value Prediction (EXP-021 style)")

K = 250
xgb_params = {
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

    # Train
    model = xgb.XGBRegressor(**xgb_params)
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
        'pos_mean': np.mean(positions),
        'pos_std': np.std(positions),
    })

df_baseline = pd.DataFrame(baseline_results)
baseline_sharpe = df_baseline['sharpe'].mean()
print(f"\n   Baseline CV Sharpe: {baseline_sharpe:.3f} ± {df_baseline['sharpe'].std():.3f}")

# ============================================================================
# 3. Strategy 1: Rank-Based Prediction (Percentile)
# ============================================================================
print("\n[3] Strategy 1: Rank-Based Prediction (Percentile Transform)")

strategy1_results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    print(f"   Fold {fold_idx}/5...", end=" ")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_excess[train_idx], y_excess[val_idx]
    fwd_ret_val = fwd_returns[val_idx]
    rf_val = risk_free[val_idx]

    # Convert target to rank (0~1 scale)
    # Use percentile rank: 0 = lowest, 1 = highest
    y_train_rank = rankdata(y_train, method='average') / len(y_train)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train on rank
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train_scaled, y_train_rank, verbose=False)

    # Predict rank
    y_pred_rank = model.predict(X_val_scaled)
    y_pred_rank = np.clip(y_pred_rank, 0.0, 1.0)  # Ensure valid rank range

    # Convert rank back to value scale using training statistics
    # Map rank to z-score, then to value
    from scipy.stats import norm
    z_scores = norm.ppf(np.clip(y_pred_rank, 0.01, 0.99))  # Avoid extreme values
    y_pred = y_train.mean() + z_scores * y_train.std()

    # Positions
    positions = np.clip(1.0 + y_pred * K, 0.0, 2.0)

    # Sharpe
    sharpe = calculate_sharpe(positions, fwd_ret_val, rf_val)

    print(f"Sharpe: {sharpe:.3f}")

    strategy1_results.append({
        'fold': fold_idx,
        'sharpe': sharpe,
        'pos_mean': np.mean(positions),
        'pos_std': np.std(positions),
    })

df_s1 = pd.DataFrame(strategy1_results)
s1_sharpe = df_s1['sharpe'].mean()
s1_improvement = (s1_sharpe - baseline_sharpe) / baseline_sharpe * 100
s1_symbol = "✅" if s1_sharpe > baseline_sharpe else "❌"
print(f"\n   Strategy 1 CV Sharpe: {s1_sharpe:.3f} ± {df_s1['sharpe'].std():.3f} ({s1_improvement:+.1f}%) {s1_symbol}")

# ============================================================================
# 4. Strategy 2: Quantile-Based Prediction
# ============================================================================
print("\n[4] Strategy 2: Direct Quantile Mapping")

strategy2_results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    print(f"   Fold {fold_idx}/5...", end=" ")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_excess[train_idx], y_excess[val_idx]
    fwd_ret_val = fwd_returns[val_idx]
    rf_val = risk_free[val_idx]

    # Convert target to rank (0~1 scale)
    y_train_rank = rankdata(y_train, method='average') / len(y_train)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train on rank
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train_scaled, y_train_rank, verbose=False)

    # Predict rank
    y_pred_rank = model.predict(X_val_scaled)
    y_pred_rank = np.clip(y_pred_rank, 0.0, 1.0)

    # Convert rank to value using quantile mapping
    # For each predicted rank, find corresponding value in training distribution
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
        'pos_mean': np.mean(positions),
        'pos_std': np.std(positions),
    })

df_s2 = pd.DataFrame(strategy2_results)
s2_sharpe = df_s2['sharpe'].mean()
s2_improvement = (s2_sharpe - baseline_sharpe) / baseline_sharpe * 100
s2_symbol = "✅" if s2_sharpe > baseline_sharpe else "❌"
print(f"\n   Strategy 2 CV Sharpe: {s2_sharpe:.3f} ± {df_s2['sharpe'].std():.3f} ({s2_improvement:+.1f}%) {s2_symbol}")

# ============================================================================
# 5. Strategy 3: Log-Transformed Target
# ============================================================================
print("\n[5] Strategy 3: Log-Transformed Target")

strategy3_results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    print(f"   Fold {fold_idx}/5...", end=" ")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_excess[train_idx], y_excess[val_idx]
    fwd_ret_val = fwd_returns[val_idx]
    rf_val = risk_free[val_idx]

    # Shift and log-transform target (handle negative values)
    y_min = y_train.min()
    shift = abs(y_min) + 0.001 if y_min < 0 else 0.001
    y_train_log = np.log(y_train + shift)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train on log-transformed target
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train_scaled, y_train_log, verbose=False)

    # Predict and inverse transform
    y_pred_log = model.predict(X_val_scaled)
    y_pred = np.exp(y_pred_log) - shift

    # Positions
    positions = np.clip(1.0 + y_pred * K, 0.0, 2.0)

    # Sharpe
    sharpe = calculate_sharpe(positions, fwd_ret_val, rf_val)

    print(f"Sharpe: {sharpe:.3f}")

    strategy3_results.append({
        'fold': fold_idx,
        'sharpe': sharpe,
        'pos_mean': np.mean(positions),
        'pos_std': np.std(positions),
    })

df_s3 = pd.DataFrame(strategy3_results)
s3_sharpe = df_s3['sharpe'].mean()
s3_improvement = (s3_sharpe - baseline_sharpe) / baseline_sharpe * 100
s3_symbol = "✅" if s3_sharpe > baseline_sharpe else "❌"
print(f"\n   Strategy 3 CV Sharpe: {s3_sharpe:.3f} ± {df_s3['sharpe'].std():.3f} ({s3_improvement:+.1f}%) {s3_symbol}")

# ============================================================================
# 6. Results Comparison
# ============================================================================
print("\n[6] Results Comparison")
print("="*80)

results_summary = {
    'Baseline (Raw)': baseline_sharpe,
    'Strategy 1 (Rank-Percentile)': s1_sharpe,
    'Strategy 2 (Rank-Quantile)': s2_sharpe,
    'Strategy 3 (Log-Transform)': s3_sharpe,
}

print(f"\n📊 All Strategies:")
best_sharpe = baseline_sharpe
best_strategy = 'Baseline'

for strategy, sharpe in results_summary.items():
    improvement = (sharpe - baseline_sharpe) / baseline_sharpe * 100 if strategy != 'Baseline (Raw)' else 0
    symbol = "✅" if sharpe > baseline_sharpe else "❌" if strategy != 'Baseline (Raw)' else "🔵"
    print(f"   {strategy:30s}: {sharpe:.3f} ({improvement:+.1f}%) {symbol}")

    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_strategy = strategy

print(f"\n🏆 Best Strategy: {best_strategy}")
print(f"   CV Sharpe: {best_sharpe:.3f}")
print(f"   vs EXP-021 (0.637): {(best_sharpe - 0.637) / 0.637 * 100:+.1f}%")
print(f"   Expected Public (8.6x): {best_sharpe * 8.6:.2f}")

# ============================================================================
# 7. Save Results
# ============================================================================
print(f"\n[7] Saving results...")

Path("experiments/031/results").mkdir(parents=True, exist_ok=True)

# Save all results
df_baseline.to_csv("experiments/031/results/baseline.csv", index=False)
df_s1.to_csv("experiments/031/results/strategy1_rank_percentile.csv", index=False)
df_s2.to_csv("experiments/031/results/strategy2_rank_quantile.csv", index=False)
df_s3.to_csv("experiments/031/results/strategy3_log_transform.csv", index=False)

# Summary
summary_data = []
for strategy, sharpe in results_summary.items():
    improvement = (sharpe - baseline_sharpe) / baseline_sharpe * 100 if strategy != 'Baseline (Raw)' else 0
    summary_data.append({
        'strategy': strategy,
        'cv_sharpe': sharpe,
        'improvement_pct': improvement,
        'expected_public': sharpe * 8.6,
    })

summary = pd.DataFrame(summary_data)
summary.to_csv("experiments/031/results/summary.csv", index=False)
print(f"   Results saved to: experiments/031/results/")

print("\n" + "="*80)
print("EXP-031 Complete!")
print("="*80)
