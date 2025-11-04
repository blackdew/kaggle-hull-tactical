"""
EXP-032: Time-Based Features

Add temporal features based on date_id to capture time-dependent patterns.
All features are single-row calculable (InferenceServer compatible).
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

def add_time_features(df):
    """Add time-based features from date_id"""
    df_time = df.copy()

    # Trend feature (normalized)
    df_time['time_trend'] = df_time['date_id'] / df_time['date_id'].max()

    # Periodic features (weekly, monthly cycles)
    # Assume ~252 trading days per year
    df_time['day_of_week_approx'] = df_time['date_id'] % 5  # 5-day trading week
    df_time['week_of_month'] = (df_time['date_id'] % 21) // 5  # ~21 trading days per month
    df_time['month_of_quarter'] = (df_time['date_id'] % 63) // 21  # ~63 trading days per quarter

    # Cyclical encoding (sin/cos for periodicity)
    # Weekly cycle
    df_time['week_sin'] = np.sin(2 * np.pi * df_time['date_id'] / 5)
    df_time['week_cos'] = np.cos(2 * np.pi * df_time['date_id'] / 5)

    # Monthly cycle
    df_time['month_sin'] = np.sin(2 * np.pi * df_time['date_id'] / 21)
    df_time['month_cos'] = np.cos(2 * np.pi * df_time['date_id'] / 21)

    # Quarterly cycle
    df_time['quarter_sin'] = np.sin(2 * np.pi * df_time['date_id'] / 63)
    df_time['quarter_cos'] = np.cos(2 * np.pi * df_time['date_id'] / 63)

    # Yearly cycle
    df_time['year_sin'] = np.sin(2 * np.pi * df_time['date_id'] / 252)
    df_time['year_cos'] = np.cos(2 * np.pi * df_time['date_id'] / 252)

    time_feature_names = [
        'time_trend', 'day_of_week_approx', 'week_of_month', 'month_of_quarter',
        'week_sin', 'week_cos', 'month_sin', 'month_cos',
        'quarter_sin', 'quarter_cos', 'year_sin', 'year_cos'
    ]

    return df_time, time_feature_names

print("="*80)
print("EXP-032: Time-Based Features")
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

# Add time features
df_time, time_feature_names = add_time_features(df)

print(f"   Added {len(time_feature_names)} time features:")
for feat in time_feature_names:
    print(f"      - {feat}")

# Load EXP-021 features
print("\n   Loading EXP-021 features...")
top_30_df = pd.read_csv("experiments/016/results/top_30_with_interactions.csv")
top_30 = top_30_df['feature'].tolist()
top_20 = pd.read_csv("experiments/016/results/top_20_features.csv")['feature'].tolist()

# Create base features
X_base = df_time[top_20].copy()
feature_names = top_20.copy()

# Add interactions (from top 10 base features)
interactions = []
top_10 = top_20[:10]
for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i+1:]:
        interactions.append(df_time[feat1] * df_time[feat2])
        feature_names.append(f'{feat1}*{feat2}')

for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i+1:]:
        interactions.append(df_time[feat1] / (df_time[feat2].abs() + 1e-8))
        feature_names.append(f'{feat1}/{feat2}')

# Add polynomials (from top 5 base features)
top_5 = top_20[:5]
for feat in top_5:
    interactions.append(df_time[feat] ** 2)
    feature_names.append(f'{feat}²')
    interactions.append(df_time[feat] ** 3)
    feature_names.append(f'{feat}³')

X_all_features = pd.concat(
    [X_base] + [pd.Series(feat, name=name) for feat, name in zip(interactions, feature_names[len(top_20):])],
    axis=1
)
X_all_features.columns = feature_names

# Select only top_30 features (baseline)
X_baseline = X_all_features[top_30].copy()
X_baseline = X_baseline.fillna(0).replace([np.inf, -np.inf], 0)

print(f"\n   Baseline features: {X_baseline.shape}")

# ============================================================================
# 2. Baseline (EXP-021 features only)
# ============================================================================
print("\n[2] Baseline: EXP-021 Features Only")

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

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_baseline), 1):
    print(f"   Fold {fold_idx}/5...", end=" ")

    X_train, X_val = X_baseline.iloc[train_idx], X_baseline.iloc[val_idx]
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
    })

df_baseline = pd.DataFrame(baseline_results)
baseline_sharpe = df_baseline['sharpe'].mean()
print(f"\n   Baseline CV Sharpe: {baseline_sharpe:.3f} ± {df_baseline['sharpe'].std():.3f}")

# ============================================================================
# 3. Strategy 1: Add ALL time features
# ============================================================================
print("\n[3] Strategy 1: EXP-021 Features + All Time Features")

# Combine baseline features with time features
X_with_time = pd.concat([X_baseline, df_time[time_feature_names]], axis=1)
X_with_time = X_with_time.fillna(0).replace([np.inf, -np.inf], 0)

print(f"   Total features: {X_with_time.shape[1]} ({X_baseline.shape[1]} + {len(time_feature_names)})")

strategy1_results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_with_time), 1):
    print(f"   Fold {fold_idx}/5...", end=" ")

    X_train, X_val = X_with_time.iloc[train_idx], X_with_time.iloc[val_idx]
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
# 4. Strategy 2: Select best time features by importance
# ============================================================================
print("\n[4] Strategy 2: Select Best Time Features")

# Train model on all features to get feature importance
print("   Training model to assess feature importance...")
X_train_full = X_with_time.iloc[:int(0.8*len(X_with_time))]
y_train_full = y_excess[:int(0.8*len(y_excess))]

scaler_full = StandardScaler()
X_train_full_scaled = scaler_full.fit_transform(X_train_full)

model_full = xgb.XGBRegressor(**xgb_params)
model_full.fit(X_train_full_scaled, y_train_full, verbose=False)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X_with_time.columns,
    'importance': model_full.feature_importances_
}).sort_values('importance', ascending=False)

# Identify top time features
time_features_importance = feature_importance[feature_importance['feature'].isin(time_feature_names)]
print(f"\n   Time feature importance:")
for _, row in time_features_importance.head(5).iterrows():
    print(f"      {row['feature']:20s}: {row['importance']:.4f}")

# Select top 3 time features
top_time_features = time_features_importance.head(3)['feature'].tolist()
print(f"\n   Selected top 3 time features: {top_time_features}")

# Create feature set with top time features
X_selective = pd.concat([X_baseline, df_time[top_time_features]], axis=1)
X_selective = X_selective.fillna(0).replace([np.inf, -np.inf], 0)

strategy2_results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_selective), 1):
    print(f"   Fold {fold_idx}/5...", end=" ")

    X_train, X_val = X_selective.iloc[train_idx], X_selective.iloc[val_idx]
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
# 5. Results Comparison
# ============================================================================
print("\n[5] Results Comparison")
print("="*80)

results_summary = {
    'Baseline (EXP-021 only)': baseline_sharpe,
    'Strategy 1 (+ All time features)': s1_sharpe,
    'Strategy 2 (+ Top 3 time features)': s2_sharpe,
}

print(f"\n📊 All Strategies:")
best_sharpe = baseline_sharpe
best_strategy = 'Baseline'

for strategy, sharpe in results_summary.items():
    improvement = (sharpe - baseline_sharpe) / baseline_sharpe * 100 if strategy != 'Baseline (EXP-021 only)' else 0
    symbol = "✅" if sharpe > baseline_sharpe else "❌" if strategy != 'Baseline (EXP-021 only)' else "🔵"
    print(f"   {strategy:35s}: {sharpe:.3f} ({improvement:+.1f}%) {symbol}")

    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_strategy = strategy

print(f"\n🏆 Best Strategy: {best_strategy}")
print(f"   CV Sharpe: {best_sharpe:.3f}")
print(f"   vs EXP-021 (0.637): {(best_sharpe - 0.637) / 0.637 * 100:+.1f}%")
print(f"   Expected Public (8.6x): {best_sharpe * 8.6:.2f}")

# ============================================================================
# 6. Save Results
# ============================================================================
print(f"\n[6] Saving results...")

Path("experiments/032/results").mkdir(parents=True, exist_ok=True)

# Save all results
df_baseline.to_csv("experiments/032/results/baseline.csv", index=False)
df_s1.to_csv("experiments/032/results/strategy1_all_time_features.csv", index=False)
df_s2.to_csv("experiments/032/results/strategy2_top3_time_features.csv", index=False)

# Save feature importance
feature_importance.to_csv("experiments/032/results/feature_importance.csv", index=False)

# Save selected time features
pd.DataFrame({'feature': top_time_features}).to_csv("experiments/032/results/selected_time_features.csv", index=False)

# Summary
summary_data = []
for strategy, sharpe in results_summary.items():
    improvement = (sharpe - baseline_sharpe) / baseline_sharpe * 100 if strategy != 'Baseline (EXP-021 only)' else 0
    summary_data.append({
        'strategy': strategy,
        'cv_sharpe': sharpe,
        'improvement_pct': improvement,
        'expected_public': sharpe * 8.6,
    })

summary = pd.DataFrame(summary_data)
summary.to_csv("experiments/032/results/summary.csv", index=False)
print(f"   Results saved to: experiments/032/results/")

print("\n" + "="*80)
print("EXP-032 Complete!")
print("="*80)
