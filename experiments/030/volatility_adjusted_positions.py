"""
EXP-030: Volatility-Adjusted Position Sizing

Current: position = clip(1.0 + pred * K, 0.0, 2.0)
New: position = clip(1.0 + (pred / volatility) * K, 0.0, 2.0)

Hypothesis: Reduce positions during high-volatility periods to improve Sharpe ratio
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

print("="*80)
print("EXP-030: Volatility-Adjusted Position Sizing")
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

# Identify volatility features
vol_features = [f for f in top_30 if f.startswith('V')]
print(f"   Volatility features in top 30: {vol_features}")

# ============================================================================
# 2. Baseline (EXP-021 reproduction)
# ============================================================================
print("\n[2] Baseline: Fixed K=250 (EXP-021 style)")

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

    # Positions (baseline)
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
# 3. Strategy 1: Use V7 (most important vol feature) for adjustment
# ============================================================================
print("\n[3] Strategy 1: Volatility-Adjusted (using V7)")

# Test different volatility scaling factors
vol_scales = [0.5, 1.0, 2.0, 3.0, 5.0]
strategy1_results = {}

for vol_scale in vol_scales:
    print(f"\n   Testing vol_scale={vol_scale}...")

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
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

        # Get V7 values for validation set (from original data)
        v7_val = df.iloc[val_idx]['V7'].values

        # Normalize volatility (higher V7 = higher volatility)
        # Use percentile-based scaling to avoid extreme values
        v7_normalized = (v7_val - np.percentile(v7_val, 10)) / (np.percentile(v7_val, 90) - np.percentile(v7_val, 10))
        v7_normalized = np.clip(v7_normalized, 0.1, 2.0)  # Avoid division by zero and extreme values

        # Adjust predictions by volatility
        # When volatility is high, reduce position size
        vol_adjustment = 1.0 / (1.0 + v7_normalized * vol_scale)
        y_pred_adjusted = y_pred * vol_adjustment

        # Positions
        positions = np.clip(1.0 + y_pred_adjusted * K, 0.0, 2.0)

        # Sharpe
        sharpe = calculate_sharpe(positions, fwd_ret_val, rf_val)

        fold_results.append({
            'fold': fold_idx,
            'sharpe': sharpe,
            'pos_mean': np.mean(positions),
            'pos_std': np.std(positions),
        })

    df_results = pd.DataFrame(fold_results)
    cv_sharpe = df_results['sharpe'].mean()
    strategy1_results[vol_scale] = {
        'cv_sharpe': cv_sharpe,
        'cv_std': df_results['sharpe'].std(),
        'results': df_results
    }

    improvement = (cv_sharpe - baseline_sharpe) / baseline_sharpe * 100
    symbol = "✅" if cv_sharpe > baseline_sharpe else "❌"
    print(f"      CV Sharpe: {cv_sharpe:.3f} ± {df_results['sharpe'].std():.3f} ({improvement:+.1f}%) {symbol}")

# ============================================================================
# 4. Strategy 2: Use predicted volatility from separate model
# ============================================================================
print("\n[4] Strategy 2: Predicted Volatility Adjustment")

# Train a separate model to predict forward volatility
# Use realized volatility as proxy (std of recent returns)
print("   Training volatility prediction model...")

fold_results_s2 = []
for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    print(f"   Fold {fold_idx}/5...", end=" ")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_excess[train_idx], y_excess[val_idx]
    fwd_ret_val = fwd_returns[val_idx]
    rf_val = risk_free[val_idx]

    # Create volatility target (use V7 as proxy for realized vol)
    vol_train = df.iloc[train_idx]['V7'].values
    vol_val = df.iloc[val_idx]['V7'].values

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train return prediction model
    model_return = xgb.XGBRegressor(**xgb_params)
    model_return.fit(X_train_scaled, y_train, verbose=False)
    y_pred = model_return.predict(X_val_scaled)

    # Train volatility prediction model
    vol_params = xgb_params.copy()
    vol_params['objective'] = 'reg:squarederror'
    model_vol = xgb.XGBRegressor(**vol_params)
    model_vol.fit(X_train_scaled, vol_train, verbose=False)
    vol_pred = model_vol.predict(X_val_scaled)

    # Normalize predicted volatility
    vol_pred_norm = (vol_pred - np.percentile(vol_pred, 10)) / (np.percentile(vol_pred, 90) - np.percentile(vol_pred, 10))
    vol_pred_norm = np.clip(vol_pred_norm, 0.1, 2.0)

    # Adjust positions by predicted volatility (using best vol_scale from Strategy 1)
    best_vol_scale = max(strategy1_results.keys(), key=lambda k: strategy1_results[k]['cv_sharpe'])
    vol_adjustment = 1.0 / (1.0 + vol_pred_norm * best_vol_scale)
    y_pred_adjusted = y_pred * vol_adjustment

    # Positions
    positions = np.clip(1.0 + y_pred_adjusted * K, 0.0, 2.0)

    # Sharpe
    sharpe = calculate_sharpe(positions, fwd_ret_val, rf_val)

    print(f"Sharpe: {sharpe:.3f}")

    fold_results_s2.append({
        'fold': fold_idx,
        'sharpe': sharpe,
        'pos_mean': np.mean(positions),
        'pos_std': np.std(positions),
    })

df_s2 = pd.DataFrame(fold_results_s2)
s2_sharpe = df_s2['sharpe'].mean()
print(f"\n   Strategy 2 CV Sharpe: {s2_sharpe:.3f} ± {df_s2['sharpe'].std():.3f}")

# ============================================================================
# 5. Results Comparison
# ============================================================================
print("\n[5] Results Comparison")
print("="*80)

print(f"\n📊 All Strategies:")
print(f"   Baseline (Fixed K=250)        : {baseline_sharpe:.3f}")

best_s1_scale = max(strategy1_results.keys(), key=lambda k: strategy1_results[k]['cv_sharpe'])
best_s1_sharpe = strategy1_results[best_s1_scale]['cv_sharpe']
improvement_s1 = (best_s1_sharpe - baseline_sharpe) / baseline_sharpe * 100
symbol_s1 = "✅" if best_s1_sharpe > baseline_sharpe else "❌"
print(f"   Strategy 1 (V7 adjust, scale={best_s1_scale}): {best_s1_sharpe:.3f} ({improvement_s1:+.1f}%) {symbol_s1}")

improvement_s2 = (s2_sharpe - baseline_sharpe) / baseline_sharpe * 100
symbol_s2 = "✅" if s2_sharpe > baseline_sharpe else "❌"
print(f"   Strategy 2 (Predicted vol)    : {s2_sharpe:.3f} ({improvement_s2:+.1f}%) {symbol_s2}")

# Find best overall
best_sharpe = max(baseline_sharpe, best_s1_sharpe, s2_sharpe)
if best_sharpe == baseline_sharpe:
    best_strategy = "Baseline"
elif best_sharpe == best_s1_sharpe:
    best_strategy = f"Strategy 1 (vol_scale={best_s1_scale})"
else:
    best_strategy = "Strategy 2"

print(f"\n🏆 Best Strategy: {best_strategy}")
print(f"   CV Sharpe: {best_sharpe:.3f}")
print(f"   vs EXP-021 (0.637): {(best_sharpe - 0.637) / 0.637 * 100:+.1f}%")
print(f"   Expected Public (8.6x): {best_sharpe * 8.6:.2f}")

# ============================================================================
# 6. Save Results
# ============================================================================
print(f"\n[6] Saving results...")

Path("experiments/030/results").mkdir(parents=True, exist_ok=True)

# Save baseline
df_baseline.to_csv("experiments/030/results/baseline.csv", index=False)

# Save strategy 1 results
for vol_scale, data in strategy1_results.items():
    data['results'].to_csv(f"experiments/030/results/strategy1_volscale_{vol_scale}.csv", index=False)

# Save strategy 2
df_s2.to_csv("experiments/030/results/strategy2_predicted_vol.csv", index=False)

# Summary
summary_data = [
    {'strategy': 'Baseline', 'cv_sharpe': baseline_sharpe, 'cv_std': df_baseline['sharpe'].std()},
]
for vol_scale, data in strategy1_results.items():
    summary_data.append({
        'strategy': f'Strategy1_scale{vol_scale}',
        'cv_sharpe': data['cv_sharpe'],
        'cv_std': data['cv_std']
    })
summary_data.append({
    'strategy': 'Strategy2_predicted_vol',
    'cv_sharpe': s2_sharpe,
    'cv_std': df_s2['sharpe'].std()
})

summary = pd.DataFrame(summary_data)
summary['vs_baseline_pct'] = (summary['cv_sharpe'] - baseline_sharpe) / baseline_sharpe * 100
summary['expected_public'] = summary['cv_sharpe'] * 8.6
summary.to_csv("experiments/030/results/summary.csv", index=False)
print(f"   Results saved to: experiments/030/results/")

print("\n" + "="*80)
print("EXP-030 Complete!")
print("="*80)
