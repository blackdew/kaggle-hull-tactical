"""
EXP-024 Phase 4: Model Training with 305 Features

강한 regularization으로 overfitting 방지
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
print("EXP-024 Phase 4: Model Training with 305 Features")
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

# Load all 305 features
feature_list = pd.read_csv("experiments/024/results/phase3_selected_features_all.csv")
all_features = feature_list['feature'].tolist()

print(f"   Features: {len(all_features)}")
print(f"   Samples: {len(df)}")
print(f"   Ratio: {len(df) / len(all_features):.1f} samples per feature")

# ============================================================================
# 2. Prepare Feature Matrix
# ============================================================================
print("\n[2] Preparing feature matrix...")

# Load engineered features
engineered_data = pd.read_parquet("experiments/024/results/phase2_engineered_features.parquet")

# Get original features
phase2_list = pd.read_csv("experiments/024/results/phase2_feature_list.csv")
original_features = phase2_list[phase2_list['type'] == 'original']['feature'].tolist()
original_data = df[original_features].copy()

# Combine
X_all = pd.concat([original_data, engineered_data], axis=1)
X_all = X_all[all_features]

# Fill missing
X_all = X_all.fillna(0).replace([np.inf, -np.inf], 0)

print(f"   Feature matrix: {X_all.shape}")

# ============================================================================
# 3. Model Training with Strong Regularization
# ============================================================================
print("\n[3] Training models with 5-fold CV...")
print("   Using STRONG regularization to prevent overfitting")

K = 250  # From EXP-016

# XGBoost with strong regularization
xgb_params = {
    'n_estimators': 100,  # Reduced from 150
    'max_depth': 4,  # Reduced from 7 (strong regularization!)
    'learning_rate': 0.01,  # Reduced from 0.025
    'subsample': 0.7,  # Reduced from 1.0
    'colsample_bytree': 0.3,  # Reduced from 0.6 (use 30% features per tree)
    'colsample_bylevel': 0.5,
    'reg_alpha': 1.0,  # L1 regularization
    'reg_lambda': 5.0,  # L2 regularization (increased)
    'min_child_weight': 3,  # Prevent overfitting
    'gamma': 0.1,  # Minimum loss reduction
    'random_state': 42,
    'n_jobs': -1,
}

print(f"\n   XGBoost parameters:")
for k, v in xgb_params.items():
    print(f"      {k}: {v}")

tscv = TimeSeriesSplit(n_splits=5)
results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_all), 1):
    print(f"\n   Fold {fold_idx}/5...")

    X_train, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
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

    # Stats
    pos_mean = np.mean(positions)
    pos_std = np.std(positions)
    pred_std = np.std(y_pred)

    print(f"      Sharpe: {sharpe:.3f}")
    print(f"      Position: {pos_mean:.3f} ± {pos_std:.3f}")
    print(f"      Pred std: {pred_std:.6f}")

    results.append({
        'fold': fold_idx,
        'sharpe': sharpe,
        'pos_mean': pos_mean,
        'pos_std': pos_std,
        'pred_std': pred_std,
    })

# ============================================================================
# 4. Results Analysis
# ============================================================================
print("\n[4] Results analysis")
print("="*80)

df_results = pd.DataFrame(results)

cv_sharpe_mean = df_results['sharpe'].mean()
cv_sharpe_std = df_results['sharpe'].std()
cv_sharpe_min = df_results['sharpe'].min()
cv_sharpe_max = df_results['sharpe'].max()

print(f"\n📊 Cross-Validation Results:")
print(f"   CV Sharpe: {cv_sharpe_mean:.3f} ± {cv_sharpe_std:.3f}")
print(f"   Range: [{cv_sharpe_min:.3f}, {cv_sharpe_max:.3f}]")
print(f"   Fold details:")
for i, row in df_results.iterrows():
    print(f"      Fold {row['fold']}: {row['sharpe']:.3f}")

# ============================================================================
# 5. Comparison with Baselines
# ============================================================================
print(f"\n[5] Comparison with baselines")
print("="*80)

baselines = {
    'EXP-016': 0.559,
    'EXP-020': 0.637,
    'EXP-021': 0.637,
    'EXP-023': 0.597,
}

print(f"\n   EXP-024: {cv_sharpe_mean:.3f}")
for exp, sharpe in baselines.items():
    improvement = (cv_sharpe_mean - sharpe) / sharpe * 100
    symbol = "✅" if improvement > 0 else "❌"
    print(f"   vs {exp} ({sharpe:.3f}): {improvement:+.1f}% {symbol}")

# ============================================================================
# 6. Expected Public Score
# ============================================================================
print(f"\n[6] Expected Public Score")
print("="*80)

# Historical CV-to-Public ratios
ratios = {
    'EXP-016': 7.9,  # CV 0.559 → Public 4.44
    'EXP-021': 9.2,  # CV 0.637 → Public 5.87
}

print(f"\n   CV Sharpe: {cv_sharpe_mean:.3f}")
print(f"\n   Expected Public Score:")
for exp, ratio in ratios.items():
    expected = cv_sharpe_mean * ratio
    print(f"      Using {exp} ratio ({ratio}x): {expected:.2f}")

avg_ratio = np.mean(list(ratios.values()))
expected_avg = cv_sharpe_mean * avg_ratio
print(f"      Average ratio ({avg_ratio:.1f}x): {expected_avg:.2f}")

# Conservative estimate (accounting for possible overfitting)
conservative_ratio = 6.0
expected_conservative = cv_sharpe_mean * conservative_ratio
print(f"      Conservative (6.0x): {expected_conservative:.2f}")

# ============================================================================
# 7. Save Results
# ============================================================================
print(f"\n[7] Saving results...")

output_path = Path("experiments/024/results/phase4_cv_results.csv")
df_results.to_csv(output_path, index=False)
print(f"   Saved to: {output_path}")

# Summary
summary = {
    'n_features': len(all_features),
    'cv_sharpe_mean': cv_sharpe_mean,
    'cv_sharpe_std': cv_sharpe_std,
    'cv_sharpe_min': cv_sharpe_min,
    'cv_sharpe_max': cv_sharpe_max,
    'expected_public_conservative': expected_conservative,
    'expected_public_avg': expected_avg,
}

summary_df = pd.DataFrame([summary])
summary_path = Path("experiments/024/results/phase4_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"   Summary saved to: {summary_path}")

# ============================================================================
# 8. Final Recommendations
# ============================================================================
print(f"\n[8] Final Assessment")
print("="*80)

# Success criteria
if cv_sharpe_mean > 0.8:
    print(f"\n✅ SUCCESS: CV Sharpe {cv_sharpe_mean:.3f} > 0.8")
    print(f"   Expected Public: {expected_avg:.2f}")
    print(f"   → Proceed to Kaggle submission")
elif cv_sharpe_mean > 0.7:
    print(f"\n🟡 GOOD: CV Sharpe {cv_sharpe_mean:.3f} > 0.7")
    print(f"   Expected Public: {expected_avg:.2f}")
    print(f"   → Consider submission or further tuning")
elif cv_sharpe_mean > 0.6:
    print(f"\n🟠 MARGINAL: CV Sharpe {cv_sharpe_mean:.3f} > 0.6")
    print(f"   Expected Public: {expected_avg:.2f}")
    print(f"   → Similar to baseline, limited improvement")
else:
    print(f"\n❌ UNDERPERFORMING: CV Sharpe {cv_sharpe_mean:.3f} < 0.6")
    print(f"   Expected Public: {expected_avg:.2f}")
    print(f"   → Worse than baseline")

# Overfitting check
fold_std = df_results['sharpe'].std()
if fold_std > 0.4:
    print(f"\n⚠️  HIGH VARIANCE: Fold std = {fold_std:.3f}")
    print(f"   → Possible overfitting or unstable model")
    print(f"   → Consider: More regularization, fewer features")
elif fold_std > 0.3:
    print(f"\n⚠️  MODERATE VARIANCE: Fold std = {fold_std:.3f}")
    print(f"   → Some instability, acceptable")
else:
    print(f"\n✅ LOW VARIANCE: Fold std = {fold_std:.3f}")
    print(f"   → Stable model")

# Feature engineering effectiveness
baseline_sharpe = 0.637  # EXP-021
fe_improvement = (cv_sharpe_mean - baseline_sharpe) / baseline_sharpe * 100
if fe_improvement > 20:
    print(f"\n🎉 EXCELLENT IMPROVEMENT: +{fe_improvement:.1f}%")
    print(f"   → Feature engineering very effective!")
elif fe_improvement > 10:
    print(f"\n✅ GOOD IMPROVEMENT: +{fe_improvement:.1f}%")
    print(f"   → Feature engineering effective")
elif fe_improvement > 0:
    print(f"\n🟡 MINOR IMPROVEMENT: +{fe_improvement:.1f}%")
    print(f"   → Some benefit from new features")
else:
    print(f"\n❌ NO IMPROVEMENT: {fe_improvement:.1f}%")
    print(f"   → New features not helpful")

print("\n" + "="*80)
print("Phase 4 Complete!")
print("="*80)
