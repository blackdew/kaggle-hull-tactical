"""
EXP-027: Top 50 Features by MI Score

305 features 중 MI score 상위 50개만 선택
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
print("EXP-027: Top 50 Features by MI Score")
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

# Load MI scores and select top 50
mi_scores = pd.read_csv("experiments/024/results/phase3_mi_analysis_all_features.csv")
top50_features = mi_scores.head(50)['feature'].tolist()

print(f"   Total features available: {len(mi_scores)}")
print(f"   Selected top 50 features")
print(f"   MI score range: [{mi_scores.iloc[49]['mi_score']:.4f}, {mi_scores.iloc[0]['mi_score']:.4f}]")

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

# Select only top 50 features
X_top50 = X_all[top50_features].copy()

# Fill missing
X_top50 = X_top50.fillna(0).replace([np.inf, -np.inf], 0)

print(f"   Feature matrix: {X_top50.shape}")
print(f"   Samples: {len(df)}")
print(f"   Ratio: {len(df) / len(top50_features):.1f} samples per feature")

# Show top 10 features
print(f"\n   Top 10 features by MI:")
for i, row in mi_scores.head(10).iterrows():
    print(f"      {i+1}. {row['feature']:20s} (MI: {row['mi_score']:.4f}, {row['type']})")

# ============================================================================
# 3. Model Training
# ============================================================================
print("\n[3] Training XGBoost with 5-fold CV...")

K = 250

# XGBoost parameters optimized for 50 features
xgb_params = {
    'n_estimators': 150,
    'max_depth': 5,             # Shallower for fewer features
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,    # Can use more since only 50 features
    'colsample_bylevel': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_weight': 1,
    'random_state': 42,
    'n_jobs': -1,
}

print(f"   Parameters optimized for 50 features")

tscv = TimeSeriesSplit(n_splits=5)
results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_top50), 1):
    print(f"\n   Fold {fold_idx}/5...")

    X_train, X_val = X_top50.iloc[train_idx], X_top50.iloc[val_idx]
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

print(f"\n📊 Cross-Validation Results:")
print(f"   CV Sharpe: {cv_sharpe_mean:.3f} ± {cv_sharpe_std:.3f}")
print(f"   Range: [{df_results['sharpe'].min():.3f}, {df_results['sharpe'].max():.3f}]")

# ============================================================================
# 5. Comparison with Baselines
# ============================================================================
print(f"\n[5] Comparison with baselines")
print("="*80)

baselines = {
    'EXP-021 (94 features)': 0.637,
    'EXP-024 (305 over-reg)': 0.559,
    'EXP-026 (305 proper)': 0.565,
}

print(f"\n   EXP-027 (Top 50): {cv_sharpe_mean:.3f}")
for exp, sharpe in baselines.items():
    improvement = (cv_sharpe_mean - sharpe) / sharpe * 100
    symbol = "✅" if improvement > 0 else "❌"
    print(f"   vs {exp} ({sharpe:.3f}): {improvement:+.1f}% {symbol}")

# ============================================================================
# 6. Expected Public Score
# ============================================================================
print(f"\n[6] Expected Public Score")
print("="*80)

avg_ratio = 8.6
expected_avg = cv_sharpe_mean * avg_ratio
conservative_ratio = 6.0
expected_conservative = cv_sharpe_mean * conservative_ratio

print(f"\n   CV Sharpe: {cv_sharpe_mean:.3f}")
print(f"   Expected Public (avg 8.6x): {expected_avg:.2f}")
print(f"   Expected Public (conservative 6.0x): {expected_conservative:.2f}")

# ============================================================================
# 7. Save Results
# ============================================================================
print(f"\n[7] Saving results...")

Path("experiments/027/results").mkdir(parents=True, exist_ok=True)

output_path = Path("experiments/027/results/cv_results.csv")
df_results.to_csv(output_path, index=False)
print(f"   Saved to: {output_path}")

# Save selected features
selected_features_df = pd.DataFrame({
    'feature': top50_features,
    'mi_score': mi_scores.head(50)['mi_score'].values,
    'type': mi_scores.head(50)['type'].values,
})
features_path = Path("experiments/027/results/top50_features.csv")
selected_features_df.to_csv(features_path, index=False)
print(f"   Features saved to: {features_path}")

# Summary
summary = {
    'n_features': 50,
    'cv_sharpe_mean': cv_sharpe_mean,
    'cv_sharpe_std': cv_sharpe_std,
    'expected_public_avg': expected_avg,
    'expected_public_conservative': expected_conservative,
}

summary_df = pd.DataFrame([summary])
summary_path = Path("experiments/027/results/summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"   Summary saved to: {summary_path}")

# ============================================================================
# 8. Final Assessment
# ============================================================================
print(f"\n[8] Final Assessment")
print("="*80)

if cv_sharpe_mean > 0.7:
    print(f"\n🎉 SUCCESS: CV Sharpe {cv_sharpe_mean:.3f} > 0.7")
    print(f"   Expected Public: {expected_avg:.2f}")
    print(f"   → Proceed to Kaggle submission")
elif cv_sharpe_mean > 0.65:
    print(f"\n✅ GOOD: CV Sharpe {cv_sharpe_mean:.3f} > 0.65")
    print(f"   Expected Public: {expected_avg:.2f}")
    print(f"   → Competitive with EXP-021")
elif cv_sharpe_mean > 0.6:
    print(f"\n🟡 MARGINAL: CV Sharpe {cv_sharpe_mean:.3f} > 0.6")
    print(f"   Expected Public: {expected_avg:.2f}")
    print(f"   → Similar to baseline")
else:
    print(f"\n❌ UNDERPERFORMING: CV Sharpe {cv_sharpe_mean:.3f} < 0.6")
    print(f"   Expected Public: {expected_avg:.2f}")
    print(f"   → Worse than baseline")

# Variance check
if cv_sharpe_std < 0.2:
    print(f"\n✅ LOW VARIANCE: Fold std = {cv_sharpe_std:.3f}")
    print(f"   → Stable model")
elif cv_sharpe_std < 0.3:
    print(f"\n🟡 MODERATE VARIANCE: Fold std = {cv_sharpe_std:.3f}")
    print(f"   → Acceptable stability")
else:
    print(f"\n⚠️  HIGH VARIANCE: Fold std = {cv_sharpe_std:.3f}")
    print(f"   → Model unstable")

print("\n" + "="*80)
print("EXP-027 Complete!")
print("="*80)
