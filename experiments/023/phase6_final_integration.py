"""
EXP-023 Phase 6: Final Integration

EXP-020 Quantile Regression + EXP-023 Sigmoid Amplification 결합
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

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
print("EXP-023 Phase 6: Final Integration")
print("EXP-020 Quantile + EXP-023 Sigmoid-20")
print("="*80)

# Load data
print("\n[1] Loading data...")
train = pd.read_csv("data/train.csv")
y_excess = train['market_forward_excess_returns'].copy()
fwd_returns = train['forward_returns'].copy()
risk_free = train['risk_free_rate'].copy()

# Load EXP-016 features
print("\n[2] Loading features (EXP-016)...")
top_30_df = pd.read_csv("experiments/016/results/top_30_with_interactions.csv")
top_30 = top_30_df['feature'].tolist()
top_20 = pd.read_csv("experiments/016/results/top_20_features.csv")['feature'].tolist()

X_base = train[top_20].fillna(train[top_20].median()).replace([np.inf, -np.inf], np.nan).fillna(0)

# Create interactions
interactions = []
feature_names = top_20.copy()
top_10 = top_20[:10]
eps = 1e-8

for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i + 1:]:
        interactions.append(X_base[feat1] * X_base[feat2])
        feature_names.append(f'{feat1}*{feat2}')

for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i + 1:]:
        interactions.append(X_base[feat1] / (X_base[feat2].abs() + eps))
        feature_names.append(f'{feat1}/{feat2}')

top_5 = top_20[:5]
for feat in top_5:
    interactions.append(X_base[feat] ** 2)
    feature_names.append(f'{feat}²')
    interactions.append(X_base[feat] ** 3)
    feature_names.append(f'{feat}³')

X_all = pd.concat(
    [X_base] + [pd.Series(feat, name=name) for feat, name in zip(interactions, feature_names[len(top_20):])],
    axis=1
)
X_all.columns = feature_names
X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)

X = X_all[top_30].copy()
print(f"   Features: {X.shape}")

# Test final strategy
print("\n[3] Testing final strategy...")
print()

K = 250
sigmoid_slope = 20  # From Phase 2
confidence_scale = 5.0  # From EXP-020

results = []
tscv = TimeSeriesSplit(n_splits=5)

for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
    print(f"Fold {fold_idx}/5...")

    X_tr = X.iloc[tr_idx]
    y_tr = y_excess.iloc[tr_idx]
    X_va = X.iloc[va_idx]
    y_va = y_excess.iloc[va_idx]

    fwd_returns_va = fwd_returns.iloc[va_idx]
    risk_free_va = risk_free.iloc[va_idx]

    # Scale
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)

    # Train quantile models
    print("   Training quantile models...")

    xgb_q10 = XGBRegressor(
        n_estimators=150,
        max_depth=7,
        learning_rate=0.025,
        subsample=1.0,
        colsample_bytree=0.6,
        reg_lambda=0.5,
        objective='reg:quantileerror',
        quantile_alpha=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb_q10.fit(X_tr_scaled, y_tr, verbose=False)

    xgb_q50 = XGBRegressor(
        n_estimators=150,
        max_depth=7,
        learning_rate=0.025,
        subsample=1.0,
        colsample_bytree=0.6,
        reg_lambda=0.5,
        objective='reg:quantileerror',
        quantile_alpha=0.5,
        random_state=42,
        n_jobs=-1
    )
    xgb_q50.fit(X_tr_scaled, y_tr, verbose=False)

    xgb_q90 = XGBRegressor(
        n_estimators=150,
        max_depth=7,
        learning_rate=0.025,
        subsample=1.0,
        colsample_bytree=0.6,
        reg_lambda=0.5,
        objective='reg:quantileerror',
        quantile_alpha=0.9,
        random_state=42,
        n_jobs=-1
    )
    xgb_q90.fit(X_tr_scaled, y_tr, verbose=False)

    # Predict
    q10_pred = xgb_q10.predict(X_va_scaled)
    q50_pred = xgb_q50.predict(X_va_scaled)
    q90_pred = xgb_q90.predict(X_va_scaled)

    # Confidence from quantile interval
    CI = np.abs(q90_pred - q10_pred)
    confidence = 1.0 / (CI + 0.001)
    confidence = np.clip(confidence * confidence_scale, 0.5, 5.0)

    # Sigmoid amplification with confidence weighting
    signal = q50_pred * K * confidence
    positions = 2.0 / (1.0 + np.exp(-sigmoid_slope * signal))

    # Calculate metrics
    sharpe = calculate_sharpe(positions, fwd_returns_va, risk_free_va)

    pos_mean = np.mean(positions)
    pos_std = np.std(positions)
    pct_extreme_low = np.sum(positions < 0.2) / len(positions) * 100
    pct_extreme_high = np.sum(positions > 1.8) / len(positions) * 100
    profit_proxy = np.sum(np.abs(positions - 1.0))

    results.append({
        'fold': fold_idx,
        'sharpe': sharpe,
        'pos_mean': pos_mean,
        'pos_std': pos_std,
        'pct_extreme_low': pct_extreme_low,
        'pct_extreme_high': pct_extreme_high,
        'profit_proxy': profit_proxy,
    })

    print(f"   Sharpe: {sharpe:.3f}, Pos: {pos_mean:.3f}±{pos_std:.3f}, "
          f"Extremes: {pct_extreme_low:.1f}%/{pct_extreme_high:.1f}%, "
          f"Profit: {profit_proxy:.1f}")

# Results
df_results = pd.DataFrame(results)

print("\n" + "="*80)
print("Final Results:")
print("="*80)

print(f"\n📊 CV Performance (5-fold):")
print(f"   Sharpe: {df_results['sharpe'].mean():.3f} ± {df_results['sharpe'].std():.3f}")
print(f"   Range: [{df_results['sharpe'].min():.3f}, {df_results['sharpe'].max():.3f}]")

print(f"\n💰 Position Statistics:")
print(f"   Mean: {df_results['pos_mean'].mean():.3f}")
print(f"   Std: {df_results['pos_std'].mean():.3f}")
print(f"   Extreme low: {df_results['pct_extreme_low'].mean():.1f}%")
print(f"   Extreme high: {df_results['pct_extreme_high'].mean():.1f}%")
print(f"   Profit proxy: {df_results['profit_proxy'].mean():.1f}")

# Save
output_path = Path("experiments/023/results/phase6_final_cv.csv")
df_results.to_csv(output_path, index=False)
print(f"\n💾 Saved to: {output_path}")

# Expected Public Score
print(f"\n🎯 Expected Public Score:")
cv_sharpe = df_results['sharpe'].mean()

# Conservative (EXP-016 ratio 7.9x)
conservative = cv_sharpe * 7.9
# Expected (EXP-021 ratio 9.2x)
expected = cv_sharpe * 9.2
# Optimistic (10x)
optimistic = cv_sharpe * 10.0

print(f"   Conservative (7.9x): {conservative:.2f}")
print(f"   Expected (9.2x): {expected:.2f}")
print(f"   Optimistic (10x): {optimistic:.2f}")

if expected >= 20.0:
    print(f"\n✅ 목표 20점 달성 가능!")
elif expected >= 15.0:
    print(f"\n🟡 15점+ 예상 (목표 근접)")
elif expected >= 10.0:
    print(f"\n🟠 10점+ 예상 (개선 필요)")
else:
    print(f"\n🔴 10점 미만 예상 (추가 전략 필요)")

print("\n" + "="*80)
print("Phase 6 Complete! Next: Create submission file")
print("="*80)
