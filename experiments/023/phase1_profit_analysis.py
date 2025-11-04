"""
EXP-023 Phase 1: Profit Analysis

현재 position strategy의 profit bottleneck을 파악한다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EXP-023 Phase 1: Profit Analysis")
print("="*80)

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n[1] Loading data...")
data_path = Path("data/train.csv")
df = pd.read_csv(data_path)
print(f"   Data shape: {df.shape}")

# ============================================================================
# 2. Feature Engineering (EXP-016 style)
# ============================================================================
print("\n[2] Creating features (EXP-016 baseline)...")

# EXP-016 Top 30 features (simplified)
base_features = ['M4', 'V13', 'V7', 'P8', 'S2', 'P7', 'S5', 'M2', 'V9', 'P5',
                 'I2', 'E19', 'M3', 'P10', 'V10', 'P12', 'E12', 'M12', 'P11', 'S8']

# Create interaction features
interaction_features = []

# Top 10 interactions from EXP-016
interactions = [
    ('P8', 'S2', 'mul'),
    ('M4', 'V7', 'mul'),
    ('P8', 'P7', 'div'),
    ('V7', 'P7', 'mul'),
    ('M4', 'S2', 'div'),
    ('S2', 'S5', 'mul'),
    ('S5', 'P7', 'div'),
    ('M4', 'P8', 'mul'),
    ('M4', 'M4', 'pow2'),
    ('V13', 'V13', 'pow2'),
]

for f1, f2, op in interactions:
    if op == 'mul':
        fname = f"{f1}*{f2}"
        df[fname] = df[f1] * df[f2]
    elif op == 'div':
        fname = f"{f1}/{f2}"
        df[fname] = df[f1] / (df[f2] + 1e-8)
    elif op == 'pow2':
        fname = f"{f1}²"
        df[fname] = df[f1] ** 2
    interaction_features.append(fname)

# Fill missing values
all_features = base_features + interaction_features
for col in all_features:
    df[col] = df[col].fillna(0.0)

print(f"   Features: {len(all_features)}")

# ============================================================================
# 3. Prepare X, y
# ============================================================================
X = df[all_features].values
y = df['market_forward_excess_returns'].values
date_ids = df['date_id'].values

print(f"   X shape: {X.shape}, y shape: {y.shape}")

# ============================================================================
# 4. Cross-Validation with Position Analysis
# ============================================================================
print("\n[3] Running CV with position analysis...")

tscv = TimeSeriesSplit(n_splits=5)
K_values = [50, 250, 500, 1000]

results = []

for k_val in K_values:
    print(f"\n--- K = {k_val} ---")

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train XGBoost
        model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.025,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_val_scaled)

        # Calculate positions
        positions = np.clip(1.0 + y_pred * k_val, 0.0, 2.0)

        # Calculate returns
        portfolio_returns = positions * y_val

        # Calculate metrics
        sharpe = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8) * np.sqrt(252)

        # Position statistics
        pos_mean = np.mean(positions)
        pos_std = np.std(positions)
        pos_min = np.min(positions)
        pos_max = np.max(positions)
        pos_q25 = np.percentile(positions, 25)
        pos_q75 = np.percentile(positions, 75)

        # Count extreme positions
        n_extreme_low = np.sum(positions < 0.2)
        n_extreme_high = np.sum(positions > 1.8)
        pct_extreme_low = n_extreme_low / len(positions) * 100
        pct_extreme_high = n_extreme_high / len(positions) * 100

        # Profit proxy (sum of absolute deviations from 1.0)
        profit_proxy = np.sum(np.abs(positions - 1.0))

        # Position-Return correlation
        pos_return_corr = np.corrcoef(positions, portfolio_returns)[0, 1]

        fold_results.append({
            'k': k_val,
            'fold': fold_idx + 1,
            'sharpe': sharpe,
            'pos_mean': pos_mean,
            'pos_std': pos_std,
            'pos_min': pos_min,
            'pos_max': pos_max,
            'pos_q25': pos_q25,
            'pos_q75': pos_q75,
            'pct_extreme_low': pct_extreme_low,
            'pct_extreme_high': pct_extreme_high,
            'profit_proxy': profit_proxy,
            'pos_return_corr': pos_return_corr,
        })

        print(f"   Fold {fold_idx+1}: Sharpe={sharpe:.3f}, Pos={pos_mean:.3f}±{pos_std:.3f}, "
              f"Range=[{pos_min:.3f}, {pos_max:.3f}], Extremes={pct_extreme_low:.1f}%/{pct_extreme_high:.1f}%")

    # Aggregate
    df_fold = pd.DataFrame(fold_results)
    avg_result = {
        'k': k_val,
        'sharpe_mean': df_fold['sharpe'].mean(),
        'sharpe_std': df_fold['sharpe'].std(),
        'pos_mean': df_fold['pos_mean'].mean(),
        'pos_std_mean': df_fold['pos_std'].mean(),
        'pos_range_mean': df_fold['pos_max'].mean() - df_fold['pos_min'].mean(),
        'pct_extreme_low': df_fold['pct_extreme_low'].mean(),
        'pct_extreme_high': df_fold['pct_extreme_high'].mean(),
        'profit_proxy_mean': df_fold['profit_proxy'].mean(),
        'pos_return_corr_mean': df_fold['pos_return_corr'].mean(),
    }
    results.append(avg_result)

    print(f"   Average: Sharpe={avg_result['sharpe_mean']:.3f}±{avg_result['sharpe_std']:.3f}, "
          f"Pos={avg_result['pos_mean']:.3f}, Std={avg_result['pos_std_mean']:.3f}")

# ============================================================================
# 5. Save Results
# ============================================================================
print("\n[4] Saving results...")

df_results = pd.DataFrame(results)
output_path = Path("experiments/023/results/phase1_profit_analysis.csv")
df_results.to_csv(output_path, index=False)

print(f"   Results saved to: {output_path}")

# ============================================================================
# 6. Analysis and Recommendations
# ============================================================================
print("\n[5] Analysis and Recommendations")
print("="*80)

# Find current best (k=250, from EXP-016)
baseline = df_results[df_results['k'] == 250].iloc[0]

print(f"\n📊 Baseline (K=250, EXP-016):")
print(f"   Sharpe: {baseline['sharpe_mean']:.3f} ± {baseline['sharpe_std']:.3f}")
print(f"   Position: {baseline['pos_mean']:.3f} (std: {baseline['pos_std_mean']:.3f})")
print(f"   Range: {baseline['pos_range_mean']:.3f}")
print(f"   Extremes: {baseline['pct_extreme_low']:.1f}% (low) / {baseline['pct_extreme_high']:.1f}% (high)")
print(f"   Profit proxy: {baseline['profit_proxy_mean']:.1f}")

print(f"\n🎯 Bottleneck 분석:")

# Position 분포가 너무 좁음
if baseline['pos_std_mean'] < 0.3:
    print(f"   ⚠️  Position std가 {baseline['pos_std_mean']:.3f}로 매우 낮음")
    print(f"       → [0, 2] 범위 중 ~{baseline['pos_std_mean']/2*100:.0f}%만 활용")
    print(f"       → Position을 더 극단적으로 만들어야 함 (target std: 0.4+)")

# Extreme position 비율이 낮음
if baseline['pct_extreme_low'] + baseline['pct_extreme_high'] < 10:
    print(f"   ⚠️  Extreme position이 {baseline['pct_extreme_low'] + baseline['pct_extreme_high']:.1f}%만")
    print(f"       → 대부분 시장 중립 포지션 (0.9~1.1)")
    print(f"       → 확신 있는 예측에 더 큰 포지션 필요")

# k가 너무 작음
if baseline['pos_std_mean'] < df_results['pos_std_mean'].max() * 0.7:
    best_k = df_results.loc[df_results['profit_proxy_mean'].idxmax(), 'k']
    print(f"   💡 K={best_k}일 때 profit proxy 최대")
    print(f"       → K를 조정하는 것만으로는 부족 (EXP-006 교훈)")
    print(f"       → Position formula 자체를 변경해야 함")

print(f"\n✅ Phase 2 목표:")
print(f"   1. Position std: {baseline['pos_std_mean']:.3f} → 0.4~0.5 (+50~100%)")
print(f"   2. Extreme positions: {baseline['pct_extreme_low'] + baseline['pct_extreme_high']:.1f}% → 20~30%")
print(f"   3. Profit proxy: {baseline['profit_proxy_mean']:.1f} → {baseline['profit_proxy_mean']*1.5:.1f}+ (+50%)")
print(f"   4. Sharpe 하락: < 30% (acceptable)")

print(f"\n📋 추천 전략 (Phase 2):")
print(f"   - Sigmoid amplification (slope=10~20)")
print(f"   - Quantile-based binary (threshold=90th)")
print(f"   - Threshold amplification (threshold=0.3)")

print("\n" + "="*80)
print("Phase 1 Complete! Next: phase2_position_amplification.py")
print("="*80)
