"""
EXP-023 Phase 2: Position Amplification

[0, 2] 범위를 전체 활용하여 profit 극대화
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EXP-023 Phase 2: Position Amplification")
print("="*80)

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n[1] Loading data...")
data_path = Path("data/train.csv")
df = pd.read_csv(data_path)

# ============================================================================
# 2. Feature Engineering (same as Phase 1)
# ============================================================================
print("\n[2] Creating features...")

base_features = ['M4', 'V13', 'V7', 'P8', 'S2', 'P7', 'S5', 'M2', 'V9', 'P5',
                 'I2', 'E19', 'M3', 'P10', 'V10', 'P12', 'E12', 'M12', 'P11', 'S8']

interactions = [
    ('P8', 'S2', 'mul'), ('M4', 'V7', 'mul'), ('P8', 'P7', 'div'),
    ('V7', 'P7', 'mul'), ('M4', 'S2', 'div'), ('S2', 'S5', 'mul'),
    ('S5', 'P7', 'div'), ('M4', 'P8', 'mul'), ('M4', 'M4', 'pow2'),
    ('V13', 'V13', 'pow2'),
]

interaction_features = []
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

all_features = base_features + interaction_features
for col in all_features:
    df[col] = df[col].fillna(0.0)

X = df[all_features].values
y = df['market_forward_excess_returns'].values

print(f"   Features: {len(all_features)}, Samples: {len(X)}")

# ============================================================================
# 3. Define Position Strategies
# ============================================================================

def baseline_position(pred, K=250):
    """Baseline: clip(1.0 + pred * K, 0, 2)"""
    return np.clip(1.0 + pred * K, 0.0, 2.0)


def sigmoid_position(pred, K=250, slope=10):
    """Sigmoid: 2 / (1 + exp(-slope * pred * K))"""
    signal = pred * K
    return 2.0 / (1.0 + np.exp(-slope * signal))


def quantile_binary_position(pred, K=250, threshold_pct=90):
    """Quantile-based: Binary 0/2 for extreme, clip(1+pred*K) for middle"""
    positions = np.ones_like(pred)

    # Use training data statistics for threshold (will be fit per fold)
    # For now, use local percentile
    threshold_high = np.percentile(np.abs(pred), threshold_pct)

    for i, p in enumerate(pred):
        if np.abs(p) > threshold_high:
            positions[i] = 2.0 if p > 0 else 0.0
        else:
            positions[i] = np.clip(1.0 + p * K, 0.0, 2.0)

    return positions


def threshold_amplification_position(pred, K=250, threshold=0.3, amp_factor=2.0):
    """Threshold: Amplify position when |signal| > threshold"""
    signal = pred * K
    amp = np.where(np.abs(signal) > threshold, amp_factor, 1.0)
    return np.clip(1.0 + signal * amp, 0.0, 2.0)


# ============================================================================
# 4. Test All Strategies
# ============================================================================
print("\n[3] Testing position strategies...")

strategies = [
    # Baseline
    {'name': 'Baseline', 'func': baseline_position, 'params': {'K': 250}},

    # Sigmoid amplification
    {'name': 'Sigmoid-5', 'func': sigmoid_position, 'params': {'K': 250, 'slope': 5}},
    {'name': 'Sigmoid-10', 'func': sigmoid_position, 'params': {'K': 250, 'slope': 10}},
    {'name': 'Sigmoid-15', 'func': sigmoid_position, 'params': {'K': 250, 'slope': 15}},
    {'name': 'Sigmoid-20', 'func': sigmoid_position, 'params': {'K': 250, 'slope': 20}},

    # Quantile-based binary
    {'name': 'Quantile-80', 'func': quantile_binary_position, 'params': {'K': 250, 'threshold_pct': 80}},
    {'name': 'Quantile-85', 'func': quantile_binary_position, 'params': {'K': 250, 'threshold_pct': 85}},
    {'name': 'Quantile-90', 'func': quantile_binary_position, 'params': {'K': 250, 'threshold_pct': 90}},
    {'name': 'Quantile-95', 'func': quantile_binary_position, 'params': {'K': 250, 'threshold_pct': 95}},

    # Threshold amplification
    {'name': 'Threshold-0.2-1.5x', 'func': threshold_amplification_position, 'params': {'K': 250, 'threshold': 0.2, 'amp_factor': 1.5}},
    {'name': 'Threshold-0.2-2.0x', 'func': threshold_amplification_position, 'params': {'K': 250, 'threshold': 0.2, 'amp_factor': 2.0}},
    {'name': 'Threshold-0.3-1.5x', 'func': threshold_amplification_position, 'params': {'K': 250, 'threshold': 0.3, 'amp_factor': 1.5}},
    {'name': 'Threshold-0.3-2.0x', 'func': threshold_amplification_position, 'params': {'K': 250, 'threshold': 0.3, 'amp_factor': 2.0}},
    {'name': 'Threshold-0.5-2.0x', 'func': threshold_amplification_position, 'params': {'K': 250, 'threshold': 0.5, 'amp_factor': 2.0}},
]

tscv = TimeSeriesSplit(n_splits=5)
results = []

for strategy in strategies:
    print(f"\n--- {strategy['name']} ---")

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train
        model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.025,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train, verbose=False)

        # Predict
        y_pred = model.predict(X_val_scaled)

        # Apply position strategy
        positions = strategy['func'](y_pred, **strategy['params'])

        # Metrics
        portfolio_returns = positions * y_val
        sharpe = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8) * np.sqrt(252)

        pos_mean = np.mean(positions)
        pos_std = np.std(positions)
        pos_min = np.min(positions)
        pos_max = np.max(positions)

        pct_extreme_low = np.sum(positions < 0.2) / len(positions) * 100
        pct_extreme_high = np.sum(positions > 1.8) / len(positions) * 100

        profit_proxy = np.sum(np.abs(positions - 1.0))

        fold_results.append({
            'strategy': strategy['name'],
            'fold': fold_idx + 1,
            'sharpe': sharpe,
            'pos_mean': pos_mean,
            'pos_std': pos_std,
            'pos_min': pos_min,
            'pos_max': pos_max,
            'pct_extreme_low': pct_extreme_low,
            'pct_extreme_high': pct_extreme_high,
            'profit_proxy': profit_proxy,
        })

    # Aggregate
    df_fold = pd.DataFrame(fold_results)
    avg_result = {
        'strategy': strategy['name'],
        'sharpe_mean': df_fold['sharpe'].mean(),
        'sharpe_std': df_fold['sharpe'].std(),
        'pos_mean': df_fold['pos_mean'].mean(),
        'pos_std_mean': df_fold['pos_std'].mean(),
        'pos_range': df_fold['pos_max'].mean() - df_fold['pos_min'].mean(),
        'pct_extreme_low': df_fold['pct_extreme_low'].mean(),
        'pct_extreme_high': df_fold['pct_extreme_high'].mean(),
        'pct_extreme_total': df_fold['pct_extreme_low'].mean() + df_fold['pct_extreme_high'].mean(),
        'profit_proxy': df_fold['profit_proxy'].mean(),
    }
    results.append(avg_result)

    print(f"   Sharpe: {avg_result['sharpe_mean']:.3f} ± {avg_result['sharpe_std']:.3f}")
    print(f"   Pos: {avg_result['pos_mean']:.3f} (std: {avg_result['pos_std_mean']:.3f})")
    print(f"   Extremes: {avg_result['pct_extreme_total']:.1f}%")
    print(f"   Profit proxy: {avg_result['profit_proxy']:.1f}")

# ============================================================================
# 5. Save and Analyze
# ============================================================================
print("\n[4] Saving results...")

df_results = pd.DataFrame(results)
output_path = Path("experiments/023/results/phase2_position_strategies.csv")
df_results.to_csv(output_path, index=False)

print(f"   Saved to: {output_path}")

# ============================================================================
# 6. Best Strategy Selection
# ============================================================================
print("\n[5] Best Strategy Selection")
print("="*80)

baseline_row = df_results[df_results['strategy'] == 'Baseline'].iloc[0]

print(f"\n📊 Baseline:")
print(f"   Sharpe: {baseline_row['sharpe_mean']:.3f}")
print(f"   Pos std: {baseline_row['pos_std_mean']:.3f}")
print(f"   Extremes: {baseline_row['pct_extreme_total']:.1f}%")
print(f"   Profit proxy: {baseline_row['profit_proxy']:.1f}")

# Sort by profit proxy
df_sorted = df_results.sort_values('profit_proxy', ascending=False)

print(f"\n🏆 Top 5 by Profit Proxy:")
for i, row in df_sorted.head(5).iterrows():
    sharpe_change = (row['sharpe_mean'] - baseline_row['sharpe_mean']) / baseline_row['sharpe_mean'] * 100
    profit_change = (row['profit_proxy'] - baseline_row['profit_proxy']) / baseline_row['profit_proxy'] * 100

    print(f"\n   {i+1}. {row['strategy']}")
    print(f"      Sharpe: {row['sharpe_mean']:.3f} ({sharpe_change:+.1f}%)")
    print(f"      Pos std: {row['pos_std_mean']:.3f}")
    print(f"      Extremes: {row['pct_extreme_total']:.1f}%")
    print(f"      Profit: {row['profit_proxy']:.1f} ({profit_change:+.1f}%)")

# Best strategy with Sharpe > 0.5
df_valid = df_results[df_results['sharpe_mean'] >= 0.5]
if len(df_valid) > 0:
    best_row = df_valid.loc[df_valid['profit_proxy'].idxmax()]

    print(f"\n✅ Recommended Strategy: {best_row['strategy']}")
    print(f"   Sharpe: {best_row['sharpe_mean']:.3f} (baseline: {baseline_row['sharpe_mean']:.3f})")
    print(f"   Profit: {best_row['profit_proxy']:.1f} (baseline: {baseline_row['profit_proxy']:.1f})")
    print(f"   Improvement: {(best_row['profit_proxy'] - baseline_row['profit_proxy']) / baseline_row['profit_proxy'] * 100:+.1f}%")
else:
    print(f"\n⚠️  No strategy with Sharpe >= 0.5")
    print(f"   Consider less aggressive parameters")

print("\n" + "="*80)
print("Phase 2 Complete! Next: phase3_regime_detection.py")
print("="*80)
