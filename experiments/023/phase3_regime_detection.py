"""
EXP-023 Phase 3: Market Regime Detection

시장 상황별로 차별화된 position sizing을 적용한다.
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
print("EXP-023 Phase 3: Market Regime Detection")
print("="*80)

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n[1] Loading data...")
data_path = Path("data/train.csv")
df = pd.read_csv(data_path)

# ============================================================================
# 2. Feature Engineering
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

# Regime features (for regime detection)
regime_features = {
    'M4': df['M4'].values,
    'V13': df['V13'].values,
    'S2': df['S2'].values,
}

print(f"   Features: {len(all_features)}, Samples: {len(X)}")

# ============================================================================
# 3. Define Regime-Based Strategies
# ============================================================================

def detect_regime(features_dict, train_data):
    """
    Detect market regime based on percentiles from training data.
    Returns: regime code for each sample
    """
    M4 = features_dict['M4']
    V13 = features_dict['V13']
    S2 = features_dict['S2']

    # Compute percentiles from training data
    M4_train = train_data['M4']
    V13_train = train_data['V13']

    M4_p25 = np.percentile(M4_train, 25)
    M4_p75 = np.percentile(M4_train, 75)
    V13_p25 = np.percentile(V13_train, 25)
    V13_p75 = np.percentile(V13_train, 75)

    regimes = []
    for i in range(len(M4)):
        # Market trend
        if M4[i] > M4_p75:
            trend = 'bull'
        elif M4[i] < M4_p25:
            trend = 'bear'
        else:
            trend = 'neutral'

        # Volatility regime
        if V13[i] > V13_p75:
            vol = 'high'
        elif V13[i] < V13_p25:
            vol = 'low'
        else:
            vol = 'normal'

        regimes.append(f"{trend}_{vol}")

    return regimes


def regime_position(pred, regimes, K=250):
    """Apply regime-based base adjustment"""
    # Base adjustments per regime
    base_map = {
        # Bullish + low vol: aggressive long
        'bull_low': 1.3,
        'bull_normal': 1.2,
        'bull_high': 1.1,

        # Bearish + high vol: defensive
        'bear_high': 0.7,
        'bear_normal': 0.8,
        'bear_low': 0.9,

        # Neutral: balanced
        'neutral_low': 1.0,
        'neutral_normal': 1.0,
        'neutral_high': 0.9,
    }

    positions = np.zeros_like(pred)
    for i, (p, r) in enumerate(zip(pred, regimes)):
        base = base_map.get(r, 1.0)
        positions[i] = np.clip(base + p * K, 0.0, 2.0)

    return positions


def volatility_adaptive_position(pred, vol_values, K=250):
    """Volatility-adaptive K adjustment"""
    positions = np.zeros_like(pred)

    # High vol → reduce K, Low vol → increase K
    for i, (p, v) in enumerate(zip(pred, vol_values)):
        if np.isnan(v):
            v = 0.0

        # K adjustment: inverse to volatility
        # High vol (e.g., v > 0.01) → K * 0.7
        # Low vol (e.g., v < 0.005) → K * 1.3
        vol_factor = 1.0 / (1.0 + 10 * np.abs(v))  # [0.5, 1.0] range roughly
        vol_factor = np.clip(vol_factor, 0.7, 1.3)

        K_adj = K * vol_factor
        positions[i] = np.clip(1.0 + p * K_adj, 0.0, 2.0)

    return positions


def sentiment_adaptive_position(pred, sentiment_values, K=250):
    """Sentiment-adaptive position sizing"""
    positions = np.zeros_like(pred)

    for i, (p, s) in enumerate(zip(pred, sentiment_values)):
        if np.isnan(s):
            s = 0.0

        # Positive sentiment → favor long
        # Negative sentiment → favor short
        if s > 0.5:  # Very positive
            base = 1.1
        elif s < -0.5:  # Very negative
            base = 0.9
        else:
            base = 1.0

        positions[i] = np.clip(base + p * K, 0.0, 2.0)

    return positions


# ============================================================================
# 4. Test Regime Strategies
# ============================================================================
print("\n[3] Testing regime-based strategies...")

strategies = [
    {'name': 'Baseline', 'type': 'baseline'},
    {'name': 'Regime-Bull/Bear-Vol', 'type': 'regime'},
    {'name': 'Volatility-Adaptive-K', 'type': 'vol_adaptive'},
    {'name': 'Sentiment-Adaptive', 'type': 'sentiment'},
]

tscv = TimeSeriesSplit(n_splits=5)
results = []

for strategy in strategies:
    print(f"\n--- {strategy['name']} ---")

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Regime features
        M4_train, M4_val = regime_features['M4'][train_idx], regime_features['M4'][val_idx]
        V13_train, V13_val = regime_features['V13'][train_idx], regime_features['V13'][val_idx]
        S2_train, S2_val = regime_features['S2'][train_idx], regime_features['S2'][val_idx]

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

        # Apply strategy
        if strategy['type'] == 'baseline':
            positions = np.clip(1.0 + y_pred * 250, 0.0, 2.0)

        elif strategy['type'] == 'regime':
            train_data = {'M4': M4_train, 'V13': V13_train}
            val_features = {'M4': M4_val, 'V13': V13_val, 'S2': S2_val}
            regimes = detect_regime(val_features, train_data)
            positions = regime_position(y_pred, regimes, K=250)

        elif strategy['type'] == 'vol_adaptive':
            positions = volatility_adaptive_position(y_pred, V13_val, K=250)

        elif strategy['type'] == 'sentiment':
            positions = sentiment_adaptive_position(y_pred, S2_val, K=250)

        # Metrics
        portfolio_returns = positions * y_val
        sharpe = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8) * np.sqrt(252)

        pos_mean = np.mean(positions)
        pos_std = np.std(positions)
        profit_proxy = np.sum(np.abs(positions - 1.0))

        fold_results.append({
            'strategy': strategy['name'],
            'fold': fold_idx + 1,
            'sharpe': sharpe,
            'pos_mean': pos_mean,
            'pos_std': pos_std,
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
        'profit_proxy': df_fold['profit_proxy'].mean(),
    }
    results.append(avg_result)

    print(f"   Sharpe: {avg_result['sharpe_mean']:.3f} ± {avg_result['sharpe_std']:.3f}")
    print(f"   Pos std: {avg_result['pos_std_mean']:.3f}")
    print(f"   Profit: {avg_result['profit_proxy']:.1f}")

# ============================================================================
# 5. Save and Analyze
# ============================================================================
print("\n[4] Saving results...")

df_results = pd.DataFrame(results)
output_path = Path("experiments/023/results/phase3_regime_strategies.csv")
df_results.to_csv(output_path, index=False)

print(f"   Saved to: {output_path}")

print("\n[5] Best Strategy")
print("="*80)

baseline_row = df_results[df_results['strategy'] == 'Baseline'].iloc[0]

print(f"\n📊 Baseline:")
print(f"   Sharpe: {baseline_row['sharpe_mean']:.3f}")
print(f"   Profit: {baseline_row['profit_proxy']:.1f}")

best_row = df_results.loc[df_results['profit_proxy'].idxmax()]

print(f"\n🏆 Best: {best_row['strategy']}")
print(f"   Sharpe: {best_row['sharpe_mean']:.3f} ({(best_row['sharpe_mean']-baseline_row['sharpe_mean'])/baseline_row['sharpe_mean']*100:+.1f}%)")
print(f"   Profit: {best_row['profit_proxy']:.1f} ({(best_row['profit_proxy']-baseline_row['profit_proxy'])/baseline_row['profit_proxy']*100:+.1f}%)")

print("\n" + "="*80)
print("Phase 3 Complete!")
print("="*80)
