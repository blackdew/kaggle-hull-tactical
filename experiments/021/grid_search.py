#!/usr/bin/env python3
"""
EXP-021: Quantile Regression Grid Search

전략: Quantile alpha와 Confidence scaling의 모든 조합 테스트
목표: 최고 성능 configuration 찾기
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from itertools import product
import time

def calculate_sharpe(positions, fwd_returns, risk_free):
    """Calculate Sharpe ratio from positions"""
    positions = np.clip(positions, 0.0, 2.0)
    strategy_returns = risk_free * (1.0 - positions) + fwd_returns * positions
    excess_returns = strategy_returns - risk_free

    if np.std(strategy_returns) > 0:
        sharpe = (np.mean(excess_returns) / np.std(strategy_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    return sharpe


print("=" * 80)
print("EXP-021: Quantile Regression Grid Search")
print("=" * 80)
print()

# Load data
print("[1] Loading data...")
train = pd.read_csv("data/train.csv")
y = train['market_forward_excess_returns'].values
fwd_returns = train['forward_returns'].copy()
risk_free = train['risk_free_rate'].copy()

# Load features (EXP-016 Top 30)
print("[2] Loading features...")
top_30_df = pd.read_csv("experiments/016/results/top_30_with_interactions.csv")
top_30 = top_30_df['feature'].tolist()
top_20 = pd.read_csv("experiments/016/results/top_20_features.csv")['feature'].tolist()

# Recreate features
print("[3] Creating features...")
X_base = train[top_20].fillna(train[top_20].median()).replace([np.inf, -np.inf], np.nan).fillna(0)

interactions = []
feature_names = top_20.copy()
top_10 = top_20[:10]
eps = 1e-8

for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i + 1 :]:
        interactions.append(X_base[feat1] * X_base[feat2])
        feature_names.append(f'{feat1}*{feat2}')

for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i + 1 :]:
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
print()

# Define grid search space
print("[4] Defining grid search space...")

# Quantile alpha configurations
quantile_configs = [
    {'name': 'Wide (0.05-0.95)', 'alphas': [0.05, 0.50, 0.95]},
    {'name': 'EXP-020 (0.1-0.9)', 'alphas': [0.10, 0.50, 0.90]},
    {'name': 'Moderate (0.15-0.85)', 'alphas': [0.15, 0.50, 0.85]},
    {'name': 'Narrow (0.2-0.8)', 'alphas': [0.20, 0.50, 0.80]},
    {'name': '5Q-Wide', 'alphas': [0.05, 0.25, 0.50, 0.75, 0.95]},
    {'name': '5Q-Moderate', 'alphas': [0.10, 0.30, 0.50, 0.70, 0.90]},
    {'name': 'IQR (0.25-0.75)', 'alphas': [0.25, 0.50, 0.75]},
]

# Confidence scaling (symmetric)
symmetric_scalings = [
    {'name': 'No-scale (x1)', 'scale': 1.0, 'type': 'symmetric'},
    {'name': 'Conservative (x2)', 'scale': 2.0, 'type': 'symmetric'},
    {'name': 'Moderate (x3)', 'scale': 3.0, 'type': 'symmetric'},
    {'name': 'EXP-020 (x5)', 'scale': 5.0, 'type': 'symmetric'},
    {'name': 'Aggressive (x7)', 'scale': 7.0, 'type': 'symmetric'},
    {'name': 'VeryAggr (x10)', 'scale': 10.0, 'type': 'symmetric'},
]

# Asymmetric scalings
asymmetric_scalings = [
    {'name': 'Asym-Conservative', 'upside': 2.0, 'downside': 1.0, 'type': 'asymmetric'},
    {'name': 'Asym-Moderate', 'upside': 5.0, 'downside': 3.0, 'type': 'asymmetric'},
    {'name': 'Asym-Aggressive', 'upside': 7.0, 'downside': 5.0, 'type': 'asymmetric'},
    {'name': 'Asym-VeryAggr', 'upside': 10.0, 'downside': 5.0, 'type': 'asymmetric'},
    {'name': 'Defensive', 'upside': 3.0, 'downside': 5.0, 'type': 'asymmetric'},
    {'name': 'RiskAverse', 'upside': 5.0, 'downside': 7.0, 'type': 'asymmetric'},
    {'name': 'Bearish', 'upside': 1.0, 'downside': 2.0, 'type': 'asymmetric'},
    {'name': 'ModBearish', 'upside': 2.0, 'downside': 3.0, 'type': 'asymmetric'},
]

all_scalings = symmetric_scalings + asymmetric_scalings

print(f"   Quantile configs: {len(quantile_configs)}")
print(f"   Scaling configs: {len(all_scalings)}")
print(f"   Total combinations: {len(quantile_configs) * len(all_scalings)}")
print()

# Grid search
print("[5] Running grid search...")
print()

K = 250  # From EXP-016
results = []
total_configs = len(quantile_configs) * len(all_scalings)
current = 0

for q_config in quantile_configs:
    for s_config in all_scalings:
        current += 1
        config_name = f"{q_config['name']} + {s_config['name']}"

        print(f"[{current}/{total_configs}] Testing: {config_name}")
        start_time = time.time()

        alphas = q_config['alphas']
        n_quantiles = len(alphas)

        sharpes = []
        tscv = TimeSeriesSplit(n_splits=5)

        for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
            X_tr = X.iloc[tr_idx]
            y_tr = y[tr_idx]
            X_va = X.iloc[va_idx]
            y_va = y[va_idx]

            # Scale
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_va_scaled = scaler.transform(X_va)

            # Train quantile models
            models = []
            for alpha in alphas:
                xgb = XGBRegressor(
                    n_estimators=150,
                    max_depth=7,
                    learning_rate=0.025,
                    subsample=1.0,
                    colsample_bytree=0.6,
                    reg_lambda=0.5,
                    objective='reg:quantileerror',
                    quantile_alpha=alpha,
                    random_state=42,
                    n_jobs=-1
                )
                xgb.fit(X_tr_scaled, y_tr)
                models.append(xgb)

            # Predict
            predictions = [model.predict(X_va_scaled) for model in models]

            # Calculate confidence and position
            if n_quantiles == 3:
                q_low, q_mid, q_high = predictions
            elif n_quantiles == 5:
                q_low, q_mid, q_high = predictions[0], predictions[2], predictions[4]
            else:
                q_low, q_mid, q_high = predictions[0], predictions[1], predictions[2]

            # Confidence interval
            ci_width = q_high - q_low
            confidence = 1.0 / (np.abs(ci_width) + 0.001)

            # Apply scaling
            if s_config['type'] == 'symmetric':
                confidence = np.clip(confidence, 0.5, s_config['scale'])
            else:  # asymmetric
                # Separate upside and downside
                upside_mask = q_mid > 0
                downside_mask = q_mid <= 0

                confidence_scaled = confidence.copy()
                confidence_scaled[upside_mask] = np.clip(
                    confidence[upside_mask], 0.5, s_config['upside']
                )
                confidence_scaled[downside_mask] = np.clip(
                    confidence[downside_mask], 0.5, s_config['downside']
                )
                confidence = confidence_scaled

            # Position
            position = 1.0 + q_mid * K * confidence

            # Evaluate
            va_fwd = fwd_returns.iloc[va_idx].values
            va_rf = risk_free.iloc[va_idx].values

            sharpe = calculate_sharpe(position, va_fwd, va_rf)
            sharpes.append(sharpe)

        avg_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes)
        elapsed = time.time() - start_time

        print(f"   Sharpe: {avg_sharpe:.4f} ± {std_sharpe:.4f} ({elapsed:.1f}s)")

        results.append({
            'config_id': current,
            'quantile_config': q_config['name'],
            'quantile_alphas': str(alphas),
            'scaling_config': s_config['name'],
            'scaling_type': s_config['type'],
            'scaling_value': s_config.get('scale', f"up={s_config.get('upside')},down={s_config.get('downside')}"),
            'avg_sharpe': avg_sharpe,
            'std_sharpe': std_sharpe,
            'fold1': sharpes[0],
            'fold2': sharpes[1],
            'fold3': sharpes[2],
            'fold4': sharpes[3],
            'fold5': sharpes[4],
        })

print()
print("=" * 80)
print("Grid Search Complete!")
print("=" * 80)
print()

# Save results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('avg_sharpe', ascending=False)
results_df.to_csv('experiments/021/results/grid_search_results.csv', index=False)

print("[6] Top 10 configurations:")
print()
for i, row in results_df.head(10).iterrows():
    print(f"{row['config_id']:3d}. {row['quantile_config']:20s} + {row['scaling_config']:20s}")
    print(f"     Sharpe: {row['avg_sharpe']:.4f} ± {row['std_sharpe']:.4f}")
    print()

# Best configuration
best = results_df.iloc[0]
print("=" * 80)
print("BEST CONFIGURATION:")
print("=" * 80)
print(f"Quantile: {best['quantile_config']} {best['quantile_alphas']}")
print(f"Scaling: {best['scaling_config']} ({best['scaling_value']})")
print(f"Sharpe: {best['avg_sharpe']:.4f} ± {best['std_sharpe']:.4f}")
print()
print(f"Improvement vs EXP-020 (0.6368): {((best['avg_sharpe'] - 0.6368) / 0.6368 * 100):+.2f}%")
print()

# Expected Public Score
exp020_cv = 0.6368
exp020_public = 5.872
cv_to_public_ratio = exp020_public / exp020_cv

expected_public = best['avg_sharpe'] * cv_to_public_ratio

print(f"Expected Public Score: {expected_public:.2f} (EXP-020 ratio {cv_to_public_ratio:.2f}x)")
print()

print("Results saved to: experiments/021/results/grid_search_results.csv")
print()
print("Next: Create submission with best configuration")
