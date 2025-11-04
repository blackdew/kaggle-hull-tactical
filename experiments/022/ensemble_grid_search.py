#!/usr/bin/env python3
"""
EXP-022: Ensemble Models Grid Search

전략: XGBoost + LightGBM ensemble
목표: Model diversity로 성능 개선
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
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
print("EXP-022: Ensemble Models Grid Search")
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

# Define model configurations
print("[4] Training base models (5-fold CV)...")
print()

K = 250
alphas = [0.1, 0.5, 0.9]
tscv = TimeSeriesSplit(n_splits=5)

# Store all fold predictions for ensemble testing
all_fold_predictions = {
    'xgb': {'q10': [], 'q50': [], 'q90': [], 'indices': []},
    'lgb': {'q10': [], 'q50': [], 'q90': [], 'indices': []},
}

all_fold_data = []

for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
    print(f"[Fold {fold_idx}/5] Training base models...")
    start_time = time.time()

    X_tr = X.iloc[tr_idx]
    y_tr = y[tr_idx]
    X_va = X.iloc[va_idx]
    y_va = y[va_idx]

    # Scale
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)

    # Store fold data
    all_fold_data.append({
        'va_idx': va_idx,
        'va_fwd': fwd_returns.iloc[va_idx].values,
        'va_rf': risk_free.iloc[va_idx].values,
        'scaler': scaler,
        'X_tr_scaled': X_tr_scaled,
        'y_tr': y_tr,
        'X_va_scaled': X_va_scaled,
    })

    # Train XGBoost quantiles
    for q_idx, alpha in enumerate(alphas):
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
            n_jobs=-1,
            verbosity=0
        )
        xgb.fit(X_tr_scaled, y_tr)
        pred = xgb.predict(X_va_scaled)

        if q_idx == 0:
            all_fold_predictions['xgb']['q10'].append(pred)
        elif q_idx == 1:
            all_fold_predictions['xgb']['q50'].append(pred)
        else:
            all_fold_predictions['xgb']['q90'].append(pred)

    # Train LightGBM quantiles
    for q_idx, alpha in enumerate(alphas):
        lgb = LGBMRegressor(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.025,
            subsample=1.0,
            colsample_bytree=0.6,
            reg_lambda=0.5,
            objective='quantile',
            alpha=alpha,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        lgb.fit(X_tr_scaled, y_tr)
        pred = lgb.predict(X_va_scaled)

        if q_idx == 0:
            all_fold_predictions['lgb']['q10'].append(pred)
        elif q_idx == 1:
            all_fold_predictions['lgb']['q50'].append(pred)
        else:
            all_fold_predictions['lgb']['q90'].append(pred)

    elapsed = time.time() - start_time
    print(f"   Completed in {elapsed:.1f}s")

print()
print("=" * 80)
print("Base Models Trained! Starting Ensemble Grid Search...")
print("=" * 80)
print()

# Define ensemble configurations
model_combinations = [
    {'name': 'XGB-only', 'models': ['xgb']},
    {'name': 'LGB-only', 'models': ['lgb']},
    {'name': 'XGB+LGB', 'models': ['xgb', 'lgb']},
]

ensemble_methods = [
    {'name': 'Equal-Avg', 'type': 'equal'},
    {'name': 'Weighted-CV', 'type': 'weighted'},
    {'name': 'Best-q50', 'type': 'best_median'},
]

total_configs = len(model_combinations) * len(ensemble_methods)
current = 0
results = []

for comb in model_combinations:
    for method in ensemble_methods:
        current += 1
        config_name = f"{comb['name']} + {method['name']}"

        print(f"[{current}/{total_configs}] Testing: {config_name}")
        start_time = time.time()

        sharpes = []

        # Evaluate on each fold
        for fold_idx in range(5):
            fold_data = all_fold_data[fold_idx]

            # Get predictions from selected models
            model_preds = []
            for model_name in comb['models']:
                q10 = all_fold_predictions[model_name]['q10'][fold_idx]
                q50 = all_fold_predictions[model_name]['q50'][fold_idx]
                q90 = all_fold_predictions[model_name]['q90'][fold_idx]
                model_preds.append({'q10': q10, 'q50': q50, 'q90': q90})

            # Ensemble predictions
            if method['type'] == 'equal':
                # Simple average
                q10_ens = np.mean([p['q10'] for p in model_preds], axis=0)
                q50_ens = np.mean([p['q50'] for p in model_preds], axis=0)
                q90_ens = np.mean([p['q90'] for p in model_preds], axis=0)

            elif method['type'] == 'weighted':
                # Weighted by inverse std (more stable = higher weight)
                weights = []
                for p in model_preds:
                    ci_width = np.abs(p['q90'] - p['q10'])
                    weight = 1.0 / (np.mean(ci_width) + 0.001)
                    weights.append(weight)

                weights = np.array(weights)
                weights = weights / weights.sum()

                q10_ens = sum(w * p['q10'] for w, p in zip(weights, model_preds))
                q50_ens = sum(w * p['q50'] for w, p in zip(weights, model_preds))
                q90_ens = sum(w * p['q90'] for w, p in zip(weights, model_preds))

            elif method['type'] == 'best_median':
                # Use median with narrowest CI
                best_idx = 0
                min_ci = float('inf')
                for idx, p in enumerate(model_preds):
                    ci_width = np.mean(np.abs(p['q90'] - p['q10']))
                    if ci_width < min_ci:
                        min_ci = ci_width
                        best_idx = idx

                q10_ens = model_preds[best_idx]['q10']
                q50_ens = model_preds[best_idx]['q50']
                q90_ens = model_preds[best_idx]['q90']

            # Calculate confidence and position
            ci_width = q90_ens - q10_ens
            confidence = 1.0 / (np.abs(ci_width) + 0.001)
            confidence = np.clip(confidence, 0.5, 5.0)  # x5 scaling

            position = 1.0 + q50_ens * K * confidence

            # Evaluate
            va_fwd = fold_data['va_fwd']
            va_rf = fold_data['va_rf']

            sharpe = calculate_sharpe(position, va_fwd, va_rf)
            sharpes.append(sharpe)

        avg_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes)
        elapsed = time.time() - start_time

        print(f"   Sharpe: {avg_sharpe:.4f} ± {std_sharpe:.4f} ({elapsed:.1f}s)")

        results.append({
            'config_id': current,
            'model_combination': comb['name'],
            'ensemble_method': method['name'],
            'num_models': len(comb['models']),
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
print("Ensemble Grid Search Complete!")
print("=" * 80)
print()

# Save results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('avg_sharpe', ascending=False)
results_df.to_csv('experiments/022/results/ensemble_results.csv', index=False)

print("[5] Top 10 configurations:")
print()
for i, row in results_df.head(10).iterrows():
    print(f"{row['config_id']:3d}. {row['model_combination']:15s} + {row['ensemble_method']:15s}")
    print(f"     Sharpe: {row['avg_sharpe']:.4f} ± {row['std_sharpe']:.4f}")
    print()

# Best configuration
best = results_df.iloc[0]
print("=" * 80)
print("BEST CONFIGURATION:")
print("=" * 80)
print(f"Models: {best['model_combination']}")
print(f"Ensemble: {best['ensemble_method']}")
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

print("Results saved to: experiments/022/results/ensemble_results.csv")
print()
print("Next: Create submission with best ensemble configuration")
