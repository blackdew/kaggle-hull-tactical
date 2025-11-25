#!/usr/bin/env python3
"""
EXP-039 Phase 1: Train Hybrid Ensemble
Goal: Train XGBoost, LightGBM, and CatBoost on Hybrid Features and evaluate Ensemble.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import os
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("EXP-039 Phase 1: Train Hybrid Ensemble")
print("="*80)

# 1. Load Data
print("\n[1] Loading data...")
train = pd.read_csv("data/train.csv")
y_excess = train['market_forward_excess_returns'].values
fwd_returns = train['forward_returns'].values
risk_free = train['risk_free_rate'].values

# 2. Load Hybrid Features
print("\n[2] Loading Hybrid Features...")
hybrid_path = "experiments/038_hybrid_features/results/hybrid_features.csv"
if os.path.exists(hybrid_path):
    hybrid_df = pd.read_csv(hybrid_path)
    hybrid_features = hybrid_df['feature'].tolist()
else:
    # Fallback if file missing (should not happen)
    print("[WARNING] Hybrid features file missing. Using hardcoded list.")
    hybrid_features = [
        'E12', 'E19', 'E19*S5', 'E19/P7', 'E19/S5', 'I2*E19', 'I2*S5', 'I2/S5', 'M4*E19', 'M4*I2',
        'M4*P8', 'M4*V7', 'M4/E19', 'M4/I2', 'M4/P7', 'M4/P8', 'M4/S2', 'M4/V7', 'M4²', 'P5',
        'P5/P7', 'P8*E19', 'P8*S2', 'P8/P5', 'P8/P7', 'P8/S2', 'P8²', 'S2*E19', 'S2*I2', 'S2*S5',
        'S2/P5', 'S2/P7', 'S5/P7', 'S8', 'V13*I2', 'V13*V7', 'V13/S2', 'V13/S5', 'V13/P7', 'V13²',
        'V7*E19', 'V7*I2', 'V7*P5', 'V7*P7', 'V7*P8', 'V7*S2', 'V7/I2', 'V7/P7', 'V7/P8', 'V7/S2',
        'V7/S5'
    ]

print(f"   Hybrid Features ({len(hybrid_features)}):")
print(f"   {hybrid_features}")

# 3. Create Feature Matrix
print("\n[3] Creating feature matrix...")
top_20 = [
    'M4', 'V13', 'V7', 'P8', 'S2', 'I2', 'E19', 'S5', 'P5', 'P7',
    'M2', 'V9', 'M3', 'P12', 'P10', 'V10', 'E12', 'P11', 'M12', 'S8'
]
X_base = train[top_20].fillna(train[top_20].median()).replace([np.inf, -np.inf], np.nan).fillna(0)

# Generate interactions
interactions = []
feature_names = top_20.copy()
top_10 = top_20[:10]
eps = 1e-8

for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i+1:]:
        interactions.append(X_base[feat1] * X_base[feat2])
        feature_names.append(f'{feat1}*{feat2}')

for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i+1:]:
        interactions.append(X_base[feat1] / (X_base[feat2].abs() + eps))
        feature_names.append(f'{feat1}/{feat2}')

top_5 = top_20[:5]
for feat in top_5:
    interactions.append(X_base[feat] ** 2)
    feature_names.append(f'{feat}²')
    interactions.append(X_base[feat] ** 3)
    feature_names.append(f'{feat}³')

X_all = pd.concat([X_base] + [pd.Series(feat, name=name) for feat, name in zip(interactions, feature_names[len(top_20):])], axis=1)
X_all.columns = feature_names
X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)

# Select Hybrid Features
X = X_all[hybrid_features].copy()
print(f"   Feature Matrix: {X.shape}")

# 4. Train and Evaluate (CV)
print("\n[4] Evaluating Ensemble Strategy (5-fold CV)...")

def calculate_sharpe(positions, fwd_returns, risk_free):
    positions = np.clip(positions, 0.0, 2.0)
    strategy_returns = risk_free * (1.0 - positions) + fwd_returns * positions
    excess_returns = strategy_returns - risk_free
    if np.std(strategy_returns) > 0:
        sharpe = (np.mean(excess_returns) / np.std(strategy_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0
    return sharpe

sharpes = []
tscv = TimeSeriesSplit(n_splits=5)
K = 250

for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
    print(f"\n   Fold {fold_idx}/5...")
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr = y_excess[tr_idx]
    
    # Scale
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    
    # Models
    models = {
        'xgb': {},
        'lgb': {},
        'cat': {}
    }
    
    # Quantiles to train
    quantiles = [0.1, 0.5, 0.9]
    q_names = ['q10', 'q50', 'q90']
    
    # 1. XGBoost
    print("      Training XGBoost...")
    for q, name in zip(quantiles, q_names):
        model = XGBRegressor(
            n_estimators=150, max_depth=7, learning_rate=0.025,
            subsample=1.0, colsample_bytree=0.6, reg_lambda=0.5,
            objective='reg:quantileerror', quantile_alpha=q,
            random_state=42, n_jobs=-1
        )
        model.fit(X_tr_scaled, y_tr)
        models['xgb'][name] = model

    # 2. LightGBM
    print("      Training LightGBM...")
    for q, name in zip(quantiles, q_names):
        model = LGBMRegressor(
            n_estimators=150, max_depth=7, learning_rate=0.025,
            subsample=0.8, colsample_bytree=0.6, reg_lambda=0.5,
            objective='quantile', alpha=q,
            random_state=42, n_jobs=-1, verbose=-1
        )
        model.fit(X_tr_scaled, y_tr)
        models['lgb'][name] = model

    # 3. CatBoost
    print("      Training CatBoost...")
    for q, name in zip(quantiles, q_names):
        model = CatBoostRegressor(
            iterations=150, depth=7, learning_rate=0.025,
            loss_function=f'Quantile:alpha={q}',
            random_state=42, verbose=0, thread_count=-1
        )
        model.fit(X_tr_scaled, y_tr)
        models['cat'][name] = model
        
    # Predict
    preds = {'q10': [], 'q50': [], 'q90': []}
    
    for name in q_names:
        p_xgb = models['xgb'][name].predict(X_va_scaled)
        p_lgb = models['lgb'][name].predict(X_va_scaled)
        p_cat = models['cat'][name].predict(X_va_scaled)
        
        # Simple Average Ensemble
        preds[name] = (p_xgb + p_lgb + p_cat) / 3.0
        
    # Strategy
    q10_pred = preds['q10']
    q50_pred = preds['q50']
    q90_pred = preds['q90']
    
    ci_width = q90_pred - q10_pred
    confidence = 1.0 / (np.abs(ci_width) + 0.001)
    confidence = np.clip(confidence, 0.5, 5.0)
    positions = 1.0 + q50_pred * K * confidence
    
    # Evaluate
    va_fwd = fwd_returns[va_idx]
    va_rf = risk_free[va_idx]
    sharpe = calculate_sharpe(positions, va_fwd, va_rf)
    sharpes.append(sharpe)
    
    print(f"      Sharpe: {sharpe:.4f}")

# 5. Results
avg_sharpe = np.mean(sharpes)
std_sharpe = np.std(sharpes)

print("\n" + "="*80)
print("RESULTS:")
print("="*80)
print(f"EXP-039 (Hybrid Ensemble) CV Sharpe: {avg_sharpe:.4f} ± {std_sharpe:.4f}")
print(f"EXP-038 v3 (Hybrid Single) CV Sharpe: 0.7025")
print(f"EXP-022 (Ensemble Baseline) CV Sharpe: 0.6368")

improvement_v3 = ((avg_sharpe - 0.7025) / 0.7025) * 100
print(f"Improvement vs v3: {improvement_v3:+.2f}%")

# Save results
results_df = pd.DataFrame({
    'fold': range(1, 6),
    'sharpe': sharpes
})
results_df.to_csv("experiments/039_hybrid_ensemble/results/cv_results.csv", index=False)
print("\nResults saved to experiments/039_hybrid_ensemble/results/")
