#!/usr/bin/env python3
"""
EXP-038 v2 Phase 3: Evaluate Regime Strategy
Goal: Evaluate the Regime-Based Strategy (Dynamic Model Selection) on CV.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import os
import joblib

print("="*80)
print("EXP-038 v2 Phase 3: Evaluate Regime Strategy")
print("="*80)

# 1. Load Data and Threshold
print("\n[1] Loading data and threshold...")
train = pd.read_csv("data/train.csv")
y_excess = train['market_forward_excess_returns'].values
fwd_returns = train['forward_returns'].values
risk_free = train['risk_free_rate'].values
v13 = train['V13'].fillna(0).values

threshold_df = pd.read_csv("experiments/038_regime_based_modeling/results/regime_threshold.csv")
threshold = threshold_df['threshold_value'].iloc[0]
print(f"   Regime Threshold (V13): {threshold:.6f}")

# 2. Create Features (EXP-016 Top 30)
print("\n[2] Creating Features...")
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

top_30 = [
    'P8*S2', 'M4*V7', 'P8/P7', 'V7*P7', 'M4/S2',
    'S2*S5', 'S5/P7', 'M4*P8', 'M4²', 'V13²',
    'V7/P7', 'P8²', 'V7*I2', 'I2*E19', 'M4/P8',
    'S2/P5', 'V7*P5', 'P5', 'P5/P7', 'V7/P8',
    'M4/I2', 'M4/V7', 'M4/P5', 'P8/P5', 'V13/S2',
    'V13*I2', 'M4/E19', 'M4/P7', 'I2/S5', 'V13/P7'
]
X = X_all[top_30].copy()

# 3. Evaluate Strategy (5-fold CV)
print("\n[3] Evaluating Strategy (5-fold CV)...")

def calculate_sharpe(positions, fwd_returns, risk_free):
    positions = np.clip(positions, 0.0, 2.0)
    strategy_returns = risk_free * (1.0 - positions) + fwd_returns * positions
    excess_returns = strategy_returns - risk_free
    if np.std(strategy_returns) > 0:
        sharpe = (np.mean(excess_returns) / np.std(strategy_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0
    return sharpe

def train_and_predict(X_tr, y_tr, X_va, alpha):
    model = XGBRegressor(
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
    model.fit(X_tr, y_tr)
    return model.predict(X_va)

sharpes = []
tscv = TimeSeriesSplit(n_splits=5)
K = 250

for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
    print(f"\n   Fold {fold_idx}/5...")
    
    # Split Train/Val
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr = y_excess[tr_idx]
    v13_tr, v13_va = v13[tr_idx], v13[va_idx]
    
    # Split Train by Regime
    low_mask_tr = v13_tr <= threshold
    high_mask_tr = v13_tr > threshold
    
    X_tr_low, y_tr_low = X_tr[low_mask_tr], y_tr[low_mask_tr]
    X_tr_high, y_tr_high = X_tr[high_mask_tr], y_tr[high_mask_tr]
    
    # Train Low Models
    low_preds = {}
    has_low = len(X_tr_low) > 10  # Require at least 10 samples
    if has_low:
        scaler_low = StandardScaler()
        X_tr_low_scaled = scaler_low.fit_transform(X_tr_low)
        for alpha, name in [(0.1, 'q10'), (0.5, 'q50'), (0.9, 'q90')]:
            low_preds[name] = train_and_predict(X_tr_low_scaled, y_tr_low, scaler_low.transform(X_va), alpha)
    
    # Train High Models
    high_preds = {}
    has_high = len(X_tr_high) > 10
    if has_high:
        scaler_high = StandardScaler()
        X_tr_high_scaled = scaler_high.fit_transform(X_tr_high)
        for alpha, name in [(0.1, 'q10'), (0.5, 'q50'), (0.9, 'q90')]:
            high_preds[name] = train_and_predict(X_tr_high_scaled, y_tr_high, scaler_high.transform(X_va), alpha)
        
    # Dynamic Inference
    # For each sample in validation set, choose model based on its V13 value
    low_mask_va = v13_va <= threshold
    
    # Helper to get prediction
    def get_pred(mask, name):
        # If sample is Low Vol (mask=True)
        #   Use Low Model if available
        #   Else use High Model
        # If sample is High Vol (mask=False)
        #   Use High Model if available
        #   Else use Low Model
        
        # Initialize with zeros
        preds = np.zeros(len(mask))
        
        # Low Vol Samples
        if np.any(mask):
            if has_low:
                preds[mask] = low_preds[name][mask]
            elif has_high:
                # Fallback to High Model
                # Note: High Model scaler was used on X_va, so predictions are valid relative to that scaler?
                # No, X_va was transformed by scaler_low or scaler_high in train_and_predict.
                # We need to predict using the available model's scaler.
                # This is tricky because train_and_predict returns predictions for ALL X_va.
                # So low_preds[name] contains predictions for all X_va using Low Model.
                # So we can just use high_preds[name][mask].
                preds[mask] = high_preds[name][mask]
            else:
                preds[mask] = 0.0 # Should not happen
                
        # High Vol Samples
        if np.any(~mask):
            if has_high:
                preds[~mask] = high_preds[name][~mask]
            elif has_low:
                preds[~mask] = low_preds[name][~mask]
            else:
                preds[~mask] = 0.0
                
        return preds

    q10_final = get_pred(low_mask_va, 'q10')
    q50_final = get_pred(low_mask_va, 'q50')
    q90_final = get_pred(low_mask_va, 'q90')
    
    # Strategy
    ci_width = q90_final - q10_final
    confidence = 1.0 / (np.abs(ci_width) + 0.001)
    confidence = np.clip(confidence, 0.5, 5.0)
    positions = 1.0 + q50_final * K * confidence
    
    # Evaluate
    va_fwd = fwd_returns[va_idx]
    va_rf = risk_free[va_idx]
    sharpe = calculate_sharpe(positions, va_fwd, va_rf)
    sharpes.append(sharpe)
    
    print(f"      Sharpe: {sharpe:.4f}")

# 4. Results
avg_sharpe = np.mean(sharpes)
std_sharpe = np.std(sharpes)

print("\n" + "="*80)
print("RESULTS:")
print("="*80)
print(f"EXP-038 v2 CV Sharpe: {avg_sharpe:.4f} ± {std_sharpe:.4f}")
print(f"EXP-020 CV Sharpe: 0.6368 (Baseline)")

improvement = ((avg_sharpe - 0.6368) / 0.6368) * 100
print(f"Improvement: {improvement:+.2f}%")

if avg_sharpe > 0.6368:
    print("✅ SUCCESS: Regime-Based Strategy improved performance!")
else:
    print("❌ FAILURE: No improvement.")

# Save results
results_df = pd.DataFrame({
    'fold': range(1, 6),
    'sharpe': sharpes
})
results_df.to_csv("experiments/038_regime_based_modeling/results/cv_results_v2.csv", index=False)
print("\nResults saved to experiments/038_regime_based_modeling/results/")
