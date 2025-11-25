#!/usr/bin/env python3
"""
EXP-041 Phase 2: Evaluate Genetic Features
Goal: Train XGBoost on the discovered Genetic Features + Base Features and evaluate performance.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import os
import joblib
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("EXP-041 Phase 2: Evaluate Genetic Features")
print("="*80)

# 1. Load Data
print("\n[1] Loading data...")
train = pd.read_csv("data/train.csv")
y_excess = train['market_forward_excess_returns'].values
fwd_returns = train['forward_returns'].values
risk_free = train['risk_free_rate'].values

# 2. Prepare Base Features
print("\n[2] Preparing Base Features...")
top_20 = [
    'M4', 'V13', 'V7', 'P8', 'S2', 'I2', 'E19', 'S5', 'P5', 'P7',
    'M2', 'V9', 'M3', 'P12', 'P10', 'V10', 'E12', 'P11', 'M12', 'S8'
]
X_base = train[top_20].fillna(train[top_20].median()).replace([np.inf, -np.inf], np.nan).fillna(0)

# Scale base features (needed for GP transformer)
scaler = StandardScaler()
X_base_scaled = scaler.fit_transform(X_base)

# 3. Load Genetic Transformer
print("\n[3] Loading Genetic Transformer...")
transformer_path = "experiments/041_genetic_features/results/genetic_transformer.pkl"
if not os.path.exists(transformer_path):
    print("[ERROR] Genetic transformer not found. Please run Phase 1 first.")
    exit(1)

est_gp = joblib.load(transformer_path)
print("   Transformer loaded.")

# 4. Transform Data (Generate Genetic Features)
print("\n[4] Generating Genetic Features...")
X_genetic = est_gp.transform(X_base_scaled)
genetic_cols = [f'GP_{i}' for i in range(X_genetic.shape[1])]
X_genetic_df = pd.DataFrame(X_genetic, columns=genetic_cols)

print(f"   Generated {len(genetic_cols)} genetic features.")

# 5. Combine Features
print("\n[5] Combining Features (Base + Genetic)...")
# We use Base features (unscaled) + Genetic features
X = pd.concat([X_base, X_genetic_df], axis=1)
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
print(f"   Feature Matrix: {X.shape}")

# 6. Train and Evaluate (CV)
print("\n[6] Evaluating Strategy (5-fold CV)...")

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
    
    # Scale combined features
    scaler_comb = StandardScaler()
    X_tr_scaled = scaler_comb.fit_transform(X_tr)
    X_va_scaled = scaler_comb.transform(X_va)
    
    # Train Quantile Models (XGBoost)
    quantiles = [0.1, 0.5, 0.9]
    preds = {}
    
    for q in quantiles:
        model = XGBRegressor(
            n_estimators=150, max_depth=7, learning_rate=0.025,
            subsample=1.0, colsample_bytree=0.6, reg_lambda=0.5,
            objective='reg:quantileerror', quantile_alpha=q,
            random_state=42, n_jobs=-1
        )
        model.fit(X_tr_scaled, y_tr)
        preds[q] = model.predict(X_va_scaled)
        
    # Strategy
    q10_pred = preds[0.1]
    q50_pred = preds[0.5]
    q90_pred = preds[0.9]
    
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

# 7. Results
avg_sharpe = np.mean(sharpes)
std_sharpe = np.std(sharpes)

print("\n" + "="*80)
print("RESULTS:")
print("="*80)
print(f"EXP-041 (Genetic Features) CV Sharpe: {avg_sharpe:.4f} ± {std_sharpe:.4f}")
print(f"EXP-038 v3 (Hybrid Single) CV Sharpe: 0.7025")

improvement = ((avg_sharpe - 0.7025) / 0.7025) * 100
print(f"Improvement vs EXP-038 v3: {improvement:+.2f}%")

# Save results
results_df = pd.DataFrame({
    'fold': range(1, 6),
    'sharpe': sharpes
})
results_df.to_csv("experiments/041_genetic_features/results/cv_results.csv", index=False)
print("\nResults saved to experiments/041_genetic_features/results/")
