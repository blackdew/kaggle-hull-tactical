#!/usr/bin/env python3
"""
EXP-038 v2 Phase 2: Train Regime Models
Goal: Train separate Quantile XGBoost models for Low and High Volatility regimes.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import os
import joblib

print("="*80)
print("EXP-038 v2 Phase 2: Train Regime Models")
print("="*80)

# 1. Load Data and Threshold
print("\n[1] Loading data and threshold...")
train = pd.read_csv("data/train.csv")
threshold_df = pd.read_csv("experiments/038_regime_based_modeling/results/regime_threshold.csv")
threshold = threshold_df['threshold_value'].iloc[0]
print(f"   Regime Threshold (V13): {threshold:.6f}")

# 2. Create Features (EXP-016 Top 30)
print("\n[2] Creating Features (EXP-016 Top 30)...")
# Load Top 20 base features
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

# EXP-016 Top 30 Features
top_30 = [
    'P8*S2', 'M4*V7', 'P8/P7', 'V7*P7', 'M4/S2',
    'S2*S5', 'S5/P7', 'M4*P8', 'M4²', 'V13²',
    'V7/P7', 'P8²', 'V7*I2', 'I2*E19', 'M4/P8',
    'S2/P5', 'V7*P5', 'P5', 'P5/P7', 'V7/P8',
    'M4/I2', 'M4/V7', 'M4/P5', 'P8/P5', 'V13/S2',
    'V13*I2', 'M4/E19', 'M4/P7', 'I2/S5', 'V13/P7'
]
X = X_all[top_30].copy()
y = train['market_forward_excess_returns'].values
v13 = train['V13'].fillna(0).values

print(f"   Feature Matrix: {X.shape}")

# 3. Split by Regime
print("\n[3] Splitting by Regime...")
low_mask = v13 <= threshold
high_mask = v13 > threshold

X_low, y_low = X[low_mask], y[low_mask]
X_high, y_high = X[high_mask], y[high_mask]

print(f"   Low Vol Regime: {X_low.shape}")
print(f"   High Vol Regime: {X_high.shape}")

# 4. Train Models
print("\n[4] Training Models...")

def train_quantile_models(X_train, y_train, name):
    print(f"   Training {name} models...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    models = {}
    for alpha, q_name in [(0.1, 'q10'), (0.5, 'q50'), (0.9, 'q90')]:
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
        model.fit(X_scaled, y_train)
        models[q_name] = model
    
    return models, scaler

# Train Low Vol Models
low_models, low_scaler = train_quantile_models(X_low, y_low, "Low Vol")

# Train High Vol Models
high_models, high_scaler = train_quantile_models(X_high, y_high, "High Vol")

# 5. Save Models
print("\n[5] Saving Models...")
model_dir = "experiments/038_regime_based_modeling/results/models"
os.makedirs(model_dir, exist_ok=True)

# Save Low Vol
joblib.dump(low_models['q10'], f"{model_dir}/low_q10.pkl")
joblib.dump(low_models['q50'], f"{model_dir}/low_q50.pkl")
joblib.dump(low_models['q90'], f"{model_dir}/low_q90.pkl")
joblib.dump(low_scaler, f"{model_dir}/low_scaler.pkl")

# Save High Vol
joblib.dump(high_models['q10'], f"{model_dir}/high_q10.pkl")
joblib.dump(high_models['q50'], f"{model_dir}/high_q50.pkl")
joblib.dump(high_models['q90'], f"{model_dir}/high_q90.pkl")
joblib.dump(high_scaler, f"{model_dir}/high_scaler.pkl")

print(f"   Models saved to: {model_dir}")

print("\n" + "="*80)
print("Phase 2 Complete!")
print("="*80)
