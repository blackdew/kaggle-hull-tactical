#!/usr/bin/env python3
"""
EXP-040 Phase 1: Select Best Hybrid Features
Goal: Refine the Hybrid Feature Set (51 features) by selecting the top 35 most important features.
      This aims to reduce complexity and prevent overfitting in the Ensemble model.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt

print("="*80)
print("EXP-040 Phase 1: Select Best Hybrid Features")
print("="*80)

# 1. Load Data
print("\n[1] Loading data...")
train = pd.read_csv("data/train.csv")
y = train['market_forward_excess_returns'].values

# 2. Load Hybrid Features (from EXP-038 v3)
print("\n[2] Loading Hybrid Features...")
hybrid_path = "experiments/038_hybrid_features/results/hybrid_features.csv"
if os.path.exists(hybrid_path):
    hybrid_df = pd.read_csv(hybrid_path)
    hybrid_features = hybrid_df['feature'].tolist()
else:
    # Fallback
    print("[WARNING] Hybrid features file missing. Using hardcoded list.")
    hybrid_features = [
        'E12', 'E19', 'E19*S5', 'E19/P7', 'E19/S5', 'I2*E19', 'I2*S5', 'I2/S5', 'M4*E19', 'M4*I2',
        'M4*P8', 'M4*V7', 'M4/E19', 'M4/I2', 'M4/P7', 'M4/P8', 'M4/S2', 'M4/V7', 'M4²', 'P5',
        'P5/P7', 'P8*E19', 'P8*S2', 'P8/P5', 'P8/P7', 'P8/S2', 'P8²', 'S2*E19', 'S2*I2', 'S2*S5',
        'S2/P5', 'S2/P7', 'S5/P7', 'S8', 'V13*I2', 'V13*V7', 'V13/S2', 'V13/S5', 'V13/P7', 'V13²',
        'V7*E19', 'V7*I2', 'V7*P5', 'V7*P7', 'V7*P8', 'V7*S2', 'V7/I2', 'V7/P7', 'V7/P8', 'V7/S2',
        'V7/S5'
    ]

print(f"   Original Hybrid Features: {len(hybrid_features)}")

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

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Calculate Feature Importance
print("\n[4] Calculating Feature Importance (XGBoost)...")

# Train q10, q90 models to see what matters for tails
def get_importance(alpha, name):
    print(f"   Training {name} (alpha={alpha})...")
    model = XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        objective='reg:quantileerror', quantile_alpha=alpha,
        n_jobs=-1, random_state=42
    )
    model.fit(X_scaled, y)
    return model.feature_importances_

imp_q10 = get_importance(0.1, "q10")
imp_q90 = get_importance(0.9, "q90")

# Average importance
avg_imp = (imp_q10 + imp_q90) / 2

# Create DataFrame
imp_df = pd.DataFrame({
    'feature': hybrid_features,
    'importance': avg_imp,
    'imp_q10': imp_q10,
    'imp_q90': imp_q90
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
print(imp_df.head(10))

print("\nBottom 10 Features:")
print(imp_df.tail(10))

# 5. Select Top N
N = 35
print(f"\n[5] Selecting Top {N} Features...")
refined_features = imp_df.head(N)['feature'].tolist()
print(f"   Refined Features: {refined_features}")

# 6. Save Results
print("\n[6] Saving Results...")
output_path = "experiments/040_refined_hybrid/results/refined_features.csv"
pd.DataFrame({'feature': refined_features}).to_csv(output_path, index=False)
imp_df.to_csv("experiments/040_refined_hybrid/results/feature_importance.csv", index=False)
print(f"   Saved to: {output_path}")

print("\n" + "="*80)
print("Phase 1 Complete!")
print("="*80)
