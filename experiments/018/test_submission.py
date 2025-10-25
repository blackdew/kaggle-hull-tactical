#!/usr/bin/env python3
"""
Test EXP-018 submission logic locally (without InferenceServer)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

print("="*80)
print("Testing EXP-018 Submission Logic")
print("="*80)
print()

# Configuration
top_30_interactions = [
    'P8*S2', 'M4*V7', 'P8/P7', 'V7*P7', 'M4/S2',
    'S2*S5', 'S5/P7', 'M4*P8', 'M4²', 'V13²',
    'V7/P7', 'P8²', 'V7*I2', 'I2*E19', 'M4/P8',
    'S2/P5', 'V7*P5', 'P5', 'P5/P7', 'V7/P8',
    'M4/I2', 'M4/V7', 'M4/P5', 'P8/P5', 'V13/S2',
    'V13*I2', 'M4/E19', 'M4/P7', 'I2/S5', 'V13/P7'
]

top_5_volatility = [
    'vol_mean_V', 'vol_composite_V_all', 'vol_composite_V',
    'cross_vol_MV', 'sentiment_dispersion'
]

all_features = top_30_interactions + top_5_volatility

top_20_base = [
    'M4', 'V13', 'V7', 'P8', 'S2', 'I2', 'E19', 'S5', 'P5', 'P7',
    'M2', 'V9', 'M3', 'P12', 'P10', 'V10', 'E12', 'P11', 'M12', 'S8'
]

k_base = 350.0
vol_scale = 1.0

def create_features(df):
    """Create interaction and volatility features (1-row calculable)"""
    for feat in top_20_base:
        if feat not in df.columns:
            df[feat] = 0.0

    df = df.fillna(0).replace([np.inf, -np.inf], 0)

    eps = 1e-8
    features = {}

    top_10 = top_20_base[:10]
    top_5 = top_20_base[:5]

    # Multiplication
    for i, feat1 in enumerate(top_10):
        for feat2 in top_10[i+1:]:
            name = f'{feat1}*{feat2}'
            features[name] = df[feat1] * df[feat2]

    # Division
    for i, feat1 in enumerate(top_10):
        for feat2 in top_10[i+1:]:
            name = f'{feat1}/{feat2}'
            features[name] = df[feat1] / (df[feat2].abs() + eps)

    # Polynomial
    for feat in top_5:
        features[f'{feat}²'] = df[feat] ** 2
        features[f'{feat}³'] = df[feat] ** 3

    # Add base features
    for feat in top_20_base:
        features[feat] = df[feat]

    # Volatility features
    features['vol_mean_V'] = (abs(df['V13']) + abs(df['V7']) + abs(df['V9'])) / 3
    features['vol_composite_V_all'] = np.sqrt(df['V13']**2 + df['V7']**2 + df['V9']**2 + df['V10']**2 + eps)
    features['vol_composite_V'] = np.sqrt(df['V13']**2 + df['V7']**2 + df['V9']**2 + eps)
    features['cross_vol_MV'] = np.sqrt(df['M4']**2 + df['V13']**2 + eps)
    features['sentiment_dispersion'] = abs(df['S2'] - df['S5'])

    return pd.DataFrame(features)

# Load data
print("[1] Loading data...")
train = pd.read_csv("data/train.csv")
print(f"Train shape: {train.shape}")

# Create features
print("[2] Creating features...")
X_all = create_features(train)
print(f"All features created: {X_all.shape}")

# Select features
missing = [f for f in all_features if f not in X_all.columns]
if missing:
    print(f"Missing features: {missing}")
    for f in missing:
        X_all[f] = 0.0

X = X_all[all_features].copy()
X = X.fillna(0).replace([np.inf, -np.inf], 0)
print(f"Final feature matrix: {X.shape}")

# Train model
print("[3] Training model...")
y = train['market_forward_excess_returns'].to_numpy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

params = {
    'n_estimators': 150,
    'max_depth': 7,
    'learning_rate': 0.025,
    'subsample': 1.0,
    'colsample_bytree': 0.6,
    'reg_lambda': 0.5,
    'random_state': 42,
    'n_jobs': -1
}

model = XGBRegressor(**params)
model.fit(X_scaled, y)
print("Model trained!")

# Test prediction on single row
print("[4] Testing single-row prediction...")
test_row = train.iloc[[100]]  # Single row
print(f"Test row shape: {test_row.shape}")

X_test = create_features(test_row)
X_test = X_test[all_features].fillna(0).replace([np.inf, -np.inf], 0)
X_test_scaled = scaler.transform(X_test)

excess_return_pred = model.predict(X_test_scaled)
print(f"Predicted excess return: {excess_return_pred[0]:.6f}")

# Dynamic K
vol_proxy = X_test['vol_mean_V'].values[0]
vol_adjustment = 1.0 / (1.0 + vol_scale * vol_proxy)
k_dynamic = k_base * vol_adjustment

print(f"Volatility proxy: {vol_proxy:.6f}")
print(f"K adjustment: {vol_adjustment:.6f}")
print(f"Dynamic K: {k_dynamic:.2f}")

position = np.clip(1.0 + excess_return_pred * k_dynamic, 0.0, 2.0)
print(f"Position: {position[0]:.6f}")

# Test on multiple rows
print("\n[5] Testing on 10 random rows...")
for i in range(10):
    idx = np.random.randint(0, len(train))
    test_row = train.iloc[[idx]]

    X_test = create_features(test_row)
    X_test = X_test[all_features].fillna(0).replace([np.inf, -np.inf], 0)
    X_test_scaled = scaler.transform(X_test)

    excess_pred = model.predict(X_test_scaled)[0]
    vol_proxy = X_test['vol_mean_V'].values[0]
    vol_adj = 1.0 / (1.0 + vol_scale * vol_proxy)
    k_dyn = k_base * vol_adj
    pos = np.clip(1.0 + excess_pred * k_dyn, 0.0, 2.0)

    print(f"  Row {idx:4d}: excess={excess_pred:+.6f}, vol={vol_proxy:.4f}, K={k_dyn:.1f}, pos={pos:.4f}")

print()
print("="*80)
print("Test Complete!")
print("="*80)
print()
print("✓ Feature creation works for single rows")
print("✓ Model prediction works")
print("✓ Dynamic K calculation works")
print("✓ Position calculation works")
print()
print("Ready for Kaggle submission!")
