"""
Simple standalone submission script for Kaggle
Copy this entire file to a Kaggle notebook and run it.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

print("="*60)
print("EXP-004 k=500 Submission Script")
print("="*60)

# Load data
print("\n[1/5] Loading data...")
train = pd.read_csv('/kaggle/input/hull-tactical-market-prediction/train.csv')
test = pd.read_csv('/kaggle/input/hull-tactical-market-prediction/test.csv')
print(f"✓ Train shape: {train.shape}")
print(f"✓ Test shape: {test.shape}")

# Feature selection
print("\n[2/5] Selecting top-20 features...")
target = 'market_forward_excess_returns'
exclude = {'date_id', 'forward_returns', 'risk_free_rate',
           'market_forward_excess_returns', 'is_scored',
           'lagged_forward_returns', 'lagged_risk_free_rate',
           'lagged_market_forward_excess_returns'}
feats = [c for c in train.columns if c not in exclude]

num = train[feats + [target]].select_dtypes(include=[np.number])
corr = num.corr(numeric_only=True)[target].drop(index=target).abs().sort_values(ascending=False)
top20 = [c for c in corr.index[:20] if c in feats]
print(f"✓ Top-20 features: {top20[:5]}... (total: {len(top20)})")

# Train model
print("\n[3/5] Training Lasso model...")
X_train = train[top20].fillna(train[top20].median()).replace([np.inf, -np.inf], np.nan).fillna(0)
y_train = train[target].to_numpy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = Lasso(alpha=1e-4, max_iter=50000)
model.fit(X_train_scaled, y_train)
print("✓ Model trained!")

# Predict
print("\n[4/5] Generating predictions...")
X_test = test[top20].fillna(train[top20].median()).replace([np.inf, -np.inf], np.nan).fillna(0)
X_test_scaled = scaler.transform(X_test)
excess_pred = model.predict(X_test_scaled)

# Convert to positions with k=500 (EXP-004 best result)
k = 500.0
positions = np.clip(1.0 + excess_pred * k, 0.0, 2.0)

print(f"✓ Predictions generated:")
print(f"  - Mean: {positions.mean():.4f}")
print(f"  - Std: {positions.std():.4f}")
print(f"  - Range: [{positions.min():.4f}, {positions.max():.4f}]")

# Create submission
print("\n[5/5] Creating submission files...")
row_id_col = test.columns[0]
submission = pd.DataFrame({
    row_id_col: test[row_id_col],
    'prediction': positions
})

submission.to_parquet('submission.parquet', index=False)
submission.to_csv('submission.csv', index=False)

print("✓ Submission files created!")
print("\n" + "="*60)
print("SUCCESS! submission.parquet is ready")
print("="*60)
print("\nSubmission preview:")
print(submission)
print("\nNext steps:")
print("1. Check Output tab for submission.parquet")
print("2. Click 'Submit to Competition' button")
print("3. Enter message: 'EXP-004 k=500, Sharpe 0.836'")
print("4. Submit and wait for score!")
