"""
EXP-016 Kaggle Submission

Model: Top 20 features + Optimized XGBoost
Expected Sharpe: ~0.78 (5-fold CV)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ============================================================================
# Feature Engineering Functions
# ============================================================================

def create_lag_features(df, cols, lags=[1, 5, 10, 20, 40, 60]):
    """Create lag features."""
    lag_dfs = [df.copy()]
    for col in cols:
        for lag in lags:
            lag_dfs.append(df[col].shift(lag).rename(f'{col}_lag{lag}'))
    return pd.concat(lag_dfs, axis=1)


def create_rolling_features(df, cols, windows=[5, 10, 20, 60]):
    """Create rolling window features."""
    rolling_dfs = [df.copy()]
    for col in cols:
        for window in windows:
            rolling_dfs.append(df[col].rolling(window).mean().rename(f'{col}_rolling_mean_{window}'))
            rolling_dfs.append(df[col].rolling(window).std().rename(f'{col}_rolling_std_{window}'))
    return pd.concat(rolling_dfs, axis=1)


def create_cross_sectional_features(df, cols):
    """Create cross-sectional features."""
    cross_dfs = [df.copy()]
    for col in cols:
        cross_dfs.append(df[col].rank(pct=True).rename(f'{col}_rank'))
        cross_dfs.append(((df[col] - df[col].mean()) / df[col].std()).rename(f'{col}_zscore'))
        cross_dfs.append(pd.qcut(df[col], q=10, labels=False, duplicates='drop').rename(f'{col}_quantile'))
    return pd.concat(cross_dfs, axis=1)


def create_volatility_features(df, cols, windows=[5, 20, 60]):
    """Create volatility features."""
    vol_dfs = [df.copy()]
    for col in cols:
        for window in windows:
            rolling_std = df[col].rolling(window).std()
            vol_dfs.append(rolling_std.rename(f'{col}_vol_{window}'))
            vol_dfs.append((df[col] / (rolling_std + 1e-8)).rename(f'{col}_vol_norm_{window}'))
            vol_dfs.append((rolling_std > rolling_std.rolling(60).mean()).astype(int).rename(f'{col}_vol_regime_{window}'))
    return pd.concat(vol_dfs, axis=1)


def create_momentum_features(df, cols, periods=[5, 20, 60]):
    """Create momentum features."""
    mom_dfs = [df.copy()]
    for col in cols:
        for period in periods:
            mom_dfs.append(df[col].pct_change(period).rename(f'{col}_return_{period}d'))
            mom_dfs.append(df[col].ewm(span=period, adjust=False).mean().rename(f'{col}_ema_{period}'))

        ema_short = df[col].ewm(span=5, adjust=False).mean()
        ema_long = df[col].ewm(span=20, adjust=False).mean()
        mom_dfs.append((ema_short - ema_long).rename(f'{col}_trend'))
    return pd.concat(mom_dfs, axis=1)


def create_all_features(df, base_cols):
    """Create all features."""
    df_features = df[base_cols].copy()
    df_features = create_lag_features(df_features, base_cols)
    df_features = create_rolling_features(df_features, base_cols)
    df_features = create_cross_sectional_features(df_features, base_cols)
    df_features = create_volatility_features(df_features, base_cols)
    df_features = create_momentum_features(df_features, base_cols)
    return df_features


# ============================================================================
# Configuration
# ============================================================================

TOP_20_FEATURES = [
    'M4', 'M4_vol_norm_20', 'V13_vol_norm_20', 'I2_trend',
    'E19_rolling_std_5', 'V7_vol_norm_60', 'D8_trend', 'P8_trend',
    'S2_ema_60', 'E19_vol_regime_60', 'V13_lag1', 'M2_rolling_mean_10',
    'E12_rolling_mean_5', 'E19_lag5', 'E19_vol_norm_20', 'P7',
    'V10_lag40', 'M4_vol_norm_5', 'V13', 'P10'
]

BEST_PARAMS = {
    'n_estimators': 150,
    'learning_rate': 0.025084,
    'max_depth': 7,
    'subsample': 1.0,
    'colsample_bytree': 0.6,
    'min_child_weight': 1,
    'reg_alpha': 0.0,
    'reg_lambda': 0.5,
    'random_state': 42,
    'tree_method': 'hist',
    'verbosity': 0,
    'n_jobs': -1,
}

# Position sizing parameter (from Phase 3 evaluation)
K = 600.0

# ============================================================================
# Main
# ============================================================================

print("="*80)
print("EXP-016 Kaggle Submission")
print("="*80)

# Load data - auto-detect Kaggle paths
import os

# Try multiple possible paths
train_paths = ["data/train.csv", "/kaggle/input/train.csv"]
test_paths = ["data/test.csv", "/kaggle/input/test.csv"]

# Auto-detect Kaggle input directory
if os.path.exists('/kaggle/input'):
    print("Scanning Kaggle input directories...")
    for item in os.listdir('/kaggle/input'):
        item_path = f'/kaggle/input/{item}'
        if os.path.isdir(item_path):
            print(f"  Found: {item}/")
            for file in os.listdir(item_path):
                if file == 'train.csv':
                    train_paths.insert(0, f'{item_path}/train.csv')
                if file == 'test.csv':
                    test_paths.insert(0, f'{item_path}/test.csv')

# Load train
train = None
for path in train_paths:
    try:
        train = pd.read_csv(path)
        print(f"✓ Train loaded from: {path}")
        break
    except:
        continue

if train is None:
    raise FileNotFoundError("Could not find train.csv")

# Load test
test = None
for path in test_paths:
    try:
        test = pd.read_csv(path)
        print(f"✓ Test loaded from: {path}")
        break
    except:
        continue

if test is None:
    raise FileNotFoundError("Could not find test.csv")

print(f"Train: {train.shape}")
print(f"Test: {test.shape}")

# Get base features from train
exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate',
                'market_forward_excess_returns', 'is_scored',
                'lagged_forward_returns', 'lagged_risk_free_rate',
                'lagged_market_forward_excess_returns']
base_cols = [c for c in train.columns if c not in exclude_cols]

print(f"Base features (train): {len(base_cols)}")

# Get base features from test (excluding id/date columns only)
test_base_cols = [c for c in test.columns if c not in ['date_id', 'id', 'date']]
print(f"Base features (test): {len(test_base_cols)}")

# Use common features only
common_cols = [c for c in base_cols if c in test.columns]
print(f"Common features: {len(common_cols)}")

# Create features
print("Creating features...")
X_train_full = create_all_features(train, common_cols)
X_test_full = create_all_features(test, common_cols)

# Select Top 20
X_train = X_train_full[TOP_20_FEATURES]
y_train = train['market_forward_excess_returns']
X_test = X_test_full[TOP_20_FEATURES]

print(f"Features: {X_train.shape[1]}")

# Preprocess
X_train = X_train.fillna(X_train.median()).replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.fillna(X_test.median()).replace([np.inf, -np.inf], np.nan).fillna(0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
print("Training...")
model = XGBRegressor(**BEST_PARAMS)
model.fit(X_train_scaled, y_train)

# Predict excess returns
excess_returns_pred = model.predict(X_test_scaled)

# Convert to positions (0~2)
positions = np.clip(1.0 + excess_returns_pred * K, 0.0, 2.0)

print(f"\nPredictions:")
print(f"  Mean: {positions.mean():.4f}")
print(f"  Std:  {positions.std():.4f}")
print(f"  Min:  {positions.min():.4f}")
print(f"  Max:  {positions.max():.4f}")

# Create submission (prediction column only, no date_id)
submission = pd.DataFrame({'prediction': positions})
submission.to_csv('submission.csv', index=False)

print(f"\n✓ Saved: submission.csv ({len(submission)} rows)")
print("="*80)
