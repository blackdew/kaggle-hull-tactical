#!/usr/bin/env python3
"""EXP-007: Extended Feature Engineering

Features to add:
- Longer lags (20, 40, 60)
- Cross-sectional features (rank, quantile, z-score)
- Volatility features (rolling vol, vol regime)
- Momentum & Trend (return_5d, return_20d, EMA)
"""
from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd


def create_lag_features(
    df: pd.DataFrame,
    features: List[str],
    lags: List[int] = [1, 5, 10, 20, 40, 60]
) -> pd.DataFrame:
    """Create lag features for time series.

    Args:
        df: Input dataframe
        features: List of feature names to create lags for
        lags: List of lag values

    Returns:
        DataFrame with lag features added
    """
    df_new = df.copy()

    for col in features:
        if col not in df.columns:
            continue
        for lag in lags:
            df_new[f'{col}_lag{lag}'] = df[col].shift(lag)

    return df_new


def create_rolling_features(
    df: pd.DataFrame,
    features: List[str],
    windows: List[int] = [5, 10, 20, 60]
) -> pd.DataFrame:
    """Create rolling statistics features.

    Args:
        df: Input dataframe
        features: List of feature names
        windows: List of window sizes

    Returns:
        DataFrame with rolling features added
    """
    df_new = df.copy()

    for col in features:
        if col not in df.columns:
            continue
        for window in windows:
            df_new[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
            df_new[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()

    return df_new


def create_cross_sectional_features(
    df: pd.DataFrame,
    features: List[str],
    date_col: str = 'date_id'
) -> pd.DataFrame:
    """Create cross-sectional features within each date.

    Features:
    - rank: Rank within date (0~1 normalized)
    - zscore: (value - date_mean) / date_std
    - quantile: Quantile bin (0, 1, 2, 3, 4 for quintiles)

    Args:
        df: Input dataframe
        features: List of feature names
        date_col: Date column name

    Returns:
        DataFrame with cross-sectional features added
    """
    df_new = df.copy()

    if date_col not in df.columns:
        print(f"[WARNING] {date_col} not found, skipping cross-sectional features")
        return df_new

    for col in features:
        if col not in df.columns:
            continue

        # Rank (0~1 normalized)
        df_new[f'{col}_rank'] = df.groupby(date_col)[col].rank(pct=True)

        # Z-score
        date_mean = df.groupby(date_col)[col].transform('mean')
        date_std = df.groupby(date_col)[col].transform('std')
        df_new[f'{col}_zscore'] = (df[col] - date_mean) / (date_std + 1e-8)

        # Quantile bins (0~4)
        df_new[f'{col}_quantile'] = df.groupby(date_col)[col].transform(
            lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop')
        )

    return df_new


def create_volatility_features(
    df: pd.DataFrame,
    features: List[str],
    windows: List[int] = [5, 20, 60]
) -> pd.DataFrame:
    """Create volatility-related features.

    Features:
    - rolling_vol: Rolling standard deviation
    - vol_normalized: value / rolling_vol
    - vol_regime: High vol (1) vs Low vol (0)

    Args:
        df: Input dataframe
        features: List of feature names
        windows: List of window sizes

    Returns:
        DataFrame with volatility features added
    """
    df_new = df.copy()

    for col in features:
        if col not in df.columns:
            continue

        for window in windows:
            # Rolling volatility
            vol = df[col].rolling(window).std()
            df_new[f'{col}_vol_{window}'] = vol

            # Vol-normalized
            df_new[f'{col}_vol_norm_{window}'] = df[col] / (vol + 1e-8)

            # Vol regime (high vol if > 75th percentile)
            vol_threshold = vol.quantile(0.75)
            df_new[f'{col}_vol_regime_{window}'] = (vol > vol_threshold).astype(int)

    return df_new


def create_momentum_features(
    df: pd.DataFrame,
    features: List[str],
    periods: List[int] = [5, 20, 60]
) -> pd.DataFrame:
    """Create momentum and trend features.

    Features:
    - return_Nd: (value_t - value_{t-N}) / value_{t-N}
    - ema_N: Exponential moving average
    - trend: EMA_short - EMA_long

    Args:
        df: Input dataframe
        features: List of feature names
        periods: List of lookback periods

    Returns:
        DataFrame with momentum features added
    """
    df_new = df.copy()

    for col in features:
        if col not in df.columns:
            continue

        for period in periods:
            # Return over period
            df_new[f'{col}_return_{period}d'] = df[col].pct_change(period)

            # EMA
            df_new[f'{col}_ema_{period}'] = df[col].ewm(span=period, adjust=False).mean()

        # Trend: EMA_short - EMA_long
        if len(periods) >= 2:
            short_period = periods[0]
            long_period = periods[-1]
            ema_short = df[col].ewm(span=short_period, adjust=False).mean()
            ema_long = df[col].ewm(span=long_period, adjust=False).mean()
            df_new[f'{col}_trend'] = ema_short - ema_long

    return df_new


def create_all_features(
    df: pd.DataFrame,
    base_features: List[str],
    enable_lags: bool = True,
    enable_rolling: bool = True,
    enable_cross_sectional: bool = True,
    enable_volatility: bool = True,
    enable_momentum: bool = True,
    date_col: str = 'date_id'
) -> pd.DataFrame:
    """Create all extended features.

    Args:
        df: Input dataframe
        base_features: List of base feature names
        enable_*: Flags to enable/disable feature groups
        date_col: Date column name

    Returns:
        DataFrame with all features added
    """
    df_new = df.copy()

    print(f"[INFO] Starting feature engineering on {len(base_features)} base features")
    print(f"[INFO] Initial shape: {df_new.shape}")

    if enable_lags:
        print("[INFO] Creating lag features (1, 5, 10, 20, 40, 60)...")
        df_new = create_lag_features(df_new, base_features, lags=[1, 5, 10, 20, 40, 60])
        print(f"[INFO] After lags: {df_new.shape}")

    if enable_rolling:
        print("[INFO] Creating rolling features (5, 10, 20, 60)...")
        df_new = create_rolling_features(df_new, base_features, windows=[5, 10, 20, 60])
        print(f"[INFO] After rolling: {df_new.shape}")

    if enable_cross_sectional:
        print("[INFO] Creating cross-sectional features (rank, zscore, quantile)...")
        df_new = create_cross_sectional_features(df_new, base_features, date_col=date_col)
        print(f"[INFO] After cross-sectional: {df_new.shape}")

    if enable_volatility:
        print("[INFO] Creating volatility features (5, 20, 60)...")
        df_new = create_volatility_features(df_new, base_features, windows=[5, 20, 60])
        print(f"[INFO] After volatility: {df_new.shape}")

    if enable_momentum:
        print("[INFO] Creating momentum features (5, 20, 60)...")
        df_new = create_momentum_features(df_new, base_features, periods=[5, 20, 60])
        print(f"[INFO] After momentum: {df_new.shape}")

    print(f"[INFO] Final shape: {df_new.shape}")
    print(f"[INFO] Added {df_new.shape[1] - df.shape[1]} features")

    return df_new


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)

    # Create sample data
    n_dates = 100
    df = pd.DataFrame({
        'date_id': np.repeat(np.arange(n_dates), 1),
        'M4': np.random.randn(n_dates) * 0.01,
        'V13': np.random.randn(n_dates) * 0.02,
    })

    print("="*80)
    print("Testing Feature Engineering")
    print("="*80)
    print(f"Original shape: {df.shape}")
    print()

    # Test each feature type
    base_features = ['M4', 'V13']

    # Lags
    df_lag = create_lag_features(df.copy(), base_features, lags=[1, 5, 10])
    print(f"After lags: {df_lag.shape} (added {df_lag.shape[1] - df.shape[1]} features)")

    # Rolling
    df_roll = create_rolling_features(df.copy(), base_features, windows=[5, 10])
    print(f"After rolling: {df_roll.shape} (added {df_roll.shape[1] - df.shape[1]} features)")

    # Cross-sectional
    df_cs = create_cross_sectional_features(df.copy(), base_features)
    print(f"After cross-sectional: {df_cs.shape} (added {df_cs.shape[1] - df.shape[1]} features)")

    # Volatility
    df_vol = create_volatility_features(df.copy(), base_features, windows=[5, 20])
    print(f"After volatility: {df_vol.shape} (added {df_vol.shape[1] - df.shape[1]} features)")

    # Momentum
    df_mom = create_momentum_features(df.copy(), base_features, periods=[5, 20])
    print(f"After momentum: {df_mom.shape} (added {df_mom.shape[1] - df.shape[1]} features)")

    # All features
    print()
    print("Creating all features...")
    df_all = create_all_features(df.copy(), base_features)

    print()
    print("Sample of new features:")
    new_cols = [c for c in df_all.columns if c not in df.columns]
    print(new_cols[:20])
