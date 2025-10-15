#!/usr/bin/env python3
"""EXP-013b: ML with Technical Indicators

Combine technical indicators with ML to learn optimal trading patterns.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_macd(series: pd.Series, fast=12, slow=26, signal=9) -> tuple:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal


def bollinger_bands(series: pd.Series, period=20, std_dev=2) -> tuple:
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return ma + (std * std_dev), ma - (std * std_dev)


def create_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive technical + original features."""
    df_new = df.copy()

    # Reconstruct price
    df_new['price'] = (1 + df['forward_returns']).cumprod()

    # 1. Moving Averages (multiple timeframes)
    for window in [3, 5, 10, 20, 40, 60]:
        df_new[f'MA_{window}'] = df_new['price'].rolling(window).mean()
        df_new[f'price_vs_MA{window}'] = df_new['price'] / df_new['MA_{window}'] - 1

    # 2. MA crosses
    df_new['MA3_cross_MA10'] = ((df_new['MA_3'] > df_new['MA_10']).astype(int) -
                                (df_new['MA_3'].shift(1) > df_new['MA_10'].shift(1)).astype(int))
    df_new['MA5_cross_MA20'] = ((df_new['MA_5'] > df_new['MA_20']).astype(int) -
                                (df_new['MA_5'].shift(1) > df_new['MA_20'].shift(1)).astype(int))
    df_new['MA20_cross_MA60'] = ((df_new['MA_20'] > df_new['MA_60']).astype(int) -
                                 (df_new['MA_20'].shift(1) > df_new['MA_60'].shift(1)).astype(int))

    # 3. RSI (multiple periods)
    for period in [7, 14, 21]:
        rsi = calculate_rsi(df_new['price'], period)
        df_new[f'RSI_{period}'] = rsi
        df_new[f'RSI{period}_oversold'] = (rsi < 30).astype(int)
        df_new[f'RSI{period}_overbought'] = (rsi > 70).astype(int)

    # 4. MACD
    macd, macd_signal = calculate_macd(df_new['price'])
    df_new['MACD'] = macd
    df_new['MACD_signal'] = macd_signal
    df_new['MACD_diff'] = macd - macd_signal
    df_new['MACD_bullish'] = (df_new['MACD_diff'] > 0).astype(int)
    df_new['MACD_cross'] = ((df_new['MACD'] > df_new['MACD_signal']).astype(int) -
                            (df_new['MACD'].shift(1) > df_new['MACD_signal'].shift(1)).astype(int))

    # 5. Bollinger Bands
    bb_upper, bb_lower = bollinger_bands(df_new['price'], period=20)
    df_new['BB_upper'] = bb_upper
    df_new['BB_lower'] = bb_lower
    df_new['BB_width'] = (bb_upper - bb_lower) / df_new['price']
    df_new['BB_position'] = (df_new['price'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
    df_new['BB_squeeze'] = (df_new['BB_width'] < df_new['BB_width'].rolling(20).mean()).astype(int)

    # 6. Momentum indicators
    for period in [3, 5, 10, 20, 40]:
        df_new[f'ROC_{period}'] = df_new['price'].pct_change(period)
        df_new[f'momentum_{period}'] = df_new['price'] - df_new['price'].shift(period)

    # 7. Volatility
    for window in [5, 10, 20, 40]:
        df_new[f'volatility_{window}'] = df_new['forward_returns'].rolling(window).std()
        df_new[f'volatility_change_{window}'] = df_new[f'volatility_{window}'].pct_change()

    # 8. Volume-like (use forward_returns as proxy)
    df_new['volume_proxy'] = df_new['forward_returns'].abs()
    for window in [5, 20]:
        df_new[f'volume_MA_{window}'] = df_new['volume_proxy'].rolling(window).mean()

    # 9. Trend strength
    for window in [10, 20, 40]:
        df_new[f'trend_strength_{window}'] = (df_new['price'].rolling(window).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.std() if x.std() > 0 else 0
        ))

    # 10. Support/Resistance levels
    for window in [20, 60]:
        df_new[f'high_{window}'] = df_new['price'].rolling(window).max()
        df_new[f'low_{window}'] = df_new['price'].rolling(window).min()
        df_new[f'distance_to_high_{window}'] = (df_new['price'] - df_new[f'high_{window}']) / df_new['price']
        df_new[f'distance_to_low_{window}'] = (df_new['price'] - df_new[f'low_{window}']) / df_new['price']

    return df_new


def select_features(df: pd.DataFrame) -> List[str]:
    exclude = {
        'date_id', 'forward_returns', 'risk_free_rate',
        'market_forward_excess_returns', 'is_scored', 'price'
    }
    return [c for c in df.columns if c not in exclude]


def preprocess(df: pd.DataFrame, features: List[str], scaler=None) -> Tuple[np.ndarray, StandardScaler]:
    X = df[features].copy()
    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    if scaler is None:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = scaler.transform(X)

    return Xs, scaler


def evaluate_strategy(df: pd.DataFrame, positions: np.ndarray) -> Dict[str, float]:
    rf = df["risk_free_rate"].to_numpy()
    fwd = df["forward_returns"].to_numpy()

    strat_returns = rf * (1.0 - positions) + fwd * positions
    excess_returns = strat_returns - rf

    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)
    sharpe = (mean_excess / std_excess) * math.sqrt(252) if std_excess > 0 else 0.0

    total_profit = np.sum(excess_returns)
    utility = min(max(sharpe, 0), 6.0) * total_profit

    return {
        "sharpe": float(sharpe),
        "profit": float(total_profit),
        "utility": float(utility),
        "pos_mean": float(np.mean(positions)),
        "pos_std": float(np.std(positions)),
    }


def run_ml_with_technical(train: pd.DataFrame, k: float = 800.0) -> pd.DataFrame:
    """Train ML model with comprehensive technical features."""

    if not HAS_XGB:
        return pd.DataFrame()

    print(f"\nEXP-013b: ML with Technical Indicators (k={k})")
    print("="*80)

    # Create features
    print("[INFO] Creating comprehensive technical indicators...")
    train_eng = create_comprehensive_features(train)

    all_features = select_features(train_eng)
    print(f"[INFO] Total features: {len(all_features)}")

    target = "market_forward_excess_returns"
    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(train_eng), 1):
        print(f"\n[Fold {fold_idx}/5]")
        tr_df = train_eng.iloc[tr_idx]
        va_df = train_eng.iloc[va_idx]

        X_tr, scaler = preprocess(tr_df, all_features)
        y_tr = tr_df[target].to_numpy()

        X_va, _ = preprocess(va_df, all_features, scaler=scaler)

        # Train XGBoost
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=42,
            tree_method='hist',
            verbosity=0
        )
        model.fit(X_tr, y_tr)

        # Predict
        y_pred = model.predict(X_va)

        # Convert to positions
        positions = np.clip(1.0 + y_pred * k, 0.0, 2.0)

        # Evaluate
        metrics = evaluate_strategy(va_df, positions)
        metrics.update({"fold": fold_idx, "k": k})
        results.append(metrics)

        print(f"  Sharpe: {metrics['sharpe']:.3f}, Utility: {metrics['utility']:.3f}")

    df_results = pd.DataFrame(results)
    avg_sharpe = df_results['sharpe'].mean()
    avg_utility = df_results['utility'].mean()

    print(f"\n[RESULT]")
    print(f"  Avg Sharpe:    {avg_sharpe:.3f}")
    print(f"  Avg Utility:   {avg_utility:.3f}")
    print(f"  Baseline:      0.749 (EXP-007)")
    improvement = (avg_sharpe / 0.749 - 1) * 100
    print(f"  Improvement:   {improvement:+.1f}%")

    if avg_sharpe > 2.0:
        print("\n🎉 MAJOR BREAKTHROUGH! Sharpe > 2.0!")
    elif avg_sharpe > 1.5:
        print("\n🚀 EXCELLENT! Sharpe > 1.5!")
    elif avg_sharpe > 1.0:
        print("\n✅ SUCCESS! Sharpe > 1.0!")
    elif avg_sharpe > 0.749:
        print("\n📈 IMPROVEMENT!")

    return df_results


def load_train(path: str = "data/train.csv") -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    print("Loading data...")
    train = load_train()
    print(f"Train shape: {train.shape}\n")

    # Test different k values
    best_sharpe = 0
    best_k = 600

    for k in [600, 800, 1000, 1500]:
        df_results = run_ml_with_technical(train, k=k)
        df_results.to_csv(RESULTS / f"ml_technical_k{int(k)}.csv", index=False)

        sharpe = df_results['sharpe'].mean()
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_k = k

    print("\n" + "="*80)
    print("BEST RESULT")
    print("="*80)
    print(f"k={best_k}, Sharpe={best_sharpe:.3f}")

    print("\n[DONE]")
