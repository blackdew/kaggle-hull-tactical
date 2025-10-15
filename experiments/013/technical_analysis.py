#!/usr/bin/env python3
"""EXP-013: Technical Analysis - Chart Pattern Based Trading

Approach: Read the chart, find bottom/top, and time buy/sell decisions.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series: pd.Series, fast=12, slow=26, signal=9) -> tuple[pd.Series, pd.Series]:
    """Calculate MACD and signal line."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()

    return macd, macd_signal


def bollinger_bands(series: pd.Series, period=20, std_dev=2) -> tuple[pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()

    upper = ma + (std * std_dev)
    lower = ma - (std * std_dev)

    return upper, lower


def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical analysis features."""
    df_new = df.copy()

    # Reconstruct cumulative price from returns
    df_new['price'] = (1 + df['forward_returns']).cumprod()

    # Moving Averages
    for window in [5, 10, 20, 60]:
        df_new[f'MA_{window}'] = df_new['price'].rolling(window).mean()

    # Price relative to MAs
    df_new['price_vs_MA5'] = df_new['price'] / df_new['MA_5'] - 1
    df_new['price_vs_MA20'] = df_new['price'] / df_new['MA_20'] - 1
    df_new['price_vs_MA60'] = df_new['price'] / df_new['MA_60'] - 1

    # RSI
    df_new['RSI'] = calculate_rsi(df_new['price'], period=14)

    # MACD
    df_new['MACD'], df_new['MACD_signal'] = calculate_macd(df_new['price'])
    df_new['MACD_diff'] = df_new['MACD'] - df_new['MACD_signal']

    # Bollinger Bands
    df_new['BB_upper'], df_new['BB_lower'] = bollinger_bands(df_new['price'], period=20)
    df_new['BB_position'] = (df_new['price'] - df_new['BB_lower']) / (df_new['BB_upper'] - df_new['BB_lower'] + 1e-10)

    # Trend indicators
    df_new['MA_5_above_20'] = (df_new['MA_5'] > df_new['MA_20']).astype(int)
    df_new['MA_20_above_60'] = (df_new['MA_20'] > df_new['MA_60']).astype(int)

    # Momentum
    for period in [5, 10, 20]:
        df_new[f'ROC_{period}'] = df_new['price'].pct_change(period)

    # Volatility
    for window in [5, 20]:
        df_new[f'volatility_{window}'] = df_new['forward_returns'].rolling(window).std()

    # Signal indicators
    df_new['oversold'] = (df_new['RSI'] < 30).astype(int)
    df_new['overbought'] = (df_new['RSI'] > 70).astype(int)
    df_new['MACD_bullish'] = (df_new['MACD_diff'] > 0).astype(int)

    return df_new


def rule_based_position(row: pd.Series) -> float:
    """Rule-based trading decision based on technical indicators."""

    # Handle NaN
    if pd.isna(row.get('RSI')) or pd.isna(row.get('MA_5')):
        return 1.0  # Neutral if no data

    # Strong BUY signals (All-in)
    if (row['oversold'] == 1 and                    # RSI oversold
        row['MA_5_above_20'] == 1 and               # Short-term uptrend
        row['MACD_bullish'] == 1):                  # MACD bullish
        return 2.0

    # Strong SELL signals (All-out)
    if (row['overbought'] == 1 and                  # RSI overbought
        row['MA_5_above_20'] == 0 and               # Short-term downtrend
        row['MACD_bullish'] == 0):                  # MACD bearish
        return 0.0

    # Moderate BUY (Uptrend, not overbought)
    if (row['MA_5_above_20'] == 1 and
        row['MA_20_above_60'] == 1 and
        row['RSI'] < 65 and
        row['BB_position'] < 0.8):
        return 1.7

    # Moderate SELL (Downtrend, not oversold)
    if (row['MA_5_above_20'] == 0 and
        row['MA_20_above_60'] == 0 and
        row['RSI'] > 35 and
        row['BB_position'] > 0.2):
        return 0.3

    # Uptrend continuation
    if row['MA_5_above_20'] == 1 and row['MACD_bullish'] == 1:
        return 1.5

    # Downtrend continuation
    if row['MA_5_above_20'] == 0 and row['MACD_bullish'] == 0:
        return 0.5

    # Default: Neutral
    return 1.0


def evaluate_strategy(df: pd.DataFrame, positions: np.ndarray) -> Dict[str, float]:
    """Evaluate trading strategy."""

    rf = df["risk_free_rate"].to_numpy()
    fwd = df["forward_returns"].to_numpy()

    # Strategy returns
    strat_returns = rf * (1.0 - positions) + fwd * positions
    excess_returns = strat_returns - rf

    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)
    sharpe = (mean_excess / std_excess) * math.sqrt(252) if std_excess > 0 else 0.0

    total_profit = np.sum(excess_returns)
    utility = min(max(sharpe, 0), 6.0) * total_profit

    # Trading statistics
    n_buy = np.sum(positions > 1.5)
    n_sell = np.sum(positions < 0.5)
    n_neutral = np.sum((positions >= 0.9) & (positions <= 1.1))

    return {
        "sharpe": float(sharpe),
        "profit": float(total_profit),
        "utility": float(utility),
        "pos_mean": float(np.mean(positions)),
        "pos_std": float(np.std(positions)),
        "n_buy": int(n_buy),
        "n_sell": int(n_sell),
        "n_neutral": int(n_neutral),
    }


def run_rule_based_experiment(train: pd.DataFrame) -> pd.DataFrame:
    """Run rule-based technical analysis strategy."""

    print("\nEXP-013: Rule-Based Technical Analysis")
    print("="*80)

    # Create technical features
    print("[INFO] Creating technical indicators...")
    train_ta = create_technical_features(train)

    print("[INFO] Applying rule-based strategy...")
    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(train_ta), 1):
        print(f"\n[Fold {fold_idx}/5]")
        va_df = train_ta.iloc[va_idx]

        # Apply rules
        positions = va_df.apply(rule_based_position, axis=1).to_numpy()

        # Evaluate
        metrics = evaluate_strategy(va_df, positions)
        metrics.update({"fold": fold_idx})
        results.append(metrics)

        print(f"  Sharpe: {metrics['sharpe']:.3f}, Utility: {metrics['utility']:.3f}")
        print(f"  Trades: Buy={metrics['n_buy']}, Sell={metrics['n_sell']}, Neutral={metrics['n_neutral']}")

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
        print("\n📈 IMPROVEMENT over baseline!")

    return df_results


def load_train(path: str = "data/train.csv") -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    print("Loading data...")
    train = load_train()
    print(f"Train shape: {train.shape}\n")

    # Run rule-based strategy
    df_results = run_rule_based_experiment(train)
    df_results.to_csv(RESULTS / "rule_based.csv", index=False)

    print("\n[DONE] Technical analysis experiment completed!")
