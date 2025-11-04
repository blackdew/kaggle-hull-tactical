#!/usr/bin/env python3
"""
EXP-036: Find the 17.396 Strategy
Train 데이터에서 다양한 단순 전략의 Sharpe Ratio 계산
"""
import pandas as pd
import numpy as np

# Load train data
train = pd.read_csv('data/train.csv')
print(f"Train data shape: {train.shape}")
print(f"Date range: {train['date_id'].min()} - {train['date_id'].max()}")

# Target return
y = train['market_forward_excess_returns'].values

def calculate_sharpe(positions, returns):
    """Calculate Sharpe Ratio for given positions"""
    portfolio_returns = positions * returns
    if portfolio_returns.std() == 0:
        return 0.0
    sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    return sharpe

# Test various simple strategies
strategies = {}

# 1. Constant positions
for pos in [0.0, 0.5, 1.0, 1.5, 2.0]:
    positions = np.full(len(y), pos)
    sharpe = calculate_sharpe(positions, y)
    strategies[f'constant_{pos}'] = sharpe
    print(f"Constant {pos}: Sharpe = {sharpe:.4f}")

# 2. Momentum-based (using market features)
if 'M4' in train.columns:
    # Positive M4 -> bullish
    positions = np.where(train['M4'] > 0, 2.0, 0.0)
    sharpe = calculate_sharpe(positions, y)
    strategies['momentum_m4'] = sharpe
    print(f"Momentum M4: Sharpe = {sharpe:.4f}")

    # Scaled M4
    m4_scaled = train['M4'] / train['M4'].abs().max() * 2.0
    m4_scaled = np.clip(1.0 + m4_scaled, 0.0, 2.0)
    sharpe = calculate_sharpe(m4_scaled.values, y)
    strategies['scaled_m4'] = sharpe
    print(f"Scaled M4: Sharpe = {sharpe:.4f}")

# 3. Volatility-based (using V7 or V13)
if 'V7' in train.columns:
    # Low volatility -> higher position
    v7_inv = 1.0 / (train['V7'].abs() + 0.01)
    v7_inv = (v7_inv - v7_inv.min()) / (v7_inv.max() - v7_inv.min()) * 2.0
    sharpe = calculate_sharpe(v7_inv.values, y)
    strategies['inv_v7'] = sharpe
    print(f"Inverse V7: Sharpe = {sharpe:.4f}")

# 4. Price momentum (using P8)
if 'P8' in train.columns:
    positions = np.where(train['P8'] > 0, 2.0, 0.0)
    sharpe = calculate_sharpe(positions, y)
    strategies['price_p8'] = sharpe
    print(f"Price P8: Sharpe = {sharpe:.4f}")

# 5. Sentiment-based (using S2)
if 'S2' in train.columns:
    positions = np.where(train['S2'] > 0, 2.0, 0.0)
    sharpe = calculate_sharpe(positions, y)
    strategies['sentiment_s2'] = sharpe
    print(f"Sentiment S2: Sharpe = {sharpe:.4f}")

# 6. Mean reversion
positions = np.where(y < 0, 2.0, 0.0)  # This uses future info (cheating)
sharpe = calculate_sharpe(positions, y)
strategies['mean_reversion_cheat'] = sharpe
print(f"Mean Reversion (cheat): Sharpe = {sharpe:.4f}")

# 7. Alternating positions
positions = np.array([2.0 if i % 2 == 0 else 0.0 for i in range(len(y))])
sharpe = calculate_sharpe(positions, y)
strategies['alternating'] = sharpe
print(f"Alternating: Sharpe = {sharpe:.4f}")

# Find best strategy
best_strategy = max(strategies.items(), key=lambda x: x[1])
print(f"\n{'='*50}")
print(f"Best strategy: {best_strategy[0]}")
print(f"Sharpe Ratio: {best_strategy[1]:.4f}")
print(f"Target: 17.396")
print(f"Ratio: {best_strategy[1] / 17.396:.4f}x")
print(f"{'='*50}")

# Check if any strategy is close to 17.396
for name, sharpe in sorted(strategies.items(), key=lambda x: x[1], reverse=True):
    if sharpe > 10:
        print(f"⭐ {name}: {sharpe:.4f} (close to target!)")
