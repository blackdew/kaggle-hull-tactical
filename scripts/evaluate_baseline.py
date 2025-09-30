#!/usr/bin/env python3
"""
Evaluate the SimplePositionSizingModel used for the first submission.

Reports:
- In-sample MSE for market_forward_excess_returns
- Strategy volatility, market volatility, volatility ratio
- Sharpe-like metric (mean excess / std strategy) with 252 scaling
- Same metric after capping to 120% of market volatility
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from make_submission import SimplePositionSizingModel  # reuse standalone version


def main() -> None:
    train = pd.read_csv("data/train.csv")

    model = SimplePositionSizingModel()
    model.fit(train)

    # In-sample predictions of excess returns
    X = model.preprocess(train, model.feature_cols)
    Xs = model.scaler.transform(X)
    excess_pred = model.model.predict(Xs)
    y = train["market_forward_excess_returns"].to_numpy()
    mse = float(np.mean((y - excess_pred) ** 2))

    # Positions + strategy returns (0..2 clipping already inside to_position)
    positions = np.clip(1.0 + excess_pred * 0.5 * 100, 0.0, 2.0)

    rf = train["risk_free_rate"].to_numpy()
    fwd = train["forward_returns"].to_numpy()

    strat = rf * (1.0 - positions) + fwd * positions
    excess = strat - rf

    mkt_vol = float(np.std(fwd))
    strat_vol = float(np.std(strat))
    vol_ratio = strat_vol / mkt_vol if mkt_vol > 0 else float("nan")

    # Simple Sharpe-like metric (not official Modified Sharpe)
    sharpe = (np.mean(excess) / strat_vol) * np.sqrt(252) if strat_vol > 0 else 0.0

    # Apply 120% volatility cap via linear scaling of positions
    cap = 1.2 * mkt_vol
    if strat_vol > cap and strat_vol > 0:
        scale = cap / strat_vol
        positions_capped = np.clip(positions * scale, 0.0, 2.0)
        strat_c = rf * (1.0 - positions_capped) + fwd * positions_capped
        excess_c = strat_c - rf
        strat_vol_c = float(np.std(strat_c))
        sharpe_c = (np.mean(excess_c) / strat_vol_c) * np.sqrt(252) if strat_vol_c > 0 else 0.0
    else:
        positions_capped = positions
        strat_vol_c = strat_vol
        sharpe_c = sharpe

    print("In-sample evaluation (proxy metrics):")
    print(f"- MSE (excess): {mse:.8f}")
    print(f"- Volatility (strategy): {strat_vol:.6f}")
    print(f"- Volatility (market)  : {mkt_vol:.6f}")
    print(f"- Volatility ratio     : {vol_ratio:.4f}")
    print(f"- Sharpe (approx)      : {sharpe:.4f}")
    print(f"- Sharpe @ 120% cap    : {sharpe_c:.4f} (vol={strat_vol_c:.6f})")


if __name__ == "__main__":
    main()

