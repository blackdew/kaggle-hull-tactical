#!/usr/bin/env python3
# snapshot: experiments/001
"""
Time-series CV evaluation for the baseline (SimplePositionSizingModel).

Reports per-fold and aggregate proxy metrics:
- MSE (excess)
- Volatility ratio (strategy / market)
- Sharpe-like metric (mean excess / std strategy) * sqrt(252)
- Same Sharpe after enforcing 120% market volatility cap via linear scaling

Usage:
  python scripts/cv_evaluate.py --splits 5
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from make_submission import SimplePositionSizingModel


def fold_metrics(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict[str, float]:
    model = SimplePositionSizingModel()
    model.fit(train_df)

    # Predict excess on valid
    Xv = model.preprocess(valid_df, model.feature_cols)
    Xvs = model.scaler.transform(Xv)
    excess_pred = model.model.predict(Xvs)
    yv = valid_df["market_forward_excess_returns"].to_numpy()
    mse = float(np.mean((yv - excess_pred) ** 2))

    # Positions and returns
    positions = np.clip(1.0 + excess_pred * 0.5 * 100, 0.0, 2.0)
    rf = valid_df["risk_free_rate"].to_numpy()
    fwd = valid_df["forward_returns"].to_numpy()
    strat = rf * (1.0 - positions) + fwd * positions
    excess = strat - rf

    mkt_vol = float(np.std(fwd))
    strat_vol = float(np.std(strat))
    vol_ratio = strat_vol / mkt_vol if mkt_vol > 0 else float("nan")
    sharpe = (np.mean(excess) / strat_vol) * np.sqrt(252) if strat_vol > 0 else 0.0

    # 120% cap
    cap = 1.2 * mkt_vol
    if strat_vol > cap and strat_vol > 0:
        scale = cap / strat_vol
        positions_c = np.clip(positions * scale, 0.0, 2.0)
        strat_c = rf * (1.0 - positions_c) + fwd * positions_c
        excess_c = strat_c - rf
        strat_vol_c = float(np.std(strat_c))
        sharpe_c = (np.mean(excess_c) / strat_vol_c) * np.sqrt(252) if strat_vol_c > 0 else 0.0
    else:
        strat_vol_c = strat_vol
        sharpe_c = sharpe

    return {
        "mse": mse,
        "vol_ratio": vol_ratio,
        "sharpe": sharpe,
        "sharpe_cap": sharpe_c,
        "strat_vol": strat_vol,
        "mkt_vol": mkt_vol,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv("data/train.csv")
    tscv = TimeSeriesSplit(n_splits=args.splits)

    rows: list[dict[str, float]] = []
    for i, (tr_idx, va_idx) in enumerate(tscv.split(df), start=1):
        tr = df.iloc[tr_idx]
        va = df.iloc[va_idx]
        m = fold_metrics(tr, va)
        rows.append(m)
        print(f"Fold {i}: mse={m['mse']:.8f} vol={m['strat_vol']:.6f}/{m['mkt_vol']:.6f} ({m['vol_ratio']:.4f}) sharpe={m['sharpe']:.4f} sharpe_cap={m['sharpe_cap']:.4f}")

    agg = pd.DataFrame(rows).agg(["mean", "median"]).T
    print("\nAggregate (mean / median):")
    for k, s in agg.iterrows():
        print(f"- {k:10s}: mean={s['mean']:.6f}  median={s['median']:.6f}")


if __name__ == "__main__":
    main()
