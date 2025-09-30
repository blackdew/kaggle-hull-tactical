#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit

from pipeline import select_base_features, top_abs_corr, preprocess_fit, preprocess_apply

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)


def eval_fold(pred_excess: np.ndarray, valid_df: pd.DataFrame, k: float = 50.0) -> Tuple[float, float, float]:
    positions = np.clip(1.0 + pred_excess * (k / 1.0), 0.0, 2.0)
    rf = valid_df["risk_free_rate"].to_numpy()
    fwd = valid_df["forward_returns"].to_numpy()
    strat = rf * (1.0 - positions) + fwd * positions
    excess = strat - rf
    mkt_vol = float(np.std(fwd))
    strat_vol = float(np.std(strat))
    sharpe = (np.mean(excess) / strat_vol) * np.sqrt(252) if strat_vol > 0 else 0.0
    vol_ratio = strat_vol / mkt_vol if mkt_vol > 0 else np.nan
    mse = float(np.mean((valid_df["market_forward_excess_returns"].to_numpy() - pred_excess) ** 2))
    return sharpe, vol_ratio, mse


def cv_score(train: pd.DataFrame, top_n: int, alpha: float, k: float = 50.0, splits: int = 5) -> Tuple[float, float, float]:
    target = "market_forward_excess_returns"
    feats = select_base_features(train)
    tscv = TimeSeriesSplit(n_splits=splits)
    shs: List[float] = []
    vrs: List[float] = []
    mses: List[float] = []
    for tr_idx, va_idx in tscv.split(train):
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        top_feats = top_abs_corr(tr, feats, target, n=top_n)
        Xs, scaler = preprocess_fit(tr[top_feats])
        y = tr[target].to_numpy()
        model = Lasso(alpha=alpha, max_iter=50000)
        model.fit(Xs, y)
        Xv = preprocess_apply(va[top_feats], scaler)
        yhat = model.predict(Xv)
        s, vr, mse = eval_fold(yhat, va, k=k)
        shs.append(s); vrs.append(vr); mses.append(mse)
    return float(np.mean(shs)), float(np.mean(vrs)), float(np.mean(mses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv")
    args = ap.parse_args()
    train = pd.read_csv(args.train)

    grid_top = [15, 20, 25]
    grid_alpha = [5e-4, 1e-4, 1e-3]
    rows = []
    for n in grid_top:
        for a in grid_alpha:
            sharpe, vol_ratio, mse = cv_score(train, top_n=n, alpha=a, k=50.0)
            rows.append({"top_n": n, "alpha": a, "sharpe": sharpe, "vol_ratio": vol_ratio, "mse": mse})
            print(f"top={n} alpha={a}: sharpe={sharpe:.4f} vol={vol_ratio:.3f}")
    out = pd.DataFrame(rows).sort_values(["sharpe"], ascending=False)
    out.to_csv(OUT / "sensitivity_lasso.csv", index=False)
    print("Saved:", OUT / "sensitivity_lasso.csv")


if __name__ == "__main__":
    main()

