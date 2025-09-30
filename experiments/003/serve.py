#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso

from pipeline import (
    EXCLUDE_COLS,
    select_base_features,
    top_abs_corr,
    preprocess_fit,
    preprocess_apply,
    to_positions,
)


def validate_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    missing = [c for c in required if c not in df.columns]
    return missing


def validate_values(df: pd.DataFrame) -> Dict[str, float]:
    stats = {
        "rows": float(len(df)),
        "cols": float(len(df.columns)),
        "nan_frac": float(df.isna().mean().mean()),
        "inf_count": float(np.isinf(df.select_dtypes(include=[np.number]).to_numpy()).sum()),
    }
    return stats


def train_vol_stats(train: pd.DataFrame, positions: np.ndarray) -> Tuple[float, float, float]:
    rf = train["risk_free_rate"].to_numpy()
    fwd = train["forward_returns"].to_numpy()
    strat = rf * (1.0 - positions) + fwd * positions
    strat_vol = float(np.std(strat))
    mkt_vol = float(np.std(fwd))
    vol_ratio = strat_vol / mkt_vol if mkt_vol > 0 else np.nan
    return strat_vol, mkt_vol, vol_ratio


def compute_cap_scale(train: pd.DataFrame, positions: np.ndarray, cap: float = 1.2) -> float:
    strat_vol, mkt_vol, vol_ratio = train_vol_stats(train, positions)
    if mkt_vol <= 0 or strat_vol <= 0:
        return 1.0
    limit = cap * mkt_vol
    return min(1.0, limit / strat_vol)


def candidate_A(train: pd.DataFrame, test: pd.DataFrame, top_n: int = 20, alpha: float = 1e-4,
                k: float = 50.0, vol_cap: float | None = None) -> Tuple[pd.DataFrame, Dict]:
    target = "market_forward_excess_returns"
    feats = select_base_features(train)
    feats_top = top_abs_corr(train, feats, target, n=top_n)
    Xs, scaler = preprocess_fit(train[feats_top])
    y = train[target].to_numpy()
    model = Lasso(alpha=alpha, max_iter=50000)
    model.fit(Xs, y)
    pred_train = model.predict(Xs)
    pos_train = to_positions(pred_train, k=k)
    scale = 1.0
    if vol_cap is not None:
        scale = compute_cap_scale(train, pos_train, cap=vol_cap)

    Xt = preprocess_apply(test[feats_top], scaler)
    pred_test = model.predict(Xt)
    positions = to_positions(pred_test, k=k) * scale
    positions = np.clip(positions, 0.0, 2.0)
    sub = pd.DataFrame({"prediction": positions})
    meta = {
        "candidate": "A",
        "top_n": top_n,
        "alpha": alpha,
        "k": k,
        "vol_cap": vol_cap,
        "scale": scale,
        "features_used": feats_top,
    }
    return sub, meta


def candidate_B(train: pd.DataFrame, test: pd.DataFrame, k: float = 18.0, vol_cap: float | None = None) -> Tuple[pd.DataFrame, Dict]:
    target = "market_forward_excess_returns"
    feats = select_base_features(train)
    Xs, scaler = preprocess_fit(train[feats])
    y = train[target].to_numpy()
    model = LinearRegression()
    model.fit(Xs, y)
    pred_train = model.predict(Xs)
    pos_train = to_positions(pred_train, k=k)
    scale = 1.0
    if vol_cap is not None:
        scale = compute_cap_scale(train, pos_train, cap=vol_cap)

    Xt = preprocess_apply(test[feats], scaler)
    pred_test = model.predict(Xt)
    positions = to_positions(pred_test, k=k) * scale
    positions = np.clip(positions, 0.0, 2.0)
    sub = pd.DataFrame({"prediction": positions})
    meta = {
        "candidate": "B",
        "k": k,
        "vol_cap": vol_cap,
        "scale": scale,
        "features_used_count": len(feats),
    }
    return sub, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=["A", "B"], required=True)
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--logdir", default="experiments/003/logs")
    ap.add_argument("--vol_cap", type=float, default=1.2)
    ap.add_argument("--top_n", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--k", type=float, default=None)
    args = ap.parse_args()

    logdir = Path(args.logdir); logdir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    # validations
    required_cols = ["risk_free_rate", "forward_returns", "market_forward_excess_returns"]
    missing_train = validate_columns(train, required_cols)
    missing_test = validate_columns(test, [c for c in ["risk_free_rate", "forward_returns"] if c in test.columns])
    vtrain = validate_values(train.select_dtypes(include=[np.number]))
    vtest = validate_values(test.select_dtypes(include=[np.number]))

    if missing_train:
        print("Warning: missing columns in train:", missing_train)
    if missing_test:
        print("Note: test missing optional columns (expected in Kaggle):", missing_test)

    if args.candidate == "A":
        sub, meta = candidate_A(train, test, top_n=args.top_n, alpha=args.alpha, k=(args.k or 50.0), vol_cap=args.vol_cap)
        out = Path(args.out) if args.out else Path("experiments/003/submissions/candidate_A_served.csv")
    else:
        sub, meta = candidate_B(train, test, k=(args.k or 18.0), vol_cap=args.vol_cap)
        out = Path(args.out) if args.out else Path("experiments/003/submissions/candidate_B_served.csv")

    # final safety: ensure [0,2]
    sub["prediction"] = sub["prediction"].clip(0.0, 2.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)

    # logging
    meta.update({
        "timestamp": ts,
        "train_stats": vtrain,
        "test_stats": vtest,
        "output_file": str(out),
        "rows_test": len(test),
        "exclude_cols": sorted(list(EXCLUDE_COLS)),
    })
    (logdir / f"serve_{args.candidate}_{ts}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Wrote {out} and log {logdir}/serve_{args.candidate}_{ts}.json")


if __name__ == "__main__":
    main()

