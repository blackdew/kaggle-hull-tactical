#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso

from pipeline import select_base_features, top_abs_corr, preprocess_fit, preprocess_apply, to_positions

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "submissions"
OUTDIR.mkdir(parents=True, exist_ok=True)


def run_candidate_A(train: pd.DataFrame, test: pd.DataFrame, out_path: Path) -> None:
    # Lasso(alpha=1e-4) + Top-20(abs corr) + k=50 mapping, clip 0..2
    target = "market_forward_excess_returns"
    feats = select_base_features(train)
    feats_top = top_abs_corr(train, feats, target, n=20)

    Xs, scaler = preprocess_fit(train[feats_top])
    y = train[target].to_numpy()
    model = Lasso(alpha=1e-4, max_iter=50000)
    model.fit(Xs, y)

    Xt = preprocess_apply(test[feats_top], scaler)
    pred_excess = model.predict(Xt)
    positions = to_positions(pred_excess, k=50.0)

    sub = pd.DataFrame({"prediction": positions})
    sub.to_csv(out_path, index=False)


def run_candidate_B(train: pd.DataFrame, test: pd.DataFrame, out_path: Path) -> None:
    # OLS + base features + k=18 mapping, clip 0..2
    target = "market_forward_excess_returns"
    feats = select_base_features(train)
    Xs, scaler = preprocess_fit(train[feats])
    y = train[target].to_numpy()
    model = LinearRegression()
    model.fit(Xs, y)

    Xt = preprocess_apply(test[feats], scaler)
    pred_excess = model.predict(Xt)
    positions = to_positions(pred_excess, k=18.0)

    sub = pd.DataFrame({"prediction": positions})
    sub.to_csv(out_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=["A", "B"], required=True)
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    if args.candidate == "A":
        out = Path(args.out) if args.out else OUTDIR / "candidate_A.csv"
        run_candidate_A(train, test, out)
        print(f"Wrote {out}")
    else:
        out = Path(args.out) if args.out else OUTDIR / "candidate_B.csv"
        run_candidate_B(train, test, out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

