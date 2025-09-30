#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from run_experiments import run_experiment, load_train

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def time_blocks(df: pd.DataFrame, blocks: int = 5) -> list[tuple[int, int]]:
    n = len(df)
    size = n // blocks
    cuts = [(i * size, (i + 1) * size if i < blocks - 1 else n) for i in range(blocks)]
    return cuts


def rolling_eval(expid: str, config: dict, blocks: int = 5) -> pd.DataFrame:
    df = load_train()
    cuts = time_blocks(df, blocks)
    rows = []
    for i in range(1, blocks):
        train_end = cuts[i - 1][1]
        valid_start, valid_end = cuts[i]
        sub = df.iloc[:valid_end].copy()  # allow internal CV to respect time order
        res = run_experiment(sub, expid=f"{expid}_b{i}", config=config)
        # take last fold as the closest proxy to this block (simple heuristic)
        last = res.iloc[-1].to_dict()
        last.update({"block": i, "train_upto": train_end, "valid_range": f"{valid_start}-{valid_end}"})
        rows.append(last)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"rolling_{expid}.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expid", required=True)
    args = ap.parse_args()

    # map to configs similar to run_experiments
    configs = {
        "H7_k20_volaware": {"model": "ols", "k": 20.0, "vol_cap": 1.2},
        "R_lasso_lo": {"model": "lasso", "alpha": 1e-4},
        "H8_top20": {"model": "ols", "top_corr_n": 20},
        "H7_lasso_lo_top20_volaware": {"model": "lasso", "alpha": 1e-4, "top_corr_n": 20, "vol_cap": 1.2},
    }
    cfg = configs.get(args.expid)
    if not cfg:
        raise SystemExit(f"Unknown expid: {args.expid}")

    out = rolling_eval(args.expid, cfg, blocks=5)
    print(out[["block", "sharpe", "vol_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()

