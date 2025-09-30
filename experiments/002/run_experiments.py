#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def load_train(path: str = "data/train.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, StandardScaler]:
    X = df[features].copy()
    X = X.fillna(X.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def eval_fold(y_excess_pred: np.ndarray, valid_df: pd.DataFrame, k: float = 50.0, vol_cap: float | None = None) -> Dict[str, float]:
    # positions
    positions = np.clip(1.0 + y_excess_pred * (k / 1.0), 0.0, 2.0)
    rf = valid_df["risk_free_rate"].to_numpy()
    fwd = valid_df["forward_returns"].to_numpy()
    strat = rf * (1.0 - positions) + fwd * positions
    excess = strat - rf

    mkt_vol = float(np.std(fwd))
    strat_vol = float(np.std(strat))

    if vol_cap is not None and mkt_vol > 0:
        cap = vol_cap * mkt_vol
        if strat_vol > cap and strat_vol > 0:
            scale = cap / strat_vol
            positions = np.clip(positions * scale, 0.0, 2.0)
            strat = rf * (1.0 - positions) + fwd * positions
            excess = strat - rf
            strat_vol = float(np.std(strat))

    vol_ratio = strat_vol / mkt_vol if mkt_vol > 0 else math.nan
    sharpe = (np.mean(excess) / strat_vol) * np.sqrt(252) if strat_vol > 0 else 0.0
    mse = float(np.mean((valid_df["market_forward_excess_returns"].to_numpy() - y_excess_pred) ** 2))
    return {"mse": mse, "vol_ratio": vol_ratio, "sharpe": sharpe, "strat_vol": strat_vol, "mkt_vol": mkt_vol}


def base_features(df: pd.DataFrame) -> List[str]:
    exclude = {
        "date_id",
        "forward_returns",
        "risk_free_rate",
        "market_forward_excess_returns",
        "is_scored",
        "lagged_forward_returns",
        "lagged_risk_free_rate",
        "lagged_market_forward_excess_returns",
    }
    feats = [c for c in df.columns if c not in exclude]
    # filter high-missing > 0.5
    keep = []
    miss = df[feats].isna().mean()
    for c in feats:
        if miss.get(c, 0.0) < 0.5:
            keep.append(c)
    return keep


def winsorize_series(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def run_experiment(df: pd.DataFrame, expid: str, config: Dict) -> pd.DataFrame:
    feats = base_features(df)

    # H3 D1/D2 variants
    dd_mode = config.get("d12_mode")  # None | "d1_only" | "none"
    if dd_mode == "d1_only":
        feats = [f for f in feats if f != "D2"]
    elif dd_mode == "none":
        feats = [f for f in feats if f not in ("D1", "D2")]

    # H6 missing indicators for high-missing
    if config.get("add_missing_indicators"):
        high_missing = [c for c in ["E7", "V10", "M1", "M13", "M14", "V9"] if c in df.columns]
        for c in high_missing:
            ind = f"is_nan_{c}"
            df[ind] = df[c].isna().astype(float)
            feats.append(ind)

    # H2 interactions
    inter = config.get("interactions", [])
    for a, b in inter:
        if a in df.columns and b in df.columns:
            name = f"{a}x{b}"
            df[name] = df[a].fillna(0) * df[b].fillna(0)
            feats.append(name)

    # H1 scale fixes on selected features
    if config.get("scale_fix"):
        for c in ["M4", "V13"]:
            if c in df.columns:
                df[c] = winsorize_series(df[c])
                # simple log for positive values, else keep
                if (df[c] > 0).all():
                    df[c] = np.log1p(df[c])

    # model choice
    model_type = config.get("model", "ols")

    tscv = TimeSeriesSplit(n_splits=5)
    rows = []
    for fold, (tr, va) in enumerate(tscv.split(df), 1):
        tr_df, va_df = df.iloc[tr], df.iloc[va]

        Xs, scaler = preprocess(tr_df, feats)
        y = tr_df["market_forward_excess_returns"].to_numpy()

        if model_type == "ridge":
            model = Ridge(alpha=config.get("alpha", 1.0))
        elif model_type == "lasso":
            model = Lasso(alpha=config.get("alpha", 0.001), max_iter=10000)
        else:
            model = LinearRegression()
        model.fit(Xs, y)

        Xv = va_df[feats].copy().fillna(va_df[feats].median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
        Xvs = scaler.transform(Xv)
        yhat = model.predict(Xvs)

        metrics = eval_fold(yhat, va_df, k=config.get("k", 50.0), vol_cap=config.get("vol_cap"))
        metrics.update({"fold": fold})
        rows.append(metrics)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"{expid}_folds.csv", index=False)
    return out


def summarize(experiments: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for expid, df in experiments.items():
        rows.append({
            "expid": expid,
            "mse_mean": df["mse"].mean(),
            "sharpe_mean": df["sharpe"].mean(),
            "sharpe_median": df["sharpe"].median(),
            "vol_ratio_mean": df["vol_ratio"].mean(),
        })
    out = pd.DataFrame(rows).sort_values("expid")
    out.to_csv(RESULTS / "summary.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    df = load_train()
    exps: Dict[str, Dict] = {
        # Baseline
        "BASE": {"model": "ols", "k": 50.0},
        # H1
        "H1_scale": {"model": "ols", "scale_fix": True},
        # H2
        "H2_interact": {"model": "ols", "interactions": [("S5", "V13"), ("S2", "V7")]},
        # H3
        "H3_d1_only": {"model": "ols", "d12_mode": "d1_only"},
        "H3_none": {"model": "ols", "d12_mode": "none"},
        # H4
        "H4_k20": {"model": "ols", "k": 20.0},
        "H4_k25": {"model": "ols", "k": 25.0},
        "H4_k30": {"model": "ols", "k": 30.0},
        "H4_k35": {"model": "ols", "k": 35.0},
        "H4_k40": {"model": "ols", "k": 40.0},
        "H4_k45": {"model": "ols", "k": 45.0},
        "H4_k70": {"model": "ols", "k": 70.0},
        # H5
        "H5_volaware": {"model": "ols", "vol_cap": 1.2},
        # H6
        "H6_missing_mask": {"model": "ols", "add_missing_indicators": True},
        # Combined (new hypothesis)
        "H7_k25_volaware": {"model": "ols", "k": 25.0, "vol_cap": 1.2},
        "H7_k30_volaware": {"model": "ols", "k": 30.0, "vol_cap": 1.2},
        "H7_k35_volaware": {"model": "ols", "k": 35.0, "vol_cap": 1.2},
        # Regularization variants (bonus)
        "R_ridge": {"model": "ridge", "alpha": 1.0},
        "R_lasso_lo": {"model": "lasso", "alpha": 1e-4},
        "R_lasso": {"model": "lasso", "alpha": 1e-3},
        "R_lasso_hi": {"model": "lasso", "alpha": 1e-2},
        # Masks + regularization
        "H6mask_ridge": {"model": "ridge", "alpha": 1.0, "add_missing_indicators": True},
        "H6mask_lasso": {"model": "lasso", "alpha": 1e-3, "add_missing_indicators": True},
    }

    run_list = list(exps.keys()) if args.all else (args.only or [
        "BASE","H1_scale","H2_interact",
        "H4_k20","H4_k25","H4_k30","H4_k35","H4_k40","H4_k45",
        "H5_volaware","H7_k25_volaware","H7_k30_volaware","H7_k35_volaware",
        "H3_d1_only","H3_none","H6_missing_mask",
        "H6mask_ridge","H6mask_lasso","R_ridge","R_lasso_lo","R_lasso","R_lasso_hi"
    ])

    results: Dict[str, pd.DataFrame] = {}
    for expid in run_list:
        res = run_experiment(df.copy(), expid, exps[expid])
        results[expid] = res
        print(f"{expid}: sharpe_mean={res['sharpe'].mean():.4f}  vol_ratio_mean={res['vol_ratio'].mean():.3f}")

    summary = summarize(results)
    print("\nSummary saved:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
