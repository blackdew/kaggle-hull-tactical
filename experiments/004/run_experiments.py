#!/usr/bin/env python3
"""EXP-004: Position Scaling & Prediction Amplification Experiments"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
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


def eval_fold(y_pred: np.ndarray, valid_df: pd.DataFrame, mode: str = "excess", k: float = 50.0, vol_cap: float | None = None) -> Dict[str, float]:
    """
    Evaluate predictions on validation fold.

    Args:
        y_pred: predictions (either excess returns or direct positions)
        valid_df: validation dataframe with targets
        mode: "excess" (predict excess return) or "position" (predict position directly)
        k: leverage factor (only used in excess mode)
        vol_cap: optional volatility cap multiplier
    """
    rf = valid_df["risk_free_rate"].to_numpy()
    fwd = valid_df["forward_returns"].to_numpy()

    if mode == "excess":
        # Traditional: predict excess, convert to position
        positions = np.clip(1.0 + y_pred * k, 0.0, 2.0)
    elif mode == "position":
        # Direct: predictions are already positions
        positions = np.clip(y_pred, 0.0, 2.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    strat = rf * (1.0 - positions) + fwd * positions
    excess = strat - rf

    mkt_vol = float(np.std(fwd))
    strat_vol = float(np.std(strat))

    # Apply vol cap if specified
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
    mse = float(np.mean((valid_df["market_forward_excess_returns"].to_numpy() - (y_pred if mode == "excess" else (positions - 1.0) / k)) ** 2))

    return {
        "mse": mse,
        "vol_ratio": vol_ratio,
        "sharpe": sharpe,
        "strat_vol": strat_vol,
        "mkt_vol": mkt_vol,
        "pos_mean": float(np.mean(positions)),
        "pos_std": float(np.std(positions)),
        "pos_min": float(np.min(positions)),
        "pos_max": float(np.max(positions)),
    }


def top_n_features(df: pd.DataFrame, target: str, n: int = 20) -> List[str]:
    """Select top-N features by absolute correlation."""
    exclude = {
        "date_id", "forward_returns", "risk_free_rate",
        "market_forward_excess_returns", "is_scored",
        "lagged_forward_returns", "lagged_risk_free_rate",
        "lagged_market_forward_excess_returns"
    }
    feats = [c for c in df.columns if c not in exclude]
    num = df[feats + [target]].select_dtypes(include=[np.number])
    corr = num.corr(numeric_only=True)[target].drop(index=target).abs().sort_values(ascending=False)
    return [c for c in corr.index[:n] if c in feats]


def run_baseline(train: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Baseline: Lasso Top-20, k=50 (EXP-002 best)"""
    target = "market_forward_excess_returns"
    top20 = top_n_features(train, target, n=20)

    tscv = TimeSeriesSplit(n_splits=5)
    rows = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(train)):
        tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]

        X_tr, scaler = preprocess(tr_df, top20)
        y_tr = tr_df[target].to_numpy()

        model = Lasso(alpha=1e-4, max_iter=50000)
        model.fit(X_tr, y_tr)

        X_va, _ = preprocess(va_df, top20)
        y_pred = model.predict(scaler.transform(X_va))

        metrics = eval_fold(y_pred, va_df, mode="excess", k=config.get("k", 50.0))
        metrics["fold"] = fold
        rows.append(metrics)

    df = pd.DataFrame(rows)
    return df


# ==================== H3a: Large k ====================
def run_H3a_large_k(train: pd.DataFrame, config: dict) -> pd.DataFrame:
    """H3a: Test large k values (200, 500, 1000, 2000)"""
    k_val = config.get("k", 200)
    target = "market_forward_excess_returns"
    top20 = top_n_features(train, target, n=20)

    tscv = TimeSeriesSplit(n_splits=5)
    rows = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(train)):
        tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]

        X_tr, scaler = preprocess(tr_df, top20)
        y_tr = tr_df[target].to_numpy()

        model = Lasso(alpha=1e-4, max_iter=50000)
        model.fit(X_tr, y_tr)

        X_va = scaler.transform(preprocess(va_df, top20)[0])
        y_pred = model.predict(X_va)

        metrics = eval_fold(y_pred, va_df, mode="excess", k=k_val)
        metrics["fold"] = fold
        rows.append(metrics)

    return pd.DataFrame(rows)


# ==================== H2a: Prediction Scaling ====================
def run_H2a_pred_scaling(train: pd.DataFrame, config: dict) -> pd.DataFrame:
    """H2a: Scale predictions to match training distribution variance"""
    k_val = config.get("k", 50)
    target = "market_forward_excess_returns"
    top20 = top_n_features(train, target, n=20)

    tscv = TimeSeriesSplit(n_splits=5)
    rows = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(train)):
        tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]

        X_tr, scaler = preprocess(tr_df, top20)
        y_tr = tr_df[target].to_numpy()

        model = Lasso(alpha=1e-4, max_iter=50000)
        model.fit(X_tr, y_tr)

        X_va = scaler.transform(preprocess(va_df, top20)[0])
        y_pred = model.predict(X_va)

        # Scale predictions to match training std
        train_std = np.std(y_tr)
        pred_std = np.std(y_pred)
        if pred_std > 1e-6:
            y_pred_scaled = y_pred * (train_std / pred_std)
        else:
            y_pred_scaled = y_pred

        metrics = eval_fold(y_pred_scaled, va_df, mode="excess", k=k_val)
        metrics["fold"] = fold
        rows.append(metrics)

    return pd.DataFrame(rows)


# ==================== H1a: Direct Position Prediction ====================
def run_H1a_pos_direct(train: pd.DataFrame, config: dict) -> pd.DataFrame:
    """H1a: Predict positions [0,2] directly instead of excess returns"""
    k_ref = config.get("k_ref", 100)  # reference k for creating target
    target = "market_forward_excess_returns"
    top20 = top_n_features(train, target, n=20)

    # Create position target
    train = train.copy()
    train["target_position"] = np.clip(1.0 + train[target] * k_ref, 0.0, 2.0)

    tscv = TimeSeriesSplit(n_splits=5)
    rows = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(train)):
        tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]

        X_tr, scaler = preprocess(tr_df, top20)
        y_tr = tr_df["target_position"].to_numpy()

        model = Lasso(alpha=1e-4, max_iter=50000)
        model.fit(X_tr, y_tr)

        X_va = scaler.transform(preprocess(va_df, top20)[0])
        y_pred = model.predict(X_va)

        metrics = eval_fold(y_pred, va_df, mode="position")
        metrics["fold"] = fold
        rows.append(metrics)

    return pd.DataFrame(rows)


# ==================== H6a: Volatility Boost ====================
def run_H6a_vol_boost(train: pd.DataFrame, config: dict) -> pd.DataFrame:
    """H6a: Boost k when volatility is low, reduce when high"""
    base_k = config.get("base_k", 200)
    target = "market_forward_excess_returns"
    top20 = top_n_features(train, target, n=20)

    tscv = TimeSeriesSplit(n_splits=5)
    rows = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(train)):
        tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]

        X_tr, scaler = preprocess(tr_df, top20)
        y_tr = tr_df[target].to_numpy()

        model = Lasso(alpha=1e-4, max_iter=50000)
        model.fit(X_tr, y_tr)

        X_va = scaler.transform(preprocess(va_df, top20)[0])
        y_pred = model.predict(X_va)

        # Calculate rolling volatility (20-day window)
        fwd_returns = va_df["forward_returns"].to_numpy()
        if len(fwd_returns) >= 20:
            current_vol = np.std(fwd_returns[-20:])
        else:
            current_vol = np.std(fwd_returns)

        # Calculate training volatility
        train_vol = np.std(tr_df["forward_returns"].to_numpy())

        # Adjust k inversely with volatility
        if current_vol > 1e-6:
            k_adjusted = base_k * (train_vol / current_vol)
            k_adjusted = np.clip(k_adjusted, base_k * 0.5, base_k * 2.0)  # limit range
        else:
            k_adjusted = base_k

        metrics = eval_fold(y_pred, va_df, mode="excess", k=k_adjusted)
        metrics["fold"] = fold
        metrics["k_adjusted"] = k_adjusted
        rows.append(metrics)

    return pd.DataFrame(rows)


# ==================== H5a: k Ensemble ====================
def run_H5a_ensemble(train: pd.DataFrame, config: dict) -> pd.DataFrame:
    """H5a: Ensemble multiple k values"""
    k_values = config.get("k_values", [50, 200, 500, 1000])
    target = "market_forward_excess_returns"
    top20 = top_n_features(train, target, n=20)

    tscv = TimeSeriesSplit(n_splits=5)
    rows = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(train)):
        tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]

        X_tr, scaler = preprocess(tr_df, top20)
        y_tr = tr_df[target].to_numpy()

        model = Lasso(alpha=1e-4, max_iter=50000)
        model.fit(X_tr, y_tr)

        X_va = scaler.transform(preprocess(va_df, top20)[0])
        y_pred = model.predict(X_va)

        # Generate positions for each k
        positions_list = []
        for k in k_values:
            pos = np.clip(1.0 + y_pred * k, 0.0, 2.0)
            positions_list.append(pos)

        # Average positions
        positions = np.mean(positions_list, axis=0)

        rf = va_df["risk_free_rate"].to_numpy()
        fwd = va_df["forward_returns"].to_numpy()
        strat = rf * (1.0 - positions) + fwd * positions
        excess = strat - rf

        mkt_vol = float(np.std(fwd))
        strat_vol = float(np.std(strat))
        vol_ratio = strat_vol / mkt_vol if mkt_vol > 0 else math.nan
        sharpe = (np.mean(excess) / strat_vol) * np.sqrt(252) if strat_vol > 0 else 0.0
        mse = float(np.mean((va_df[target].to_numpy() - y_pred) ** 2))

        metrics = {
            "fold": fold,
            "sharpe": sharpe,
            "vol_ratio": vol_ratio,
            "mse": mse,
            "strat_vol": strat_vol,
            "mkt_vol": mkt_vol,
            "pos_mean": float(np.mean(positions)),
            "pos_std": float(np.std(positions)),
            "pos_min": float(np.min(positions)),
            "pos_max": float(np.max(positions)),
        }
        rows.append(metrics)

    return pd.DataFrame(rows)


# ==================== Main Experiment Runner ====================
EXPERIMENTS = {
    "BASELINE": run_baseline,
    "H3a_large_k200": lambda t, c: run_H3a_large_k(t, {"k": 200}),
    "H3a_large_k500": lambda t, c: run_H3a_large_k(t, {"k": 500}),
    "H3a_large_k1000": lambda t, c: run_H3a_large_k(t, {"k": 1000}),
    "H3a_large_k2000": lambda t, c: run_H3a_large_k(t, {"k": 2000}),
    "H2a_pred_scaling_k50": lambda t, c: run_H2a_pred_scaling(t, {"k": 50}),
    "H2a_pred_scaling_k200": lambda t, c: run_H2a_pred_scaling(t, {"k": 200}),
    "H2a_pred_scaling_k500": lambda t, c: run_H2a_pred_scaling(t, {"k": 500}),
    "H1a_pos_direct_k50": lambda t, c: run_H1a_pos_direct(t, {"k_ref": 50}),
    "H1a_pos_direct_k100": lambda t, c: run_H1a_pos_direct(t, {"k_ref": 100}),
    "H1a_pos_direct_k200": lambda t, c: run_H1a_pos_direct(t, {"k_ref": 200}),
    "H6a_vol_boost_k200": lambda t, c: run_H6a_vol_boost(t, {"base_k": 200}),
    "H6a_vol_boost_k500": lambda t, c: run_H6a_vol_boost(t, {"base_k": 500}),
    "H5a_ensemble": lambda t, c: run_H5a_ensemble(t, {}),
}

PHASE_MAP = {
    1: ["H3a_large_k200", "H3a_large_k500", "H3a_large_k1000", "H2a_pred_scaling_k50", "H2a_pred_scaling_k200", "H1a_pos_direct_k100"],
    2: ["H5a_ensemble", "H6a_vol_boost_k200", "H6a_vol_boost_k500", "H3a_large_k2000"],
    3: ["H1a_pos_direct_k50", "H1a_pos_direct_k200", "H2a_pred_scaling_k500"],
}


def main():
    parser = argparse.ArgumentParser(description="EXP-004 Runner")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Run specific phase")
    parser.add_argument("--only", nargs="+", help="Run specific experiments")
    args = parser.parse_args()

    train = load_train()

    if args.all:
        exp_list = list(EXPERIMENTS.keys())
    elif args.phase:
        exp_list = PHASE_MAP[args.phase]
    elif args.only:
        exp_list = args.only
    else:
        exp_list = ["BASELINE"]

    summary_rows = []

    for expid in exp_list:
        if expid not in EXPERIMENTS:
            print(f"⚠️  Unknown experiment: {expid}")
            continue

        print(f"\n{'='*60}")
        print(f"Running {expid}...")
        print(f"{'='*60}")

        df = EXPERIMENTS[expid](train, {})
        out_path = RESULTS / f"{expid}_folds.csv"
        df.to_csv(out_path, index=False)
        print(f"✓ Saved: {out_path}")

        # Summary stats
        summary = {
            "experiment": expid,
            "sharpe_mean": df["sharpe"].mean(),
            "sharpe_std": df["sharpe"].std(ddof=0),
            "sharpe_median": df["sharpe"].median(),
            "vol_ratio_mean": df["vol_ratio"].mean(),
            "vol_ratio_std": df["vol_ratio"].std(ddof=0),
            "mse_mean": df["mse"].mean(),
            "pos_mean_avg": df["pos_mean"].mean(),
            "pos_std_avg": df["pos_std"].mean(),
            "pos_range_avg": (df["pos_max"] - df["pos_min"]).mean(),
        }
        summary_rows.append(summary)

        print(f"  Sharpe: {summary['sharpe_mean']:.4f} ± {summary['sharpe_std']:.4f}")
        print(f"  Vol Ratio: {summary['vol_ratio_mean']:.3f}")
        print(f"  Position: mean={summary['pos_mean_avg']:.3f}, std={summary['pos_std_avg']:.3f}, range={summary['pos_range_avg']:.3f}")

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("sharpe_mean", ascending=False)
    summary_path = RESULTS / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"✓ Summary saved: {summary_path}")
    print(f"{'='*60}\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
