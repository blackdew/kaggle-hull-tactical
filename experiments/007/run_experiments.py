#!/usr/bin/env python3
"""EXP-007: Extended Feature Engineering Experiments

Goal: Sharpe 1.5~3.0 (현재 0.7 대비 2~4배)
Strategy: 예측 정확도 근본 개선

Phase 1: Feature Engineering (H1a~d)
- H1a: Longer Lags (20, 40, 60)
- H1b: Cross-Sectional (rank, zscore, quantile)
- H1c: Volatility Features
- H1d: Momentum & Trend
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Import feature engineering module
sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import (
    create_all_features,
    create_lag_features,
    create_rolling_features,
    create_cross_sectional_features,
    create_volatility_features,
    create_momentum_features,
)

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("[ERROR] XGBoost not available. Please install: pip install xgboost")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def load_train(path: str = "data/train.csv") -> pd.DataFrame:
    """Load training data."""
    return pd.read_csv(path)


def select_features(df: pd.DataFrame) -> List[str]:
    """Select features, excluding targets and metadata."""
    exclude = {
        "date_id", "forward_returns", "risk_free_rate",
        "market_forward_excess_returns", "is_scored",
        "lagged_forward_returns", "lagged_risk_free_rate",
        "lagged_market_forward_excess_returns"
    }
    return [c for c in df.columns if c not in exclude]


def top_n_features(df: pd.DataFrame, target: str, n: int = 20) -> List[str]:
    """Select top-N features by absolute correlation."""
    feats = select_features(df)
    num = df[feats + [target]].select_dtypes(include=[np.number])
    corr = num.corr(numeric_only=True)[target].drop(index=target).abs().sort_values(ascending=False)
    return [c for c in corr.index[:n] if c in feats]


def preprocess(df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, StandardScaler]:
    """Preprocess features: impute, clip, scale."""
    X = df[features].copy()
    X = X.fillna(X.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def eval_fold(
    y_pred: np.ndarray,
    valid_df: pd.DataFrame,
    k: float = 200.0,
) -> Dict[str, float]:
    """Evaluate predictions on validation fold."""
    rf = valid_df["risk_free_rate"].to_numpy()
    fwd = valid_df["forward_returns"].to_numpy()

    # Convert excess return predictions to positions
    positions = np.clip(1.0 + y_pred * k, 0.0, 2.0)

    # Calculate strategy returns
    strat = rf * (1.0 - positions) + fwd * positions
    excess = strat - rf

    mkt_vol = float(np.std(fwd))
    strat_vol = float(np.std(strat))

    vol_ratio = strat_vol / mkt_vol if mkt_vol > 0 else math.nan
    sharpe = (np.mean(excess) / strat_vol) * np.sqrt(252) if strat_vol > 0 else 0.0
    mse = float(np.mean((valid_df["market_forward_excess_returns"].to_numpy() - y_pred) ** 2))

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


# ============================================================================
# Baseline: EXP-006 best (k=600, simple features)
# ============================================================================
def run_baseline(train: pd.DataFrame, k: float = 600.0) -> pd.DataFrame:
    """Baseline: EXP-006 best performance for comparison.

    Same as EXP-005 H3: XGBoost + simple feature engineering
    """
    if not HAS_XGB:
        print("[ERROR] XGBoost not available.")
        return pd.DataFrame()

    print("\n" + "="*80)
    print("Baseline: EXP-006 Best (k=600, simple features)")
    print("="*80)

    target = "market_forward_excess_returns"
    base_features = top_n_features(train, target, n=20)

    # Simple feature engineering (EXP-005 style)
    train_eng = train.copy()
    train_eng = create_lag_features(train_eng, base_features, lags=[1, 5, 10])
    train_eng = create_rolling_features(train_eng, base_features, windows=[5, 10])

    all_features = select_features(train_eng)
    print(f"[INFO] Total features: {len(all_features)}")

    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(train_eng), 1):
        tr_df = train_eng.iloc[tr_idx]
        va_df = train_eng.iloc[va_idx]

        X_tr, scaler = preprocess(tr_df, all_features)
        y_tr = tr_df[target].to_numpy()

        X_va_raw = va_df[all_features].copy()
        X_va_raw = X_va_raw.fillna(X_va_raw.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_va = scaler.transform(X_va_raw)

        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            verbosity=0
        )
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_va)

        metrics = eval_fold(y_pred, va_df, k=k)
        metrics.update({"fold": fold_idx, "k": k, "hypothesis": "baseline"})
        results.append(metrics)

        print(f"  Fold {fold_idx}/5: Sharpe {metrics['sharpe']:.3f}")

    df_results = pd.DataFrame(results)
    avg_sharpe = df_results['sharpe'].mean()
    print(f"[RESULT] Baseline k={k}: Sharpe {avg_sharpe:.3f}")
    print()

    df_results.to_csv(RESULTS / "baseline.csv", index=False)
    return df_results


# ============================================================================
# H1a: Longer Lags
# ============================================================================
def run_H1a_longer_lags(train: pd.DataFrame, k: float = 600.0) -> pd.DataFrame:
    """H1a: Add longer lag features (20, 40, 60)."""
    if not HAS_XGB:
        return pd.DataFrame()

    print("\n" + "="*80)
    print("H1a: Longer Lags (1, 5, 10, 20, 40, 60)")
    print("="*80)

    target = "market_forward_excess_returns"
    base_features = top_n_features(train, target, n=20)

    # Extended lags
    train_eng = train.copy()
    train_eng = create_lag_features(train_eng, base_features, lags=[1, 5, 10, 20, 40, 60])
    train_eng = create_rolling_features(train_eng, base_features, windows=[5, 10])

    all_features = select_features(train_eng)
    print(f"[INFO] Total features: {len(all_features)}")

    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(train_eng), 1):
        tr_df = train_eng.iloc[tr_idx]
        va_df = train_eng.iloc[va_idx]

        X_tr, scaler = preprocess(tr_df, all_features)
        y_tr = tr_df[target].to_numpy()

        X_va_raw = va_df[all_features].copy()
        X_va_raw = X_va_raw.fillna(X_va_raw.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_va = scaler.transform(X_va_raw)

        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            verbosity=0
        )
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_va)

        metrics = eval_fold(y_pred, va_df, k=k)
        metrics.update({"fold": fold_idx, "k": k, "hypothesis": "H1a_longer_lags"})
        results.append(metrics)

        print(f"  Fold {fold_idx}/5: Sharpe {metrics['sharpe']:.3f}")

    df_results = pd.DataFrame(results)
    avg_sharpe = df_results['sharpe'].mean()
    print(f"[RESULT] H1a: Sharpe {avg_sharpe:.3f}")
    print()

    df_results.to_csv(RESULTS / "h1a_longer_lags.csv", index=False)
    return df_results


# ============================================================================
# H1b: Cross-Sectional Features
# ============================================================================
def run_H1b_cross_sectional(train: pd.DataFrame, k: float = 600.0) -> pd.DataFrame:
    """H1b: Add cross-sectional features (rank, zscore, quantile)."""
    if not HAS_XGB:
        return pd.DataFrame()

    print("\n" + "="*80)
    print("H1b: Cross-Sectional Features (rank, zscore, quantile)")
    print("="*80)

    target = "market_forward_excess_returns"
    base_features = top_n_features(train, target, n=20)

    # Cross-sectional features
    train_eng = train.copy()
    train_eng = create_lag_features(train_eng, base_features, lags=[1, 5, 10])
    train_eng = create_rolling_features(train_eng, base_features, windows=[5, 10])
    train_eng = create_cross_sectional_features(train_eng, base_features, date_col='date_id')

    all_features = select_features(train_eng)
    print(f"[INFO] Total features: {len(all_features)}")

    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(train_eng), 1):
        tr_df = train_eng.iloc[tr_idx]
        va_df = train_eng.iloc[va_idx]

        X_tr, scaler = preprocess(tr_df, all_features)
        y_tr = tr_df[target].to_numpy()

        X_va_raw = va_df[all_features].copy()
        X_va_raw = X_va_raw.fillna(X_va_raw.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_va = scaler.transform(X_va_raw)

        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            verbosity=0
        )
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_va)

        metrics = eval_fold(y_pred, va_df, k=k)
        metrics.update({"fold": fold_idx, "k": k, "hypothesis": "H1b_cross_sectional"})
        results.append(metrics)

        print(f"  Fold {fold_idx}/5: Sharpe {metrics['sharpe']:.3f}")

    df_results = pd.DataFrame(results)
    avg_sharpe = df_results['sharpe'].mean()
    print(f"[RESULT] H1b: Sharpe {avg_sharpe:.3f}")
    print()

    df_results.to_csv(RESULTS / "h1b_cross_sectional.csv", index=False)
    return df_results


# ============================================================================
# H1: All Features Combined
# ============================================================================
def run_H1_all_features(train: pd.DataFrame, k: float = 600.0) -> pd.DataFrame:
    """H1: All extended features combined."""
    if not HAS_XGB:
        return pd.DataFrame()

    print("\n" + "="*80)
    print("H1: All Extended Features Combined")
    print("="*80)

    target = "market_forward_excess_returns"
    base_features = top_n_features(train, target, n=20)
    print(f"[INFO] Base features for engineering: {len(base_features)}")

    # Create all features
    train_eng = create_all_features(
        train.copy(),
        base_features,
        enable_lags=True,
        enable_rolling=True,
        enable_cross_sectional=True,
        enable_volatility=True,
        enable_momentum=True,
        date_col='date_id'
    )

    all_features = select_features(train_eng)
    print(f"[INFO] Total features for training: {len(all_features)}")

    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(train_eng), 1):
        tr_df = train_eng.iloc[tr_idx]
        va_df = train_eng.iloc[va_idx]

        X_tr, scaler = preprocess(tr_df, all_features)
        y_tr = tr_df[target].to_numpy()

        X_va_raw = va_df[all_features].copy()
        X_va_raw = X_va_raw.fillna(X_va_raw.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_va = scaler.transform(X_va_raw)

        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            verbosity=0
        )
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_va)

        metrics = eval_fold(y_pred, va_df, k=k)
        metrics.update({"fold": fold_idx, "k": k, "hypothesis": "H1_all_features"})
        results.append(metrics)

        print(f"  Fold {fold_idx}/5: Sharpe {metrics['sharpe']:.3f}, MSE {metrics['mse']:.6f}")

    df_results = pd.DataFrame(results)
    avg_sharpe = df_results['sharpe'].mean()
    avg_mse = df_results['mse'].mean()
    print(f"[RESULT] H1 All: Sharpe {avg_sharpe:.3f}, MSE {avg_mse:.6f}")
    print()

    df_results.to_csv(RESULTS / "h1_all_features.csv", index=False)
    return df_results


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="EXP-007: Extended Feature Engineering")
    parser.add_argument(
        "--hypothesis",
        type=str,
        choices=["baseline", "H1a", "H1b", "H1"],
        help="Which hypothesis to run"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all hypotheses"
    )
    parser.add_argument(
        "--k",
        type=float,
        default=600.0,
        help="k parameter (default: 600.0)"
    )
    args = parser.parse_args()

    print("[INFO] Loading training data...")
    train = load_train()
    print(f"[INFO] Train shape: {train.shape}")

    all_results = []

    if args.hypothesis == "baseline" or args.all:
        df_baseline = run_baseline(train, k=args.k)
        all_results.append(df_baseline)

    if args.hypothesis == "H1a" or args.all:
        df_h1a = run_H1a_longer_lags(train, k=args.k)
        all_results.append(df_h1a)

    if args.hypothesis == "H1b" or args.all:
        df_h1b = run_H1b_cross_sectional(train, k=args.k)
        all_results.append(df_h1b)

    if args.hypothesis == "H1" or args.all:
        df_h1 = run_H1_all_features(train, k=args.k)
        all_results.append(df_h1)

    # Generate summary
    if all_results:
        all_results = [df for df in all_results if not df.empty]

        if all_results:
            df_all = pd.concat(all_results, ignore_index=True)
            summary = df_all.groupby("hypothesis").agg({
                "sharpe": ["mean", "std"],
                "mse": "mean",
                "vol_ratio": "mean"
            }).round(4)

            print("\n" + "="*80)
            print("EXP-007 SUMMARY")
            print("="*80)
            print(summary)
            print()

            # Best hypothesis
            avg_sharpe = df_all.groupby("hypothesis")["sharpe"].mean()
            best_hyp = avg_sharpe.idxmax()
            best_sharpe = avg_sharpe.max()

            print(f"[BEST] {best_hyp}: Sharpe {best_sharpe:.3f}")
            print(f"[BASELINE] EXP-006: Sharpe 0.665")
            print(f"[IMPROVEMENT] {(best_sharpe/0.665 - 1)*100:+.1f}%")
            print()

            # Decision guidance
            if best_sharpe >= 1.5:
                print("[DECISION] 🎯 TARGET ACHIEVED! Sharpe >= 1.5")
                print("  → Submit to Kaggle")
                print("  → Optimize k parameter")
            elif best_sharpe >= 1.0:
                print("[DECISION] ✅ Good progress! Sharpe >= 1.0")
                print("  → Try Phase 2 (Volatility Scaling)")
                print("  → Or submit current best")
            elif best_sharpe >= 0.85:
                print("[DECISION] 📊 Moderate improvement")
                print("  → Phase 2 (Vol Scaling) + Phase 3 (Ensemble) needed")
            else:
                print("[DECISION] ⚠️ Limited improvement")
                print("  → Consider H4 (Neural Network) or H5 (Target Engineering)")

            summary.to_csv(RESULTS / "summary.csv")
            print(f"\n[SAVED] {RESULTS / 'summary.csv'}")
        else:
            print("\n[WARNING] No experiments completed successfully")

    print("\n[DONE] EXP-007 experiments completed!")


if __name__ == "__main__":
    main()
