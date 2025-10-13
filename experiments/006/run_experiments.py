#!/usr/bin/env python3
"""EXP-006: k Parameter Optimization for 17.395 Target

Phase 1: k-grid search (300~3000)
Based on EXP-005 best model (H3: XGBoost + Feature Engineering)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("[ERROR] XGBoost not available. Please install: uv pip install xgboost")

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


def create_lag_features(df: pd.DataFrame, features: List[str], lags: List[int] = [1, 5, 10]) -> pd.DataFrame:
    """Create lag features for time series."""
    df_new = df.copy()
    for col in features:
        for lag in lags:
            df_new[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df_new


def create_rolling_features(df: pd.DataFrame, features: List[str], windows: List[int] = [5, 10]) -> pd.DataFrame:
    """Create rolling statistics features."""
    df_new = df.copy()
    for col in features:
        for window in windows:
            df_new[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
            df_new[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
    return df_new


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
    k: float = 50.0,
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
# Phase 1: k Parameter Grid Search
# ============================================================================
def run_phase1_k_grid(
    train: pd.DataFrame,
    k_values: List[float] = [200, 300, 400, 500, 600, 800]
) -> pd.DataFrame:
    """Phase 1: k-grid search using EXP-005 best model (XGBoost + Feature Eng).

    Target: Find optimal k for 17.395 Kaggle score
    Strategy: Start from EXP-005 best (k=200, Sharpe 0.627) and increase gradually
    """
    if not HAS_XGB:
        print("[ERROR] XGBoost not available.")
        return pd.DataFrame()

    print("\n" + "="*80)
    print("EXP-006 Phase 1: k Parameter Grid Search")
    print("="*80)
    print(f"Goal: Kaggle 17.395 (current best: 0.724)")
    print(f"Testing k values: {k_values}")

    target = "market_forward_excess_returns"

    # Use EXP-005 H3 feature engineering strategy
    base_features = top_n_features(train, target, n=20)
    print(f"[INFO] Base features: {len(base_features)}")

    # Create engineered features
    train_eng = train.copy()
    train_eng = create_lag_features(train_eng, base_features, lags=[1, 5, 10])
    train_eng = create_rolling_features(train_eng, base_features, windows=[5, 10])

    all_features = [c for c in train_eng.columns if c not in {
        "date_id", "forward_returns", "risk_free_rate",
        "market_forward_excess_returns", "is_scored",
        "lagged_forward_returns", "lagged_risk_free_rate",
        "lagged_market_forward_excess_returns"
    }]
    print(f"[INFO] Total features after engineering: {len(all_features)}")

    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    for k_val in k_values:
        print(f"\n[INFO] Testing k={k_val}")
        fold_metrics = []

        for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(train_eng), 1):
            tr_df = train_eng.iloc[tr_idx]
            va_df = train_eng.iloc[va_idx]

            X_tr, scaler = preprocess(tr_df, all_features)
            y_tr = tr_df[target].to_numpy()

            X_va_raw = va_df[all_features].copy()
            X_va_raw = X_va_raw.fillna(X_va_raw.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
            X_va = scaler.transform(X_va_raw)

            # Train XGBoost (same hyperparameters as EXP-005 H3)
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

            metrics = eval_fold(y_pred, va_df, k=k_val)
            metrics.update({"fold": fold_idx, "k": k_val, "phase": "phase1_k_grid"})
            fold_metrics.append(metrics)

            print(f"  Fold {fold_idx}/5: Sharpe {metrics['sharpe']:.3f}, "
                  f"Vol Ratio {metrics['vol_ratio']:.3f}, "
                  f"Pos Std {metrics['pos_std']:.3f}")

        results.extend(fold_metrics)

        # Summary for this k
        sharpes = [m['sharpe'] for m in fold_metrics]
        vol_ratios = [m['vol_ratio'] for m in fold_metrics]
        pos_stds = [m['pos_std'] for m in fold_metrics]

        avg_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes)
        avg_vol = np.mean(vol_ratios)
        avg_pos_std = np.mean(pos_stds)

        print(f"[RESULT] k={k_val}: Sharpe {avg_sharpe:.3f} ± {std_sharpe:.3f}, "
              f"Vol Ratio {avg_vol:.3f}, Pos Std {avg_pos_std:.3f}")

        # Decision making
        if k_val >= 500:
            if avg_sharpe < 0.5:
                print(f"[WARNING] k={k_val} Sharpe too low ({avg_sharpe:.3f}). Consider stopping.")
            elif avg_vol > 2.0:
                print(f"[WARNING] k={k_val} Vol Ratio too high ({avg_vol:.3f}). Risky.")

        print(f"[COMPARISON] vs EXP-005 k=200 (0.627): {(avg_sharpe/0.627 - 1)*100:+.1f}%")

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS / "phase1_k_grid.csv", index=False)
    print(f"\n[SAVED] {RESULTS / 'phase1_k_grid.csv'}")

    return df_results


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="EXP-006: k Parameter Optimization")
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        help="Phase to run (1: k-grid 300-800, 2: vol scaling, 3: feature eng)"
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="200,300,400,500,600,800",
        help="Comma-separated k values to test (default: 200,300,400,500,600,800)"
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Include aggressive k values (1000, 1500, 2000, 3000)"
    )
    args = parser.parse_args()

    # Parse k values
    k_values = [float(k) for k in args.k_values.split(",")]

    if args.aggressive:
        k_values.extend([1000, 1500, 2000, 3000])
        print("[INFO] Aggressive mode enabled. Testing high k values.")

    print("[INFO] Loading training data...")
    train = load_train()
    print(f"[INFO] Train shape: {train.shape}")

    all_results = []

    if args.phase == 1:
        print("\n[INFO] Starting Phase 1: k-grid search")
        df_phase1 = run_phase1_k_grid(train, k_values=k_values)
        all_results.append(df_phase1)
    else:
        print(f"[ERROR] Phase {args.phase} not implemented yet.")
        print("[INFO] Available phases: 1 (k-grid search)")
        return

    # Generate summary
    if all_results:
        all_results = [df for df in all_results if not df.empty]

        if all_results:
            df_all = pd.concat(all_results, ignore_index=True)
            summary = df_all.groupby(["k"]).agg({
                "sharpe": ["mean", "std"],
                "vol_ratio": ["mean", "std"],
                "pos_std": ["mean", "std"],
                "mse": "mean"
            }).round(4)

            print("\n" + "="*80)
            print("PHASE 1 SUMMARY - k Parameter Grid Search")
            print("="*80)
            print(summary)
            print()

            # Best k by Sharpe
            avg_sharpe = df_all.groupby("k")["sharpe"].mean()
            best_k = avg_sharpe.idxmax()
            best_sharpe = avg_sharpe.max()

            print(f"[BEST] k={best_k}: Sharpe {best_sharpe:.3f}")
            print(f"[BASELINE] EXP-005 k=200: Sharpe 0.627")
            print(f"[IMPROVEMENT] {(best_sharpe/0.627 - 1)*100:+.1f}%")
            print()

            # Decision guidance
            if best_sharpe > 0.70:
                print("[DECISION] ✅ Good progress! Suggest:")
                print(f"  1. Submit k={best_k} to Kaggle")
                print("  2. If Kaggle score > 3.0, try Phase 1b (k=1000+)")
                print("  3. If Kaggle score < 1.5, try Phase 2 (Vol Scaling)")
            elif best_sharpe > 0.65:
                print("[DECISION] 📊 Moderate improvement. Suggest:")
                print(f"  1. Submit k={best_k} to Kaggle (expect 1.0-2.5)")
                print("  2. Move to Phase 2 (Vol Scaling)")
            else:
                print("[DECISION] ⚠️ Limited improvement. Suggest:")
                print("  1. k increase not effective")
                print("  2. Prioritize Phase 2 (Vol Scaling) and Phase 3 (Feature Eng)")

            summary.to_csv(RESULTS / "summary.csv")
            print(f"\n[SAVED] {RESULTS / 'summary.csv'}")
        else:
            print("\n[WARNING] No experiments completed successfully")

    print("\n[DONE] EXP-006 Phase 1 completed!")
    print(f"[NEXT] Review results in {RESULTS}/")


if __name__ == "__main__":
    main()
