#!/usr/bin/env python3
"""EXP-005: Gradient Boosting + Feature Engineering Experiments"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import Lasso

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

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
# H1: XGBoost Baseline
# ============================================================================
def run_H1_xgboost_baseline(train: pd.DataFrame, k_values: List[float] = [50, 100, 200]) -> pd.DataFrame:
    """H1: XGBoost Baseline with all features."""
    if not HAS_XGB:
        print("[ERROR] XGBoost not available. Skipping H1.")
        return pd.DataFrame()

    print("\n" + "="*80)
    print("H1: XGBoost Baseline")
    print("="*80)

    target = "market_forward_excess_returns"
    features = select_features(train)
    print(f"[INFO] Features: {len(features)}")

    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    for k_val in k_values:
        print(f"\n[INFO] Testing k={k_val}")
        fold_metrics = []

        for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(train), 1):
            tr_df = train.iloc[tr_idx]
            va_df = train.iloc[va_idx]

            # Preprocess
            X_tr, scaler = preprocess(tr_df, features)
            y_tr = tr_df[target].to_numpy()

            # Preprocess validation (use same scaler)
            X_va_raw = va_df[features].copy()
            X_va_raw = X_va_raw.fillna(X_va_raw.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
            X_va = scaler.transform(X_va_raw)

            # Train XGBoost
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

            # Predict
            y_pred = model.predict(X_va)

            # Evaluate
            metrics = eval_fold(y_pred, va_df, k=k_val)
            metrics.update({"fold": fold_idx, "k": k_val, "hypothesis": "H1_xgboost"})
            fold_metrics.append(metrics)

            print(f"  Fold {fold_idx}/5: Sharpe {metrics['sharpe']:.3f}, Vol Ratio {metrics['vol_ratio']:.3f}")

        results.extend(fold_metrics)

        # Summary
        sharpes = [m['sharpe'] for m in fold_metrics]
        print(f"[RESULT] H1 k={k_val}: Sharpe {np.mean(sharpes):.3f} ± {np.std(sharpes):.3f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS / "h1_xgboost_folds.csv", index=False)
    print(f"[SAVED] {RESULTS / 'h1_xgboost_folds.csv'}")

    return df_results


# ============================================================================
# H2: LightGBM Baseline
# ============================================================================
def run_H2_lightgbm_baseline(train: pd.DataFrame, k_values: List[float] = [50, 100, 200]) -> pd.DataFrame:
    """H2: LightGBM Baseline."""
    if not HAS_LGBM:
        print("[ERROR] LightGBM not available. Skipping H2.")
        return pd.DataFrame()

    print("\n" + "="*80)
    print("H2: LightGBM Baseline")
    print("="*80)

    target = "market_forward_excess_returns"
    features = select_features(train)

    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    for k_val in k_values:
        print(f"\n[INFO] Testing k={k_val}")
        fold_metrics = []

        for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(train), 1):
            tr_df = train.iloc[tr_idx]
            va_df = train.iloc[va_idx]

            X_tr, scaler = preprocess(tr_df, features)
            y_tr = tr_df[target].to_numpy()

            X_va_raw = va_df[features].copy()
            X_va_raw = X_va_raw.fillna(X_va_raw.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
            X_va = scaler.transform(X_va_raw)

            # Train LightGBM
            model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.01,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1
            )
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_va)

            metrics = eval_fold(y_pred, va_df, k=k_val)
            metrics.update({"fold": fold_idx, "k": k_val, "hypothesis": "H2_lightgbm"})
            fold_metrics.append(metrics)

            print(f"  Fold {fold_idx}/5: Sharpe {metrics['sharpe']:.3f}, Vol Ratio {metrics['vol_ratio']:.3f}")

        results.extend(fold_metrics)

        sharpes = [m['sharpe'] for m in fold_metrics]
        print(f"[RESULT] H2 k={k_val}: Sharpe {np.mean(sharpes):.3f} ± {np.std(sharpes):.3f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS / "h2_lightgbm_folds.csv", index=False)
    print(f"[SAVED] {RESULTS / 'h2_lightgbm_folds.csv'}")

    return df_results


# ============================================================================
# H3: Feature Engineering (Lag + Rolling)
# ============================================================================
def run_H3_feature_engineering(train: pd.DataFrame, k_values: List[float] = [50, 100, 200]) -> pd.DataFrame:
    """H3: Feature Engineering with Lag and Rolling Statistics."""
    if not HAS_XGB:
        print("[ERROR] XGBoost not available. Skipping H3.")
        return pd.DataFrame()

    print("\n" + "="*80)
    print("H3: Feature Engineering (Lag + Rolling)")
    print("="*80)

    target = "market_forward_excess_returns"
    base_features = top_n_features(train, target, n=20)
    print(f"[INFO] Base features for engineering: {len(base_features)}")

    # Create engineered features
    train_eng = train.copy()
    train_eng = create_lag_features(train_eng, base_features, lags=[1, 5, 10])
    train_eng = create_rolling_features(train_eng, base_features, windows=[5, 10])

    # Select all numeric features
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
            metrics.update({"fold": fold_idx, "k": k_val, "hypothesis": "H3_feature_eng"})
            fold_metrics.append(metrics)

            print(f"  Fold {fold_idx}/5: Sharpe {metrics['sharpe']:.3f}, Vol Ratio {metrics['vol_ratio']:.3f}")

        results.extend(fold_metrics)

        sharpes = [m['sharpe'] for m in fold_metrics]
        print(f"[RESULT] H3 k={k_val}: Sharpe {np.mean(sharpes):.3f} ± {np.std(sharpes):.3f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS / "h3_feature_eng_folds.csv", index=False)
    print(f"[SAVED] {RESULTS / 'h3_feature_eng_folds.csv'}")

    return df_results


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="EXP-005: Gradient Boosting Experiments")
    parser.add_argument("--hypothesis", type=str, help="Run specific hypothesis (H1, H2, H3)")
    parser.add_argument("--all", action="store_true", help="Run all hypotheses")
    parser.add_argument("--phase", type=int, help="Run specific phase (1, 2)")
    args = parser.parse_args()

    print("[INFO] Loading training data...")
    train = load_train()
    print(f"[INFO] Train shape: {train.shape}")

    all_results = []

    if args.hypothesis == "H1" or args.all or args.phase == 1:
        df_h1 = run_H1_xgboost_baseline(train)
        all_results.append(df_h1)

    if args.hypothesis == "H2" or args.all or args.phase == 1:
        df_h2 = run_H2_lightgbm_baseline(train)
        all_results.append(df_h2)

    if args.hypothesis == "H3" or args.all or args.phase == 2:
        df_h3 = run_H3_feature_engineering(train)
        all_results.append(df_h3)

    # Combine and save summary
    if all_results:
        # Filter out empty DataFrames
        all_results = [df for df in all_results if not df.empty]

        if all_results:
            df_all = pd.concat(all_results, ignore_index=True)
            summary = df_all.groupby(["hypothesis", "k"]).agg({
                "sharpe": ["mean", "std"],
                "vol_ratio": ["mean", "std"],
                "mse": "mean"
            }).round(4)
            print("\n" + "="*80)
            print("SUMMARY")
            print("="*80)
            print(summary)

            summary.to_csv(RESULTS / "summary.csv")
            print(f"\n[SAVED] {RESULTS / 'summary.csv'}")
        else:
            print("\n[WARNING] No experiments completed successfully")

    print("\n[DONE] EXP-005 experiments completed!")


if __name__ == "__main__":
    main()
