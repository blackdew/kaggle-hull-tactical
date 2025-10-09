#!/usr/bin/env python3
"""EXP-005: Lasso with Feature Engineering (Fallback when XGBoost unavailable)"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def load_train(path: str = "data/train.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def select_features(df: pd.DataFrame) -> List[str]:
    exclude = {
        "date_id", "forward_returns", "risk_free_rate",
        "market_forward_excess_returns", "is_scored",
        "lagged_forward_returns", "lagged_risk_free_rate",
        "lagged_market_forward_excess_returns"
    }
    return [c for c in df.columns if c not in exclude]


def top_n_features(df: pd.DataFrame, target: str, n: int = 20) -> List[str]:
    feats = select_features(df)
    num = df[feats + [target]].select_dtypes(include=[np.number])
    corr = num.corr(numeric_only=True)[target].drop(index=target).abs().sort_values(ascending=False)
    return [c for c in corr.index[:n] if c in feats]


def create_engineered_features(df: pd.DataFrame, base_features: List[str]) -> pd.DataFrame:
    """Create lag, rolling, and interaction features."""
    df_new = df.copy()

    # Lag features (1, 5)
    for col in base_features[:10]:  # Top-10만
        df_new[f'{col}_lag1'] = df[col].shift(1)
        df_new[f'{col}_lag5'] = df[col].shift(5)

    # Rolling features (5, 10)
    for col in base_features[:10]:
        df_new[f'{col}_rolling_mean_5'] = df[col].rolling(5).mean()
        df_new[f'{col}_rolling_std_5'] = df[col].rolling(5).std()

    # Interaction features (top pairs)
    if len(base_features) >= 2:
        df_new[f'{base_features[0]}_x_{base_features[1]}'] = df[base_features[0]] * df[base_features[1]]
    if len(base_features) >= 4:
        df_new[f'{base_features[2]}_x_{base_features[3]}'] = df[base_features[2]] * df[base_features[3]]

    return df_new


def preprocess(df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, StandardScaler]:
    X = df[features].copy()
    X = X.fillna(X.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def eval_fold(y_pred: np.ndarray, valid_df: pd.DataFrame, k: float = 50.0) -> Dict[str, float]:
    rf = valid_df["risk_free_rate"].to_numpy()
    fwd = valid_df["forward_returns"].to_numpy()

    positions = np.clip(1.0 + y_pred * k, 0.0, 2.0)

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
        "pos_mean": float(np.mean(positions)),
        "pos_std": float(np.std(positions)),
        "pos_min": float(np.min(positions)),
        "pos_max": float(np.max(positions)),
    }


def run_lasso_with_feature_eng(train: pd.DataFrame, k_values: List[float] = [50, 100, 150]) -> pd.DataFrame:
    """Lasso + Feature Engineering."""
    print("\n" + "="*80)
    print("Lasso + Feature Engineering")
    print("="*80)

    target = "market_forward_excess_returns"
    base_features = top_n_features(train, target, n=20)
    print(f"[INFO] Base features: {len(base_features)}")

    # Create engineered features
    train_eng = create_engineered_features(train, base_features)

    # Select all features
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

            X_va = va_df[all_features].copy()
            X_va = X_va.fillna(scaler.mean_).replace([np.inf, -np.inf], np.nan).fillna(0)
            X_va = scaler.transform(X_va)

            model = Lasso(alpha=1e-4, max_iter=50000)
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_va)

            metrics = eval_fold(y_pred, va_df, k=k_val)
            metrics.update({"fold": fold_idx, "k": k_val, "hypothesis": "Lasso_FeatEng"})
            fold_metrics.append(metrics)

            print(f"  Fold {fold_idx}/5: Sharpe {metrics['sharpe']:.3f}, Vol Ratio {metrics['vol_ratio']:.3f}")

        results.extend(fold_metrics)

        sharpes = [m['sharpe'] for m in fold_metrics]
        print(f"[RESULT] k={k_val}: Sharpe {np.mean(sharpes):.3f} ± {np.std(sharpes):.3f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS / "lasso_feature_eng_folds.csv", index=False)
    print(f"[SAVED] {RESULTS / 'lasso_feature_eng_folds.csv'}")

    # Summary
    summary = df_results.groupby("k").agg({
        "sharpe": ["mean", "std"],
        "vol_ratio": ["mean", "std"],
        "mse": "mean"
    }).round(4)
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(summary)

    summary.to_csv(RESULTS / "lasso_summary.csv")
    print(f"\n[SAVED] {RESULTS / 'lasso_summary.csv'}")

    # Find best k
    best_row = df_results.groupby("k")["sharpe"].mean().idxmax()
    best_sharpe = df_results.groupby("k")["sharpe"].mean().max()
    print(f"\n[BEST] k={best_row}, Sharpe={best_sharpe:.3f}")

    return df_results


def main():
    print("[INFO] Loading training data...")
    train = load_train()
    print(f"[INFO] Train shape: {train.shape}")

    df_results = run_lasso_with_feature_eng(train, k_values=[50, 100, 150])

    print("\n[DONE] EXP-005 Lasso experiments completed!")
    print("[NOTE] XGBoost/LightGBM unavailable - used Lasso with feature engineering instead")


if __name__ == "__main__":
    main()
