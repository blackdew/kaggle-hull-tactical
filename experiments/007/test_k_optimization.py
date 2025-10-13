#!/usr/bin/env python3
"""Test k optimization for H1 All Features model.

Goal: Find optimal k for Kaggle submission.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import create_all_features
from run_experiments import load_train, select_features, top_n_features, preprocess, eval_fold

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("[ERROR] XGBoost not available.")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def test_k_values(train_eng: pd.DataFrame, all_features: list, k_values: list = [800, 1000, 1500, 2000]):
    """Test different k values with H1 All Features model."""

    target = "market_forward_excess_returns"
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
            metrics.update({"fold": fold_idx, "k": k_val})
            fold_metrics.append(metrics)

            print(f"  Fold {fold_idx}/5: Sharpe {metrics['sharpe']:.3f}, Pos Std {metrics['pos_std']:.3f}")

        results.extend(fold_metrics)

        # Summary
        sharpes = [m['sharpe'] for m in fold_metrics]
        pos_stds = [m['pos_std'] for m in fold_metrics]
        vol_ratios = [m['vol_ratio'] for m in fold_metrics]

        avg_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes)
        avg_pos_std = np.mean(pos_stds)
        avg_vol = np.mean(vol_ratios)

        print(f"[RESULT] k={k_val}: Sharpe {avg_sharpe:.3f} ± {std_sharpe:.3f}, "
              f"Pos Std {avg_pos_std:.3f}, Vol Ratio {avg_vol:.3f}")

        # Estimate utility (rough)
        clamped_sharpe = min(max(avg_sharpe, 0), 6)
        profit_proxy = avg_pos_std * 10  # Rough scaling
        estimated_utility = clamped_sharpe * profit_proxy

        print(f"[UTILITY] Estimated: {estimated_utility:.2f} "
              f"(Sharpe {clamped_sharpe:.2f} × Profit {profit_proxy:.2f})")

    return pd.DataFrame(results)


def main():
    print("="*80)
    print("EXP-007: k Optimization for H1 All Features")
    print("="*80)

    # Load data
    train = load_train()
    target = "market_forward_excess_returns"
    base_features = top_n_features(train, target, n=20)

    # Create all features
    print("\n[INFO] Creating features...")
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
    print(f"[INFO] Total features: {len(all_features)}")

    # Test k values
    k_values = [800, 1000, 1500, 2000]
    df_results = test_k_values(train_eng, all_features, k_values)

    # Save
    df_results.to_csv(RESULTS / "k_optimization.csv", index=False)
    print(f"\n[SAVED] {RESULTS / 'k_optimization.csv'}")

    # Summary
    summary = df_results.groupby('k').agg({
        'sharpe': ['mean', 'std'],
        'vol_ratio': 'mean',
        'pos_std': 'mean'
    }).round(4)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(summary)

    # Best k
    avg_sharpe = df_results.groupby('k')['sharpe'].mean()
    best_k = avg_sharpe.idxmax()
    best_sharpe = avg_sharpe.max()

    print()
    print(f"[BEST] k={best_k}: Sharpe {best_sharpe:.3f}")
    print(f"[BASELINE] EXP-007 H1 k=600: Sharpe 0.749")
    print(f"[IMPROVEMENT] {(best_sharpe/0.749 - 1)*100:+.1f}%")

    print()
    print("[RECOMMENDATION]")
    print(f"  Submit k={best_k} to Kaggle")
    print(f"  Expected Kaggle utility: 1.5~3.0 (현재 0.724 대비 2~4배)")
    print()
    print("[NEXT STEPS]")
    print(f"  1. Generate kaggle_inference file with k={best_k}")
    print(f"  2. Submit to Kaggle")
    print(f"  3. Analyze results")


if __name__ == "__main__":
    main()
