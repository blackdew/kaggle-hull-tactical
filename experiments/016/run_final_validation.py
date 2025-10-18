#!/usr/bin/env python3
"""EXP-016 Final Validation: 5-fold CV

Validate best model with 5-fold CV for robustness check.
Best params from Phase 3.3 (Sharpe 1.001 on 3-fold CV)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Import from feature_analysis
sys.path.insert(0, str(Path(__file__).parent))
from feature_analysis import load_exp007_features

try:
    from xgboost import XGBRegressor
except ImportError:
    print("[ERROR] XGBoost not available")
    sys.exit(1)


def eval_fold(y_pred: np.ndarray, y_true: np.ndarray, fwd_returns: np.ndarray,
              risk_free: np.ndarray, k: float = 600.0) -> dict:
    """Evaluate predictions on validation fold."""
    positions = np.clip(1.0 + y_pred * k, 0.0, 2.0)
    strat = risk_free * (1.0 - positions) + fwd_returns * positions
    excess = strat - risk_free

    mkt_vol = float(np.std(fwd_returns))
    strat_vol = float(np.std(strat))

    vol_ratio = strat_vol / mkt_vol if mkt_vol > 0 else np.nan
    sharpe = (np.mean(excess) / strat_vol) * np.sqrt(252) if strat_vol > 0 else 0.0
    mse = float(np.mean((y_true - y_pred) ** 2))

    return {
        "mse": mse,
        "vol_ratio": vol_ratio,
        "sharpe": sharpe,
    }


def main():
    print("="*80)
    print("EXP-016 Final Validation: 5-fold CV")
    print("="*80)
    print()
    print("Best model from Phase 3.3:")
    print("  - Features: Top 20")
    print("  - 3-fold CV Sharpe: 1.001")
    print()
    print("Goal: Confirm robustness with 5-fold CV")
    print()

    # Create output directory
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading data...")
    X_full, y, all_feature_names = load_exp007_features()
    train = pd.read_csv("data/train.csv")

    # Load Top 20 features
    with open(output_dir / 'top_50_common_features.txt', 'r') as f:
        top_50 = [line.strip() for line in f if line.strip()]

    top_20_features = top_50[:20]
    print(f"Top 20 features loaded: {len(top_20_features)}")
    print()

    # Load best hyperparameters
    best_params_df = pd.read_csv(output_dir / 'best_hyperparameters.csv')
    best_params = best_params_df.iloc[0].to_dict()

    # Convert to proper types
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])
    best_params['random_state'] = int(best_params['random_state'])
    best_params['verbosity'] = int(best_params['verbosity'])
    best_params['n_jobs'] = int(best_params['n_jobs'])

    print("Best hyperparameters:")
    for key, value in best_params.items():
        if key not in ['random_state', 'tree_method', 'verbosity', 'n_jobs']:
            print(f"  {key}: {value}")
    print()

    # Prepare data
    X_top20 = X_full[top_20_features]

    # 3-fold CV (baseline comparison)
    print("="*80)
    print("3-fold CV (baseline)")
    print("="*80)

    tscv_3 = TimeSeriesSplit(n_splits=3)
    results_3fold = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv_3.split(X_top20), 1):
        print(f"  Fold {fold_idx}/3...", end=" ", flush=True)

        X_tr = X_top20.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X_top20.iloc[va_idx]
        y_va = y.iloc[va_idx]

        # Preprocess
        X_tr_filled = X_tr.fillna(X_tr.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_filled)

        X_va_filled = X_va.fillna(X_va.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_va_scaled = scaler.transform(X_va_filled)

        # Train model
        model = XGBRegressor(**best_params)
        model.fit(X_tr_scaled, y_tr)
        y_pred = model.predict(X_va_scaled)

        # Evaluate
        va_df = train.iloc[va_idx]
        metrics = eval_fold(
            y_pred,
            y_va.values,
            va_df["forward_returns"].values,
            va_df["risk_free_rate"].values,
            k=600.0
        )
        metrics.update({
            "fold": fold_idx,
            "cv_type": "3-fold",
            "n_features": X_top20.shape[1]
        })
        results_3fold.append(metrics)

        print(f"Sharpe: {metrics['sharpe']:.3f}")

    df_3fold = pd.DataFrame(results_3fold)
    avg_sharpe_3 = df_3fold['sharpe'].mean()
    std_sharpe_3 = df_3fold['sharpe'].std()
    print(f"\n  Average Sharpe: {avg_sharpe_3:.3f} ± {std_sharpe_3:.3f}")

    # 5-fold CV (robustness check)
    print("\n" + "="*80)
    print("5-fold CV (robustness check)")
    print("="*80)

    tscv_5 = TimeSeriesSplit(n_splits=5)
    results_5fold = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv_5.split(X_top20), 1):
        print(f"  Fold {fold_idx}/5...", end=" ", flush=True)

        X_tr = X_top20.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X_top20.iloc[va_idx]
        y_va = y.iloc[va_idx]

        # Preprocess
        X_tr_filled = X_tr.fillna(X_tr.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_filled)

        X_va_filled = X_va.fillna(X_va.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_va_scaled = scaler.transform(X_va_filled)

        # Train model
        model = XGBRegressor(**best_params)
        model.fit(X_tr_scaled, y_tr)
        y_pred = model.predict(X_va_scaled)

        # Evaluate
        va_df = train.iloc[va_idx]
        metrics = eval_fold(
            y_pred,
            y_va.values,
            va_df["forward_returns"].values,
            va_df["risk_free_rate"].values,
            k=600.0
        )
        metrics.update({
            "fold": fold_idx,
            "cv_type": "5-fold",
            "n_features": X_top20.shape[1]
        })
        results_5fold.append(metrics)

        print(f"Sharpe: {metrics['sharpe']:.3f}")

    df_5fold = pd.DataFrame(results_5fold)
    avg_sharpe_5 = df_5fold['sharpe'].mean()
    std_sharpe_5 = df_5fold['sharpe'].std()
    print(f"\n  Average Sharpe: {avg_sharpe_5:.3f} ± {std_sharpe_5:.3f}")

    # Comparison
    print("\n" + "="*80)
    print("Results Comparison")
    print("="*80)

    print(f"\n3-fold CV:  Sharpe {avg_sharpe_3:.3f} ± {std_sharpe_3:.3f}")
    print(f"5-fold CV:  Sharpe {avg_sharpe_5:.3f} ± {std_sharpe_5:.3f}")

    diff = avg_sharpe_5 - avg_sharpe_3
    print(f"\nDifference: {diff:+.3f}")

    # Save results
    all_results = pd.concat([df_3fold, df_5fold], ignore_index=True)
    all_results.to_csv(output_dir / 'final_validation.csv', index=False)

    print()
    print("="*80)
    print("Final Validation Complete!")
    print("="*80)
    print()

    # Decision
    if avg_sharpe_5 >= 1.0:
        print("✅ **5-fold CV confirms Sharpe 1.0+!**")
        print(f"   5-fold CV Sharpe: {avg_sharpe_5:.3f}")
        print("   **Robust result confirmed!**")
    elif avg_sharpe_5 >= 0.95:
        print("📊 **5-fold CV shows strong performance (0.95+)**")
        print(f"   5-fold CV Sharpe: {avg_sharpe_5:.3f}")
        print("   Result is robust")
    elif avg_sharpe_5 >= avg_sharpe_3 * 0.95:
        print("✅ **5-fold CV confirms robustness**")
        print(f"   Within 5% of 3-fold CV result")
        print("   Result is stable")
    else:
        print("⚠️ **5-fold CV shows some variability**")
        print(f"   5-fold: {avg_sharpe_5:.3f} vs 3-fold: {avg_sharpe_3:.3f}")
        print("   Consider: More data or simpler model")

    print()
    print(f"Results saved to: {output_dir}/final_validation.csv")
    print("="*80)

    # Summary
    print()
    print("="*80)
    print("EXP-016 Final Summary")
    print("="*80)
    print(f"EXP-007 baseline:          Sharpe 0.749")
    print(f"Phase 1 (Top 20):          Sharpe 0.874 (+16.7%)")
    print(f"Phase 3 (Optimized):       Sharpe {avg_sharpe_3:.3f} (+{100*(avg_sharpe_3/0.749-1):.1f}%)")
    print(f"Final (5-fold CV):         Sharpe {avg_sharpe_5:.3f} (+{100*(avg_sharpe_5/0.749-1):.1f}%)")
    print()
    print(f"Target (Sharpe 1.0):       {'✅ Achieved!' if avg_sharpe_5 >= 1.0 else f'{100*(1.0-avg_sharpe_5)/avg_sharpe_5:.1f}% gap'}")
    print("="*80)


if __name__ == '__main__':
    main()
