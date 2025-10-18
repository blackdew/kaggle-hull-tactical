#!/usr/bin/env python3
"""EXP-016 Phase 1.4: Baseline Comparison

Compare performance with different feature subsets:
1. All 754 features (baseline)
2. Top 100 features (from Phase 1.1)
3. Top 50 features (from Phase 1.1)
4. Top 20 features (from Phase 1.1)
5. 57 Significant features (from Phase 1.2)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

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
    # Convert excess return predictions to positions
    positions = np.clip(1.0 + y_pred * k, 0.0, 2.0)

    # Calculate strategy returns
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
        "strat_vol": strat_vol,
        "mkt_vol": mkt_vol,
        "pos_mean": float(np.mean(positions)),
        "pos_std": float(np.std(positions)),
    }


def run_experiment(X: pd.DataFrame, y: pd.Series, train_df: pd.DataFrame,
                   feature_names: List[str], experiment_name: str, k: float = 600.0) -> pd.DataFrame:
    """Run 3-fold CV experiment with given features."""
    print(f"\n{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"Features: {len(feature_names)}")
    print(f"{'='*80}")

    # Select features
    X_subset = X[feature_names].copy()

    tscv = TimeSeriesSplit(n_splits=3)
    results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X_subset), 1):
        print(f"  Fold {fold_idx}/3...", end=" ", flush=True)

        X_tr = X_subset.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X_subset.iloc[va_idx]
        y_va = y.iloc[va_idx]

        # Preprocess
        X_tr_filled = X_tr.fillna(X_tr.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_filled)

        X_va_filled = X_va.fillna(X_va.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_va_scaled = scaler.transform(X_va_filled)

        # Train model
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            verbosity=0,
            n_jobs=-1
        )

        model.fit(X_tr_scaled, y_tr)

        # Predict
        y_pred = model.predict(X_va_scaled)

        # Evaluate
        va_df = train_df.iloc[va_idx]
        metrics = eval_fold(
            y_pred,
            y_va.values,
            va_df["forward_returns"].values,
            va_df["risk_free_rate"].values,
            k=k
        )
        metrics.update({
            "fold": fold_idx,
            "experiment": experiment_name,
            "n_features": len(feature_names)
        })
        results.append(metrics)

        print(f"Sharpe: {metrics['sharpe']:.3f}")

    df_results = pd.DataFrame(results)
    avg_sharpe = df_results['sharpe'].mean()
    std_sharpe = df_results['sharpe'].std()
    print(f"\n  Average Sharpe: {avg_sharpe:.3f} ± {std_sharpe:.3f}")

    return df_results


def main():
    print("="*80)
    print("EXP-016 Phase 1.4: Baseline Comparison")
    print("="*80)
    print()

    # Create output directory
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading data and creating 754 features...")
    X, y, all_feature_names = load_exp007_features()

    # Load original train data for evaluation
    train = pd.read_csv("data/train.csv")

    print()
    print("Loading feature subsets...")

    # Load Top 50 from Phase 1.1
    with open(output_dir / 'top_50_common_features.txt', 'r') as f:
        top_50_features = [line.strip() for line in f if line.strip()]

    # Top 100, Top 20 from top 50
    top_100_features = top_50_features[:min(100, len(top_50_features))]  # Will be 50 if only 50 available
    top_20_features = top_50_features[:20]

    # Load 57 significant features from Phase 1.2
    with open(output_dir / 'significant_features.txt', 'r') as f:
        significant_features = [line.strip() for line in f if line.strip()]

    print(f"  All features: {len(all_feature_names)}")
    print(f"  Top 50: {len(top_50_features)}")
    print(f"  Top 20: {len(top_20_features)}")
    print(f"  Significant (p<0.05): {len(significant_features)}")

    # Run experiments
    all_results = []

    # 1. Baseline: All 754 features
    result_754 = run_experiment(X, y, train, all_feature_names, "All_754_features")
    all_results.append(result_754)

    # 2. Top 50 features
    result_50 = run_experiment(X, y, train, top_50_features, "Top_50_features")
    all_results.append(result_50)

    # 3. Top 20 features
    result_20 = run_experiment(X, y, train, top_20_features, "Top_20_features")
    all_results.append(result_20)

    # 4. 57 Significant features
    result_sig = run_experiment(X, y, train, significant_features, "Significant_57_features")
    all_results.append(result_sig)

    # Combine all results
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(output_dir / 'baseline_comparison.csv', index=False)

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    summary = df_all.groupby('experiment').agg({
        'sharpe': ['mean', 'std'],
        'mse': 'mean',
        'vol_ratio': 'mean',
        'n_features': 'first'
    }).round(4)

    print(summary)
    print()

    # Compare to baseline
    baseline_sharpe = df_all[df_all['experiment'] == 'All_754_features']['sharpe'].mean()

    print("Comparison to Baseline (754 features):")
    for exp_name in df_all['experiment'].unique():
        if exp_name == 'All_754_features':
            continue
        exp_sharpe = df_all[df_all['experiment'] == exp_name]['sharpe'].mean()
        diff = exp_sharpe - baseline_sharpe
        pct = 100 * diff / baseline_sharpe if baseline_sharpe != 0 else 0
        n_feat = df_all[df_all['experiment'] == exp_name]['n_features'].iloc[0]

        symbol = "✅" if diff >= -0.05 else "⚠️" if diff >= -0.1 else "❌"
        print(f"  {symbol} {exp_name}: {exp_sharpe:.3f} ({diff:+.3f}, {pct:+.1f}%) with {n_feat} features")

    print()
    print("Results saved to:")
    print(f"  - {output_dir / 'baseline_comparison.csv'}")
    print()
    print("="*80)
    print("Phase 1.4 Complete!")
    print("="*80)
    print()

    # Key findings
    print("Key Findings:")
    print()

    sig_sharpe = df_all[df_all['experiment'] == 'Significant_57_features']['sharpe'].mean()
    sig_diff = sig_sharpe - baseline_sharpe

    if abs(sig_diff) < 0.05:
        print("✅ **57 significant features perform similarly to 754 features!**")
        print(f"   → Null test was correct: {len(significant_features)}/754 features are truly useful")
        print(f"   → 697 features are likely overfitting/noise")
    elif sig_diff < -0.1:
        print("⚠️ **57 significant features underperform 754 features**")
        print(f"   → Null test might be too strict (p<0.05)")
        print(f"   → Some removed features might be useful")
    else:
        print("📊 **57 significant features slightly underperform**")
        print(f"   → Acceptable trade-off for {13}x fewer features")

    print()
    print("Next: Phase 1.3 (Feature Correlation) or Phase 2 (Feature Engineering)")
    print("="*80)


if __name__ == '__main__':
    main()
