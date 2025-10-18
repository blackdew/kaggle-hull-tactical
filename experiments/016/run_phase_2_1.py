#!/usr/bin/env python3
"""EXP-016 Phase 2.1: Interaction Features

Add interaction features to Top 20 baseline.
Goal: Sharpe 0.874 → 1.0+
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from itertools import combinations

# Import from feature_analysis
sys.path.insert(0, str(Path(__file__).parent))
from feature_analysis import load_exp007_features

try:
    from xgboost import XGBRegressor
except ImportError:
    print("[ERROR] XGBoost not available")
    sys.exit(1)


def create_interaction_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """Create interaction features from Top features.

    Args:
        df: Input dataframe
        feature_list: List of feature names to create interactions from

    Returns:
        DataFrame with interaction features added
    """
    df_new = df[feature_list].copy()

    print(f"Creating interaction features from {len(feature_list)} features...")

    # All pairs (combinations without replacement)
    pairs = list(combinations(feature_list, 2))
    print(f"  Total pairs: {len(pairs)}")

    interaction_count = 0

    for f1, f2 in pairs:
        # Multiply
        df_new[f'{f1}_X_{f2}'] = df[f1] * df[f2]
        interaction_count += 1

        # Divide (with small epsilon to avoid division by zero)
        df_new[f'{f1}_DIV_{f2}'] = df[f1] / (df[f2].abs() + 1e-8)
        interaction_count += 1

        # Add
        df_new[f'{f1}_ADD_{f2}'] = df[f1] + df[f2]
        interaction_count += 1

        # Subtract
        df_new[f'{f1}_SUB_{f2}'] = df[f1] - df[f2]
        interaction_count += 1

    print(f"  Created {interaction_count} interaction features")
    print(f"  Total features: {len(feature_list)} (original) + {interaction_count} (interactions) = {len(df_new.columns)}")

    return df_new


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


def run_experiment(X: pd.DataFrame, y: pd.Series, train_df: pd.DataFrame,
                   experiment_name: str, k: float = 600.0) -> pd.DataFrame:
    """Run 3-fold CV experiment."""
    print(f"\n{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"Features: {X.shape[1]}")
    print(f"{'='*80}")

    tscv = TimeSeriesSplit(n_splits=3)
    results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        print(f"  Fold {fold_idx}/3...", end=" ", flush=True)

        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
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
            "n_features": X.shape[1]
        })
        results.append(metrics)

        print(f"Sharpe: {metrics['sharpe']:.3f}")

    df_results = pd.DataFrame(results)
    avg_sharpe = df_results['sharpe'].mean()
    std_sharpe = df_results['sharpe'].std()
    print(f"\n  Average Sharpe: {avg_sharpe:.3f} ± {std_sharpe:.3f}")

    return df_results, model


def main():
    print("="*80)
    print("EXP-016 Phase 2.1: Interaction Features")
    print("="*80)
    print()
    print("Baseline: Top 20 features (Sharpe 0.874)")
    print("Goal: Add interactions → Sharpe 1.0+")
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

    # Experiment 1: Baseline (Top 20 only)
    print("="*80)
    print("Baseline: Top 20 features")
    print("="*80)
    X_top20 = X_full[top_20_features]
    result_baseline, _ = run_experiment(X_top20, y, train, "Top_20_baseline")

    # Experiment 2: Top 20 + Interactions
    print("\n" + "="*80)
    print("Creating interaction features...")
    print("="*80)
    X_with_interactions = create_interaction_features(X_full, top_20_features)

    print()
    result_interactions, model_interactions = run_experiment(
        X_with_interactions, y, train, "Top_20_with_interactions"
    )

    # Compare results
    print("\n" + "="*80)
    print("Results Comparison")
    print("="*80)

    baseline_sharpe = result_baseline['sharpe'].mean()
    interactions_sharpe = result_interactions['sharpe'].mean()

    diff = interactions_sharpe - baseline_sharpe
    pct = 100 * diff / baseline_sharpe if baseline_sharpe != 0 else 0

    print(f"\nBaseline (Top 20):          Sharpe {baseline_sharpe:.3f}")
    print(f"With Interactions:          Sharpe {interactions_sharpe:.3f}")
    print(f"Difference:                 {diff:+.3f} ({pct:+.1f}%)")
    print()

    # Feature importance of interactions
    print("="*80)
    print("Feature Importance Analysis")
    print("="*80)

    feature_importance = model_interactions.feature_importances_
    feature_names = X_with_interactions.columns

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    # Top 20 interactions
    interaction_features = importance_df[importance_df['feature'].str.contains('_X_|_DIV_|_ADD_|_SUB_')]
    top_20_interactions = interaction_features.head(20)

    print("\nTop 20 Interaction Features:")
    for i, row in enumerate(top_20_interactions.iterrows(), 1):
        _, data = row
        print(f"  {i:2d}. {data['feature']}: {data['importance']:.6f}")

    # Save results
    all_results = pd.concat([result_baseline, result_interactions], ignore_index=True)
    all_results.to_csv(output_dir / 'phase_2_1_results.csv', index=False)
    importance_df.to_csv(output_dir / 'phase_2_1_importance.csv', index=False)
    top_20_interactions.to_csv(output_dir / 'top_20_interactions.csv', index=False)

    print()
    print("="*80)
    print("Phase 2.1 Complete!")
    print("="*80)
    print()

    # Decision
    if interactions_sharpe >= 1.0:
        print("🎉 **SUCCESS! Sharpe 1.0+ achieved!**")
        print("   Next: Feature selection + Hyperparameter tuning")
    elif interactions_sharpe >= 0.95:
        print("📊 **Close to goal!** (Sharpe 0.95+)")
        print("   Next: Phase 2.2 (Polynomial features) or Phase 3 (Hyperparameter tuning)")
    elif interactions_sharpe > baseline_sharpe:
        print("✅ **Improvement!** Interactions helped.")
        print(f"   Sharpe: {baseline_sharpe:.3f} → {interactions_sharpe:.3f}")
        print("   Next: Phase 2.2 (Polynomial features)")
    else:
        print("⚠️ **No improvement.** Interactions didn't help.")
        print("   Next: Try different approach or Phase 3 (Hyperparameter tuning)")

    print()
    print(f"Results saved to: {output_dir}/")
    print("  - phase_2_1_results.csv")
    print("  - phase_2_1_importance.csv")
    print("  - top_20_interactions.csv")
    print("="*80)


if __name__ == '__main__':
    main()
