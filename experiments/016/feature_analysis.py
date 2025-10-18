#!/usr/bin/env python3
"""EXP-016 Phase 1: Feature Analysis

Deep dive into 754 features from EXP-007:
1. SHAP analysis
2. Permutation importance
3. Null importance test
4. Feature correlation
5. Baseline comparison with feature subsets
6. Feature group analysis
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
except ImportError:
    print("[ERROR] XGBoost not available. Install: pip install xgboost")
    sys.exit(1)

try:
    import shap
except ImportError:
    print("[WARNING] SHAP not available. Install: pip install shap")
    shap = None


def load_exp007_features() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load and create 754 features from EXP-007 approach.

    Returns:
        X: Feature dataframe (754 features)
        y: Target series
        feature_names: List of feature names
    """
    print("[INFO] Loading data and creating 754 features...")

    # Load training data
    train = pd.read_csv("data/train.csv")

    # Select target
    target = "market_forward_excess_returns"
    y = train[target]

    # Select base features (top 20 by correlation, like EXP-007)
    exclude = {
        "date_id", "forward_returns", "risk_free_rate",
        "market_forward_excess_returns", "is_scored",
        "lagged_forward_returns", "lagged_risk_free_rate",
        "lagged_market_forward_excess_returns"
    }
    all_cols = [c for c in train.columns if c not in exclude]

    # Select top 20 features by absolute correlation
    num_df = train[all_cols + [target]].select_dtypes(include=[np.number])
    corr = num_df.corr(numeric_only=True)[target].drop(index=target).abs()
    corr = corr.sort_values(ascending=False)
    base_features = [c for c in corr.index[:20] if c in all_cols]

    print(f"[INFO] Base features: {len(base_features)}")
    print(f"[INFO] Top 5: {base_features[:5]}")

    # Create all extended features (like EXP-007 H1_all)
    sys.path.insert(0, "experiments/007")
    from feature_engineering import create_all_features

    train_eng = create_all_features(
        train,
        base_features,
        enable_lags=True,
        enable_rolling=True,
        enable_cross_sectional=True,
        enable_volatility=True,
        enable_momentum=True,
        date_col='date_id'
    )

    # Select all features (exclude metadata and target)
    feature_names = [c for c in train_eng.columns if c not in exclude]

    # Prepare X
    X = train_eng[feature_names].copy()

    # Preprocessing: fill NaN, clip inf
    X = X.fillna(X.median(numeric_only=True))
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"[INFO] Total features: {len(feature_names)}")
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")

    return X, y, feature_names


def shap_analysis(
    model: XGBRegressor,
    X: pd.DataFrame,
    output_dir: Path,
    top_n: int = 100
) -> pd.DataFrame:
    """1.1 SHAP Analysis

    Args:
        model: Trained XGBoost model
        X: Feature dataframe
        output_dir: Directory to save results
        top_n: Number of top features to extract

    Returns:
        DataFrame with feature names and SHAP importance
    """
    if shap is None:
        print("[ERROR] SHAP not installed")
        return pd.DataFrame()

    print("[1.1] Running SHAP analysis...")

    # Use TreeExplainer for fast computation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Calculate mean absolute SHAP values
    shap_importance = np.abs(shap_values).mean(axis=0)

    # Create dataframe
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'shap_importance': shap_importance
    }).sort_values('shap_importance', ascending=False)

    # Save top features
    top_features = feature_importance_df.head(top_n)
    top_features.to_csv(output_dir / 'shap_top_features.csv', index=False)

    # Save summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  - Top {top_n} features extracted")
    print(f"  - Results saved to {output_dir / 'shap_top_features.csv'}")

    return feature_importance_df


def permutation_importance_analysis(
    model: XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    top_n: int = 100,
    n_repeats: int = 10
) -> pd.DataFrame:
    """1.1 Permutation Importance

    Args:
        model: Trained XGBoost model
        X: Feature dataframe
        y: Target series
        output_dir: Directory to save results
        top_n: Number of top features
        n_repeats: Number of permutation repeats

    Returns:
        DataFrame with feature names and permutation importance
    """
    print("[1.1] Running Permutation Importance...")

    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1
    )

    # Create dataframe
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'perm_importance_mean': perm_importance.importances_mean,
        'perm_importance_std': perm_importance.importances_std
    }).sort_values('perm_importance_mean', ascending=False)

    # Save top features
    top_features = feature_importance_df.head(top_n)
    top_features.to_csv(output_dir / 'perm_top_features.csv', index=False)

    print(f"  - Top {top_n} features extracted")
    print(f"  - Results saved to {output_dir / 'perm_top_features.csv'}")

    return feature_importance_df


def xgboost_builtin_importance(
    model: XGBRegressor,
    output_dir: Path,
    top_n: int = 100
) -> pd.DataFrame:
    """1.1 XGBoost Built-in Feature Importance

    Args:
        model: Trained XGBoost model
        output_dir: Directory to save results
        top_n: Number of top features

    Returns:
        DataFrame with feature names and importance
    """
    print("[1.1] Extracting XGBoost built-in feature importance...")

    # Get feature importance (gain)
    importance_dict = model.get_booster().get_score(importance_type='gain')

    # Create dataframe
    feature_importance_df = pd.DataFrame({
        'feature': list(importance_dict.keys()),
        'xgb_gain': list(importance_dict.values())
    }).sort_values('xgb_gain', ascending=False)

    # Save top features
    top_features = feature_importance_df.head(top_n)
    top_features.to_csv(output_dir / 'xgb_top_features.csv', index=False)

    print(f"  - Top {top_n} features extracted")
    print(f"  - Results saved to {output_dir / 'xgb_top_features.csv'}")

    return feature_importance_df


def compare_importance_methods(
    shap_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    xgb_df: pd.DataFrame,
    output_dir: Path
) -> pd.DataFrame:
    """Compare 3 feature importance methods

    Args:
        shap_df: SHAP importance dataframe
        perm_df: Permutation importance dataframe
        xgb_df: XGBoost importance dataframe
        output_dir: Directory to save results

    Returns:
        Combined dataframe with all importance scores
    """
    print("[1.1] Comparing 3 importance methods...")

    # Merge all methods
    combined = shap_df.merge(
        perm_df[['feature', 'perm_importance_mean']],
        on='feature',
        how='outer'
    ).merge(
        xgb_df[['feature', 'xgb_gain']],
        on='feature',
        how='outer'
    ).fillna(0)

    # Normalize to 0-1 scale
    for col in ['shap_importance', 'perm_importance_mean', 'xgb_gain']:
        max_val = combined[col].max()
        if max_val > 0:
            combined[f'{col}_norm'] = combined[col] / max_val

    # Calculate average rank
    combined['avg_rank'] = (
        combined['shap_importance_norm'] +
        combined['perm_importance_mean_norm'] +
        combined['xgb_gain_norm']
    ) / 3

    # Sort by average rank
    combined = combined.sort_values('avg_rank', ascending=False)

    # Save
    combined.to_csv(output_dir / 'feature_importance_comparison.csv', index=False)

    # Extract top 50 common features
    top_50 = combined.head(50)['feature'].tolist()
    with open(output_dir / 'top_50_common_features.txt', 'w') as f:
        for feat in top_50:
            f.write(f"{feat}\n")

    print(f"  - Top 50 common features saved")
    print(f"  - Correlation between methods:")
    print(f"    SHAP vs Perm: {combined['shap_importance'].corr(combined['perm_importance_mean']):.3f}")
    print(f"    SHAP vs XGB: {combined['shap_importance'].corr(combined['xgb_gain']):.3f}")
    print(f"    Perm vs XGB: {combined['perm_importance_mean'].corr(combined['xgb_gain']):.3f}")

    return combined


def null_importance_test(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    n_iterations: int = 100,
    p_threshold: float = 0.05
) -> pd.DataFrame:
    """1.2 Null Importance Test

    Shuffle target and calculate feature importance to identify truly significant features.

    Args:
        X: Feature dataframe
        y: Target series
        output_dir: Directory to save results
        n_iterations: Number of shuffle iterations
        p_threshold: P-value threshold

    Returns:
        DataFrame with features and p-values
    """
    print(f"[1.2] Running Null Importance Test ({n_iterations} iterations)...")

    # Train model on real data
    model_real = XGBRegressor(random_state=42, n_jobs=-1)
    model_real.fit(X, y)
    real_importance = model_real.feature_importances_

    # Shuffle and collect null importances
    null_importances = []

    for i in range(n_iterations):
        if (i + 1) % 10 == 0:
            print(f"  - Iteration {i + 1}/{n_iterations}")

        # Shuffle target
        y_shuffled = y.sample(frac=1, random_state=i).values

        # Train model on shuffled data
        model_null = XGBRegressor(random_state=42, n_jobs=-1)
        model_null.fit(X, y_shuffled)

        null_importances.append(model_null.feature_importances_)

    # Calculate p-values
    null_importances = np.array(null_importances)
    p_values = []

    for i, feat in enumerate(X.columns):
        # P-value: proportion of null importances >= real importance
        p_val = (null_importances[:, i] >= real_importance[i]).mean()
        p_values.append(p_val)

    # Create dataframe
    result_df = pd.DataFrame({
        'feature': X.columns,
        'real_importance': real_importance,
        'null_mean': null_importances.mean(axis=0),
        'null_std': null_importances.std(axis=0),
        'p_value': p_values,
        'significant': np.array(p_values) < p_threshold
    }).sort_values('real_importance', ascending=False)

    # Save results
    result_df.to_csv(output_dir / 'null_importance_test.csv', index=False)

    n_significant = result_df['significant'].sum()
    print(f"  - Significant features (p < {p_threshold}): {n_significant}/{len(X.columns)}")
    print(f"  - Results saved to {output_dir / 'null_importance_test.csv'}")

    return result_df


# TODO: Implement remaining functions
# - feature_correlation_analysis (1.3)
# - baseline_comparison (1.4)
# - feature_group_analysis (1.5)
# - train_cv_importance_gap (1.6)


def main():
    """Main execution for Phase 1.1"""
    print("="*80)
    print("EXP-016 Phase 1.1: Feature Importance Analysis")
    print("="*80)
    print()

    # Create output directory
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    # Load data and create 754 features
    X, y, feature_names = load_exp007_features()

    print()
    print("-"*80)
    print("Training XGBoost model on all features...")
    print("-"*80)

    # Train baseline model (same as EXP-007)
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

    model.fit(X, y)
    print("[INFO] Model training complete")
    print()

    # Run analyses
    print("-"*80)
    print("Running 3 importance methods...")
    print("-"*80)
    print()

    # 1. SHAP analysis
    shap_df = shap_analysis(model, X.sample(n=min(5000, len(X)), random_state=42), output_dir, top_n=100)

    # 2. Permutation importance
    # Use subset for speed
    X_subset = X.sample(n=min(5000, len(X)), random_state=42)
    y_subset = y.loc[X_subset.index]
    perm_df = permutation_importance_analysis(model, X_subset, y_subset, output_dir, top_n=100, n_repeats=10)

    # 3. XGBoost built-in
    xgb_df = xgboost_builtin_importance(model, output_dir, top_n=100)

    print()
    print("-"*80)
    print("Comparing 3 methods...")
    print("-"*80)

    # 4. Compare methods
    combined_df = compare_importance_methods(shap_df, perm_df, xgb_df, output_dir)

    print()
    print("="*80)
    print("Phase 1.1 Complete!")
    print("="*80)
    print(f"Results saved to: {output_dir}/")
    print("  - shap_top_features.csv")
    print("  - perm_top_features.csv")
    print("  - xgb_top_features.csv")
    print("  - feature_importance_comparison.csv")
    print("  - top_50_common_features.txt")
    print()
    print("Next: Phase 1.2 - Null Importance Test")
    print("  Run: python feature_analysis.py --phase 1.2")
    print("="*80)


if __name__ == '__main__':
    main()
