#!/usr/bin/env python3
"""EXP-016 Phase 1.2: Null Importance Test

Run null importance test to identify truly significant features.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Import from feature_analysis
sys.path.insert(0, str(Path(__file__).parent))
from feature_analysis import load_exp007_features, null_importance_test

def main():
    print("="*80)
    print("EXP-016 Phase 1.2: Null Importance Test")
    print("="*80)
    print()
    print("This will run 100 iterations of target shuffling.")
    print("Estimated time: 30-60 minutes")
    print()

    # Create output directory
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading data and creating 754 features...")
    X, y, feature_names = load_exp007_features()

    print()
    print("-"*80)
    print("Running Null Importance Test (100 iterations)...")
    print("-"*80)
    print()

    # Run null importance test
    null_df = null_importance_test(
        X, y,
        output_dir,
        n_iterations=100,
        p_threshold=0.05
    )

    print()
    print("="*80)
    print("Phase 1.2 Complete!")
    print("="*80)
    print()

    # Summary statistics
    n_significant = null_df['significant'].sum()
    n_total = len(null_df)
    pct_significant = 100 * n_significant / n_total

    print(f"Total features: {n_total}")
    print(f"Significant features (p < 0.05): {n_significant} ({pct_significant:.1f}%)")
    print(f"Removed features: {n_total - n_significant} ({100 - pct_significant:.1f}%)")
    print()

    # Top 10 significant features
    significant_features = null_df[null_df['significant']].sort_values(
        'real_importance', ascending=False
    )

    print("Top 10 significant features:")
    for i, row in significant_features.head(10).iterrows():
        print(f"  {i+1}. {row['feature']}: p={row['p_value']:.4f}, importance={row['real_importance']:.6f}")
    print()

    # Bottom 10 (non-significant)
    non_significant = null_df[~null_df['significant']].sort_values(
        'p_value', ascending=False
    )

    if len(non_significant) > 0:
        print("Bottom 10 (non-significant, highest p-values):")
        for i, row in non_significant.head(10).iterrows():
            print(f"  {i+1}. {row['feature']}: p={row['p_value']:.4f}, importance={row['real_importance']:.6f}")
        print()

    print("Results saved to:")
    print(f"  - {output_dir / 'null_importance_test.csv'}")
    print()

    # Save significant features list
    significant_features_list = significant_features['feature'].tolist()
    with open(output_dir / 'significant_features.txt', 'w') as f:
        for feat in significant_features_list:
            f.write(f"{feat}\n")

    print(f"  - {output_dir / 'significant_features.txt'}")
    print()
    print("Next: Phase 1.3 - Feature Correlation Analysis")
    print("="*80)

if __name__ == '__main__':
    main()
