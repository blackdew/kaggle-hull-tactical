#!/usr/bin/env python3
"""Analyze Phase 1 results focusing on Profit and Utility.

Key insight: Kaggle metric is utility = min(max(sharpe, 0), 6) × Σ profits
Not just Sharpe optimization!
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def calculate_utility(sharpe: float, total_profit: float) -> float:
    """Calculate Kaggle utility score.

    utility = min(max(sharpe, 0), 6) × total_profit
    """
    clamped_sharpe = min(max(sharpe, 0), 6)
    return clamped_sharpe * total_profit


def estimate_profit_from_positions(fold_data: pd.DataFrame) -> float:
    """Estimate total profit from position statistics.

    Simplified: profit ∝ pos_mean × (market return assumption)
    More accurate analysis needs actual returns data.
    """
    # Rough estimate: assuming market excess return ~0.01 per day
    # profit ≈ (pos_mean - 1) × market_return × n_days
    # This is a placeholder - need actual fold returns for precise calculation
    pos_mean = fold_data['pos_mean'].mean()
    n_days = len(fold_data)

    # Placeholder: normalize to estimate relative profit
    # Higher pos_mean → more exposure → more profit (in theory)
    estimated_profit = (pos_mean - 1.0) * n_days * 0.001

    return estimated_profit


def analyze_phase1_results():
    """Analyze Phase 1 results with profit and utility focus."""

    # Load Phase 1 results
    df = pd.read_csv(RESULTS / "phase1_k_grid.csv")

    print("="*80)
    print("EXP-006 Phase 1 Analysis: Profit & Utility Focus")
    print("="*80)
    print()
    print("Key Insight: Kaggle metric = min(max(sharpe, 0), 6) × Σ profits")
    print("Current approach was WRONG: optimizing sharpe only!")
    print()

    # Group by k
    summary = df.groupby('k').agg({
        'sharpe': ['mean', 'std'],
        'vol_ratio': 'mean',
        'pos_mean': 'mean',
        'pos_std': 'mean',
        'mse': 'mean'
    }).round(4)

    print("Phase 1 Summary by k:")
    print(summary)
    print()

    # Calculate per-k metrics
    results = []

    for k in sorted(df['k'].unique()):
        k_data = df[df['k'] == k]

        sharpe_mean = k_data['sharpe'].mean()
        sharpe_std = k_data['sharpe'].std()
        pos_mean = k_data['pos_mean'].mean()
        pos_std = k_data['pos_std'].mean()
        vol_ratio = k_data['vol_ratio'].mean()

        # Estimate profit (placeholder - need actual returns)
        # For now, use position metrics as proxy
        # Higher k → higher pos_std → more position variation → potentially more profit
        estimated_profit_score = pos_std * 10  # Rough scaling

        # Calculate utility with clamped Sharpe
        clamped_sharpe = min(max(sharpe_mean, 0), 6)
        estimated_utility = clamped_sharpe * estimated_profit_score

        results.append({
            'k': k,
            'sharpe_mean': sharpe_mean,
            'sharpe_std': sharpe_std,
            'clamped_sharpe': clamped_sharpe,
            'pos_mean': pos_mean,
            'pos_std': pos_std,
            'vol_ratio': vol_ratio,
            'profit_proxy': estimated_profit_score,
            'estimated_utility': estimated_utility
        })

    df_analysis = pd.DataFrame(results)

    print("="*80)
    print("Utility Analysis (Sharpe clamped to [0, 6])")
    print("="*80)
    print()
    print(df_analysis.to_string(index=False))
    print()

    # Find best by utility
    best_utility_idx = df_analysis['estimated_utility'].idxmax()
    best_utility = df_analysis.iloc[best_utility_idx]

    print("="*80)
    print("FINDINGS")
    print("="*80)
    print()
    print(f"Best k by Sharpe: {df_analysis.loc[df_analysis['sharpe_mean'].idxmax(), 'k']}")
    print(f"  → Sharpe: {df_analysis['sharpe_mean'].max():.3f}")
    print()
    print(f"Best k by Utility (Sharpe × Profit): {best_utility['k']}")
    print(f"  → Clamped Sharpe: {best_utility['clamped_sharpe']:.3f}")
    print(f"  → Profit Proxy: {best_utility['profit_proxy']:.3f}")
    print(f"  → Estimated Utility: {best_utility['estimated_utility']:.3f}")
    print()

    # Key insights
    print("="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print()
    print("1. Sharpe Clamping Effect:")
    print(f"   - Current best Sharpe: 0.665 (k=600)")
    print(f"   - Target Sharpe: 6.0 (9x improvement needed)")
    print(f"   - Implication: Need to maximize Sharpe first, THEN profit")
    print()

    print("2. Profit Analysis:")
    print(f"   - Position std increases with k: {df_analysis['pos_std'].min():.3f} → {df_analysis['pos_std'].max():.3f}")
    print(f"   - Higher position variation → higher profit potential")
    print(f"   - BUT: Must maintain Sharpe > 0.5 for utility > 0")
    print()

    print("3. Current Approach Limitation:")
    print("   - k=200→600: Sharpe +6% (0.627→0.665)")
    print("   - At this rate, need k→∞ to reach Sharpe=6")
    print("   - **Fundamental issue**: Prediction accuracy too low")
    print()

    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()
    print("Option 1: Aggressive k increase (HIGH RISK)")
    print("  - Test k=1000~5000")
    print("  - Goal: Sharpe 3~6 + High profit")
    print("  - Risk: Sharpe collapse, utility=0")
    print()

    print("Option 2: Improve prediction accuracy (FUNDAMENTAL)")
    print("  - Current MSE: 0.00015, correlation ~0.03-0.06")
    print("  - Better features, better model → Higher Sharpe ceiling")
    print("  - Examples:")
    print("    - Longer lag features (20, 40, 60)")
    print("    - Cross-sectional features")
    print("    - Ensemble of models")
    print("    - Different target engineering")
    print()

    print("Option 3: Volatility scaling (MODERATE IMPROVEMENT)")
    print("  - position = 1 + (excess * k) / rolling_vol")
    print("  - Reduce position in high-vol periods")
    print("  - Expected: Sharpe +10~20%, closer to 0.8")
    print()

    print("Option 4: Rethink the problem (RADICAL)")
    print("  - Are we predicting the right target?")
    print("  - Should we predict sign (classification) instead of magnitude?")
    print("  - Can we identify high-confidence predictions only?")
    print()

    # Save analysis
    df_analysis.to_csv(RESULTS / "utility_analysis.csv", index=False)
    print(f"[SAVED] {RESULTS / 'utility_analysis.csv'}")
    print()

    # Decision tree
    print("="*80)
    print("DECISION TREE")
    print("="*80)
    print()
    print("Current: Sharpe 0.665 << 6.0 (target)")
    print()
    print("Path A: Try aggressive k (k=2000~5000)")
    print("  ├─ If Sharpe > 1.0: Good! Continue")
    print("  ├─ If Sharpe 0.5~1.0: Marginal, try Option 3")
    print("  └─ If Sharpe < 0.5: STOP, go to Path B")
    print()
    print("Path B: Fundamental improvement (prediction accuracy)")
    print("  ├─ Longer features (Lag 20, 40, 60)")
    print("  ├─ Volatility scaling")
    print("  ├─ Ensemble (XGBoost + LightGBM + Lasso)")
    print("  └─ Target engineering (sign prediction?)")
    print()
    print("Path C: Re-read competition details")
    print("  └─ Check if we're misunderstanding something fundamental")
    print()

    print("="*80)
    print("RECOMMENDED NEXT STEP")
    print("="*80)
    print()
    print("1. Test k=1000~3000 (1 hour)")
    print("   → See if Sharpe can reach 1.5~3.0")
    print()
    print("2. If failed (Sharpe < 1.0):")
    print("   → PIVOT to fundamental improvement")
    print("   → Focus on prediction accuracy, not k tuning")
    print()
    print("3. Write PIVOT.md documenting why k-tuning failed")
    print()


if __name__ == "__main__":
    analyze_phase1_results()
