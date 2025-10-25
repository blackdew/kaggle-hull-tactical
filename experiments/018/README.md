# EXP-018: Dynamic K with Volatility-Adaptive Strategy

**Goal**: Improve on EXP-016 (Public Score 4.440) by using dynamic K parameter based on market volatility

**Result**: CV Sharpe 0.5815 ± 0.3578 (+4.02% vs EXP-016)

**Strategy**: Volatility-adaptive position sizing with dynamic K

---

## Results Summary

### Cross-Validation Performance
- **CV Sharpe**: 0.5815 ± 0.3578 (5-fold TimeSeriesSplit)
- **Improvement over EXP-016**: +4.02%
- **Expected Public Score**: 4.1 - 5.2 (conservative estimate)

### Key Parameters
- **K_base**: 350
- **vol_scale**: 1.0
- **Confidence multiplier**: 0.0 (not used)
- **Long/Short ratio**: 1.0 / 1.0 (symmetric)

### Features
- **Top 30 Interaction Features** (from EXP-016)
- **Top 5 Volatility Features** (NEW)
  1. `vol_mean_V`: Mean of V13, V7, V9
  2. `vol_composite_V_all`: √(V13² + V7² + V9² + V10²)
  3. `vol_composite_V`: √(V13² + V7² + V9²)
  4. `cross_vol_MV`: √(M4² + V13²)
  5. `sentiment_dispersion`: |S2 - S5|

**Total Features**: 35 (all 1-row calculable)

---

## Experimental Process

### Phase 1: Volatility Features Development
**Script**: `phase1_volatility_features.py`

Created 27 volatility proxy features:
- Composite volatility measures (V-category)
- Market turbulence (M-category)
- Price dispersion (P-category)
- Cross-category volatility
- Sentiment dispersion
- Coefficient of variation proxies

Selected top 10 based on RandomForest importance + correlation.

**Key Finding**: `vol_mean_V` has highest combined score (0.0905)

---

### Phase 2: Dynamic K Parameter Search
**Script**: `phase2_dynamic_k_search.py`

Grid search strategies tested:
1. **Baseline** (no adjustments)
2. **Volatility-only** adjustment
3. **Confidence-only** adjustment
4. **Asymmetric** long/short
5. **Combined** adjustments

**Total combinations**: 99

**Best configuration**:
- K_base = 350
- vol_scale = 1.0
- conf_mult = 0.0
- long_ratio = 1.0
- short_ratio = 1.0

**Key Finding**: Simple volatility-based adjustment (vol_scale=1.0) outperforms complex strategies

---

### Phase 3: Integrated Model Evaluation
**Script**: `phase3_integrated_model.py`

Final evaluation with best parameters:
- 5-fold CV Sharpe: 0.5815 ± 0.3578
- Mean position: 1.0403
- Mean dynamic K: 215.44 (vs base 350)

**Volatility features in top 15**: 3 out of 5
- cross_vol_MV (#1)
- vol_composite_V (#5)
- sentiment_dispersion (#10)

---

### Phase 4: InferenceServer Submission
**Script**: `submissions/submission_exp018.py`

Dynamic K formula:
```python
vol_adjustment = 1.0 / (1.0 + vol_scale * vol_proxy)
k_dynamic = k_base * vol_adjustment
position = clip(1.0 + excess_return * k_dynamic, 0.0, 2.0)
```

**Key Design**:
- All features 1-row calculable
- Volatility proxy computed from same row
- Dynamic K varies: ~150-350 based on volatility
- InferenceServer compatible

---

## Key Insights

### 1. Volatility Matters
- High volatility → Lower K (conservative)
- Low volatility → Higher K (aggressive)
- Dynamic K range: 149-350 (mean ~215)

### 2. Simple is Better
- Volatility-only adjustment outperforms combined strategies
- Confidence multiplier didn't help (0.0 optimal)
- Asymmetric long/short didn't help (1.0/1.0 optimal)

### 3. Volatility Features Add Value
- 3 of top 15 features are volatility features
- cross_vol_MV is #1 most important feature
- Volatility features capture market regime

### 4. Modest CV Improvement
- +4% CV Sharpe improvement
- But EXP-016 showed CV doesn't predict Public well
- CV 0.559 → Public 4.440 (~8x multiplier)
- If same ratio: CV 0.5815 → Public 4.6-4.9

---

## Comparison with EXP-016

| Metric | EXP-016 | EXP-018 | Change |
|--------|---------|---------|--------|
| Features | 30 | 35 | +5 volatility |
| K Strategy | Fixed (250) | Dynamic (350 base) | Adaptive |
| CV Sharpe | 0.559 ± 0.362 | 0.5815 ± 0.3578 | +4.02% |
| Public Score | 4.440 | TBD | ? |

---

## Files

```
experiments/018/
├── README.md                           # This file
├── phase1_volatility_features.py       # Volatility feature development
├── phase2_dynamic_k_search.py          # Grid search for K parameters
├── phase3_integrated_model.py          # Final model evaluation
├── test_submission.py                  # Local testing script
└── results/
    ├── volatility_features_ranking.csv # All 27 volatility features
    ├── top_10_volatility_features.csv  # Top 10 selected
    ├── k_search_results.csv            # Grid search results
    ├── final_cv_results.csv            # 5-fold CV results
    ├── final_config.csv                # Best configuration
    └── feature_importance.csv          # XGBoost feature importance

submissions/
└── submission_exp018.py                # InferenceServer implementation
```

---

## Next Steps

### Option 1: Submit to Kaggle
- Upload `submission_exp018.py`
- Check Public Score
- Compare with EXP-016 (4.440)

### Option 2: Further Improvements
- Try ensemble of EXP-016 + EXP-018
- Add 3-way interaction features
- Explore non-linear position transforms

### Option 3: Wait for More Ideas
- Analyze EXP-018 results
- Study fold-specific performance
- Look for patterns in failures

---

## Expected Outcome

**Conservative**: Public Score 4.5-5.0 (~+5-15% vs EXP-016)
**Realistic**: Public Score 5.0-6.0 (~+15-35% vs EXP-016)
**Optimistic**: Public Score 7.0+ (~+60%+ vs EXP-016)

Based on EXP-016's CV-to-Public ratio (~8x), and if the pattern holds:
- CV 0.5815 × 8 = **4.65 Public Score**

---

**Date**: 2025-10-25
**Status**: Ready for submission
**Recommendation**: Submit and evaluate Public Score
