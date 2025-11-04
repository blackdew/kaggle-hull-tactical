# EXP-025 Results: Deep Learning with ResNet & U-Net

**Status**: ❌ **FAILED**

**Conclusion**: Deep learning approaches (ResNet, U-Net) are fundamentally unsuitable for this task.

---

## Results Summary

### Phase 1: 1D ResNet Baseline
- **CV Sharpe**: 0.547 ± 0.377
- **Range**: [0.034, 0.949]
- **Expected Public**: 4.70
- **vs EXP-021 (XGBoost)**: -14.1% ❌
- **vs EXP-024 (XGBoost)**: -2.1% ❌

**Problems**:
- High variance across folds (std = 0.377)
- Fold 2 had training instability (val loss 637.69, Sharpe 0.034)
- Overall worse than XGBoost baseline

### Phase 2: 1D U-Net with Skip Connections
- **Status**: Training collapsed
- **Epoch 10 Val Loss**: 654,285,759,146
- **Problem**: Complete training failure - gradient explosion despite gradient clipping (max_norm=1.0)

**Architecture Tested**:
- Encoder-Decoder with skip connections
- Channels: [32, 64, 128, 256]
- Residual blocks at each level
- Result: Unstable, exploding gradients

### Phase 3: Simple MLP with Strong Regularization
- **CV Sharpe**: 0.358 ± 0.461
- **Range**: [-0.325, 0.815]
- **Expected Public**: 3.08
- **vs EXP-021 (XGBoost)**: -43.8% ❌

**Problems**:
- Extremely high variance (std = 0.461)
- Fold 1 negative Sharpe (-0.325)
- Fold 2 very low (0.215)
- Initial training instability (val loss 77B, 103M at epoch 20)
- Even with strong regularization (dropout 0.5, weight_decay 1e-3, lr 0.0001), still unstable

**Architecture Tested**:
- Simple: 305 → 128 → 64 → 1
- Dropout: 0.5 (very strong)
- Learning rate: 0.0001 (very conservative)
- Weight decay: 1e-3
- Result: High variance, worse than XGBoost

---

## Why Deep Learning Failed

### 1. Limited Data
- **8,990 samples** for 305 features
- Deep networks need 10,000+ samples per feature
- Severe overfitting risk

### 2. Tabular Data Structure
- No spatial locality (unlike images)
- No temporal patterns (InferenceServer constraint)
- Features are heterogeneous (M/V/P/S/I/E categories)

### 3. Training Instability
- Phase 1: Fold 2 exploded (val loss 637.69)
- Phase 2: Fold 1 exploded (val loss 654B)
- Gradient clipping insufficient
- Learning rate scheduling insufficient

### 4. Model Complexity
- ResNet (Phase 1): 512→256→128→64 layers
- U-Net (Phase 2): Encoder-decoder with skip connections
- Both too complex for tabular data

---

## Historical Context

### All Deep Learning Experiments Failed
| Experiment | Model | CV Sharpe | Problem |
|------------|-------|-----------|---------|
| EXP-014 | LSTM | 0.471 | Temporal dimension useless |
| EXP-015 | Transformer | 0.257-0.299 | Too complex for limited data |
| EXP-025 Phase 1 | ResNet | 0.547 ± 0.377 | High variance, unstable |
| EXP-025 Phase 2 | U-Net | N/A | Training collapsed |
| EXP-025 Phase 3 | Simple MLP | 0.358 ± 0.461 | Extreme variance, negative Sharpe in Fold 1 |

### XGBoost Consistently Outperforms
| Experiment | Model | CV Sharpe | Public Score |
|------------|-------|-----------|--------------|
| EXP-016 | XGBoost | 0.559 | 4.44 |
| EXP-021 | XGBoost | 0.637 | 5.87 |
| EXP-024 | XGBoost (over-reg) | 0.559 | N/A |

---

## Recommendations

### ✅ **DO**: Use XGBoost with 305 Features

1. **Fix EXP-024 Over-Regularization**:
   - Current: `max_depth=4`, `colsample_bytree=0.3`, `lr=0.01`
   - Recommended: `max_depth=6`, `colsample_bytree=0.6`, `lr=0.025`

2. **Use Top 50-100 Features**:
   - EXP-024 Phase 3 identified top features by MI
   - Top feature: `V7_sqrt` (MI 0.109)
   - Reduce overfitting by feature selection

3. **Ensemble Multiple Models**:
   - XGBoost with different random seeds
   - Different feature subsets
   - Weighted average predictions

### ❌ **DON'T**: Try More Deep Learning

- ResNet, U-Net, LSTM, Transformer all failed
- Tabular data + limited samples = DL failure
- Focus on feature engineering + XGBoost

---

## Next Steps

**EXP-026 (Recommended)**:
1. Use top 50 features from EXP-024 Phase 3
2. Moderate XGBoost regularization (not over-regularized)
3. 5-fold CV with multiple random seeds
4. Target: CV Sharpe > 0.7, Public Score > 6

**Alternative**:
- Ensemble: XGBoost + LightGBM + CatBoost
- Stacking: Train meta-learner on CV predictions
- Feature selection: Recursive feature elimination

---

**Date**: 2025-11-04
**Status**: Experiment concluded - Deep learning not viable
