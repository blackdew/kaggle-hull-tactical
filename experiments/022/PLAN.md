# EXP-022: Ensemble Models Grid Search

**목표**: Multiple models를 ensemble하여 성능 개선
**전략**: XGBoost + LightGBM + CatBoost 조합 테스트

---

## 문제 인식

**EXP-021 결과**: Quantile alpha/scaling 튜닝으로는 더 이상 개선 불가
- Best CV Sharpe: 0.6368 (EXP-020과 동일)
- 파라미터 튜닝의 한계 도달

**돌파 전략**: 다른 모델을 ensemble하여 prediction diversity 확보

---

## Ensemble 설계

### Model Pool (3개)

1. **XGBoost** (Current baseline)
   - Quantile regression: alphas=[0.1, 0.5, 0.9]
   - Best params from EXP-020

2. **LightGBM**
   - Quantile objective
   - Similar hyperparameters
   - Faster training, different tree structure

3. **CatBoost**
   - Quantile loss
   - Different regularization approach
   - Robust to overfitting

### Ensemble Strategies (6가지)

**Simple Averaging**:
1. Equal weight (1/3 each)
2. Weighted by CV performance

**Stacking**:
3. Linear stacking (Ridge regression on predictions)
4. Non-linear stacking (XGBoost meta-model)

**Selective Ensemble**:
5. Best 2 models only
6. Confidence-weighted averaging

### Grid Search Space

**Model Combinations**:
- Single: XGB, LGB, CAT (3개)
- Pairs: XGB+LGB, XGB+CAT, LGB+CAT (3개)
- Triple: XGB+LGB+CAT (1개)

**Ensemble Methods**: 6가지

**Total**: 7 combinations × 6 methods = **42 configurations**

---

## 구현 계획

### Phase 1: 모델 학습 (30분)
```python
# Train 3 base models (각 5-fold CV)
models = {
    'xgb': train_xgboost_quantile(),
    'lgb': train_lightgbm_quantile(),
    'cat': train_catboost_quantile(),
}
```

### Phase 2: Ensemble Grid Search (1시간)
```python
# Test all ensemble combinations
for combination in model_combinations:
    for ensemble_method in ensemble_methods:
        # Combine predictions
        # Calculate Sharpe
        # Save results
```

### Phase 3: 최적 ensemble 선택 (10분)
- Best CV Sharpe configuration
- Analyze ensemble benefits
- Diversity vs. accuracy trade-off

### Phase 4: Submission 생성 (30분)
- Best ensemble을 InferenceServer로 구현
- Kaggle 제출

---

## 예상 결과

### Conservative
- +5-10% vs EXP-020 (5.87 → 6.2-6.5)

### Expected
- +10-20% vs EXP-020 (5.87 → 6.5-7.0)

### Optimistic
- +20-30% vs EXP-020 (5.87 → 7.0-7.6)

### 성공 기준
- CV Sharpe > 0.70 (EXP-020: 0.637)
- Public Score > 7.0

---

## 실행 순서

1. `experiments/022/ensemble_grid_search.py` 작성
2. `python -u experiments/022/ensemble_grid_search.py` 실행
3. `results/ensemble_results.csv` 분석
4. Best ensemble로 submission 작성
5. Kaggle 제출

---

**상태**: 계획 완료
**다음**: Phase 1 코드 작성
