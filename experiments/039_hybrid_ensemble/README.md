# EXP-039: Hybrid Ensemble

## 1. 개요
- **목표**: EXP-038 v3의 **Hybrid Feature Set (51개)**과 EXP-022의 **Ensemble Strategy (XGB+LGB+Cat)**를 결합하여 SOTA(5.86) 경신 도전.
- **가설**:
    - Hybrid Features는 Trend와 Tail Risk 정보를 모두 제공하여 예측력을 높임.
    - Ensemble은 모델 간의 상관관계를 낮춰 일반화 성능(Public Score)을 높임.
- **기간**: 2025-11-25

## 2. 방법론
1.  **Features**: EXP-038 v3에서 사용한 51개 Hybrid Feature Set.
2.  **Models**:
    - XGBoost (Quantile Regression)
    - LightGBM (Quantile Regression)
    - CatBoost (Quantile Regression)
    - 각 모델별 q10, q50, q90 학습 (총 9개 모델).
3.  **Ensemble**:
    - 각 Quantile별로 3개 모델의 예측값을 단순 평균(Simple Average).
    - `pred_q50 = (xgb_q50 + lgb_q50 + cat_q50) / 3`
4.  **Position Sizing**:
    - `confidence = 1 / (pred_q90 - pred_q10)`
    - `position = 1 + pred_q50 * K * confidence`

## 3. 결과
- **CV Sharpe**: 0.6588
    - EXP-038 v3 (Single Model): 0.7025
    - EXP-022 (Baseline Ensemble): 0.6368
- **Public Score**: **3.736** ❌
    - EXP-038 v3 (4.798) 대비 하락.
    - EXP-022 (5.860) 대비 대폭 하락.

## 4. 분석
- **실패 원인**:
    - **Negative Synergy**: Hybrid Feature Set(51개)이 LightGBM/CatBoost와 잘 맞지 않았거나, 튜닝 부족으로 과적합 발생.
    - **Complexity**: Feature 수와 모델 수를 동시에 늘린 것이 독이 됨.
- **교훈**:
    - "좋은 것 + 좋은 것 = 더 좋은 것"이 항상 성립하지 않음.
    - Feature가 늘어날수록 모델은 단순하게 유지하는 것이 유리할 수 있음 (EXP-038 v3의 성공 요인).

## 5. 결론
- **실패 (Failed)**.
- EXP-038 v3 (Hybrid Features + Single XGBoost)가 현재까지의 최선(4.798)이나, 여전히 Baseline(5.86)에는 미치지 못함.
