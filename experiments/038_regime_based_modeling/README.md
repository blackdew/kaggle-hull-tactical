# EXP-038 v2: Regime-Based Modeling

## 1. 개요
- **목표**: 시장의 변동성 국면(Regime)에 따라 별도의 모델을 적용하여 예측력 강화.
- **가설**: 고변동성 장과 저변동성 장은 시장 역학이 다르므로, 하나의 Global Model보다 각각에 특화된 Specialist Model이 더 유리할 것이다.
- **기간**: 2025-11-25

## 2. 방법론
1.  **Regime Split**:
    - 변동성 지표 `V13`을 기준으로 데이터 분할.
    - Threshold: `V13`의 Median (0.0).
    - Low Volatility (`V13` <= 0) / High Volatility (`V13` > 0).
2.  **Modeling**:
    - 각 Regime 별로 별도의 Quantile XGBoost 모델 학습.
    - Feature: EXP-016 Top 30 (Baseline과 동일).
3.  **Inference**:
    - 예측 시점의 `V13` 값에 따라 해당 Regime 모델을 선택하여 예측.

## 3. 결과
- **CV Sharpe**: 0.5683
- **Baseline (EXP-020)**: 0.6368
- **변화율**: -10.8%

## 4. 실패 원인 분석
- **Data Fragmentation**: 데이터를 둘로 나누면서 각 모델이 학습할 샘플 수가 절반으로 감소. 특히 Time Series CV 특성상 초기 Fold에서 학습 데이터 부족 심화.
- **V13 Distribution**: `V13`의 중앙값이 0.0으로, 데이터의 절반 이상이 "변동성 0"인 특이 분포를 보임. 이를 기준으로 한 단순 이분법이 시장의 질적 차이를 제대로 반영하지 못함.

## 5. 결론
- 현재 데이터 규모에서는 데이터를 나누는 것이 오히려 일반화 성능을 해침.
- **폐기(Discarded)**.
