# EXP-038 v1: Quantile Feature Selection

## 1. 개요
- **목표**: Quantile Regression 모델에 최적화된 Feature를 선별하여 성능 향상 도모.
- **가설**: 기존 MSE(Mean Squared Error) 기준으로 선별된 Feature보다, Quantile Loss(q10, q90) 기준으로 선별된 Feature가 꼬리 위험(Tail Risk) 예측에 더 효과적일 것이다.
- **기간**: 2025-11-25

## 2. 방법론
1.  **Feature Generation**: EXP-016과 동일한 Interaction Feature 생성.
2.  **Quantile Selection**:
    - `XGBRegressor(objective='reg:quantileerror')`를 사용하여 q10, q90 모델 학습.
    - 각 모델의 Feature Importance를 추출하여 평균냄.
    - 상위 30개 Feature 선정.
3.  **Evaluation**: 선정된 Feature로 EXP-020 전략(Global Quantile XGBoost) 평가.

## 3. 결과
- **CV Sharpe**: 0.5770
- **Baseline (EXP-020)**: 0.6368
- **변화율**: -9.4%

## 4. 실패 원인 분석
- **Signal vs Noise**: 금융 데이터의 특성상 꼬리(Tail) 부분은 노이즈가 심함. 이를 기준으로 Feature를 선정하니 노이즈에 과적합된 변수들이 선택됨.
- **Trend 부재**: MSE 기준 Feature들이 잡아주던 전반적인 시장 추세(Trend) 정보를 놓침.

## 5. 결론
- Quantile Feature만 단독으로 사용하는 것은 위험함.
- **폐기(Discarded)**.
