# EXP-038 v3: Hybrid Feature Set (MSE + Quantile)

## 1. 개요
- **목표**: v1(Feature 교체)과 v2(데이터 분할)의 실패를 거울삼아, 기존 모델의 장점을 유지하면서 Quantile 정보를 추가.
- **가설**: MSE 기준 Feature(Trend)와 Quantile 기준 Feature(Tail Risk)를 **결합(Union)**하여 사용하면 상호 보완적인 효과를 낼 것이다.
- **기간**: 2025-11-25

## 2. 방법론
1.  **Feature Selection**:
    - **Set A (Trend)**: EXP-016 Top 30 (MSE 기준).
    - **Set B (Tail)**: EXP-038 v1 Top 30 (Quantile 기준).
    - **Hybrid Set**: Set A ∪ Set B (합집합). 총 51개 Feature.
2.  **Modeling**:
    - Global Quantile XGBoost (EXP-020 전략 유지).
    - Feature만 Hybrid Set으로 변경.

## 3. 결과
- **CV Sharpe**: **0.7025**
- **Baseline (EXP-020)**: 0.6368
- **변화율**: **+10.32%** ✅

## 4. 성공 요인 분석
- **Augmentation vs Replacement**: v1처럼 기존 정보를 버리지 않고, 새로운 정보를 **추가**한 것이 주효함.
- **Complementary Signals**:
    - MSE Feature는 평상시의 추세를 잘 잡음.
    - Quantile Feature(`E19` 관련 등)는 특이 구간이나 꼬리 위험을 잘 잡음.
- **Robustness**: 데이터 분할 없이 전체 데이터를 사용하여 학습 안정성 확보.

## 5. 결론
- **채택(Accepted)**.
- 최종 제출 파일(`submission_exp038_v3.py`)에 적용.
