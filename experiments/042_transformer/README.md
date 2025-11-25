# EXP-042: Time Series Transformer

## 1. 개요
- **목표**: Tree 기반 모델의 한계를 넘어, **Deep Learning (Transformer)**을 도입하여 시계열 패턴(Sequence)을 학습.
- **가설**:
    - 시장의 국면(Regime) 변화나 장기적인 패턴은 단일 시점의 Feature로는 포착하기 어려움.
    - Transformer의 Self-Attention이 과거 30일의 흐름에서 중요한 신호를 찾아낼 것.
- **기간**: 2025-11-25

## 2. 방법론
1.  **Data Preparation**:
    - Sliding Window (Sequence Length = 30).
    - Top 20 Base Features 사용.
2.  **Model Architecture**:
    - Transformer Encoder (d_model=64, nhead=4, layers=2).
    - Loss: MSE (Mean Squared Error).
3.  **Inference**:
    - `HistoryBuffer`를 구현하여 실시간으로 들어오는 데이터를 큐(Queue)에 쌓아 Sequence 생성.

## 3. 결과
- **CV Sharpe**: 0.4190
    - EXP-038 v3 (Hybrid Single): 0.7025
    - EXP-041 (Genetic): 0.6672
    - **분석**: CV 점수가 기존 Tree 모델보다 현저히 낮음.
        - 데이터셋 크기(약 9000개)가 Deep Learning을 학습하기에 부족할 수 있음.
        - 또는 Hyperparameter 튜닝 부족.

- **Public Score**: **1.041** ❌❌
    - 역대 최악의 기록.
    - Baseline(5.86)은커녕 Random Guess 수준일 가능성 높음.

## 4. 결론
- **완벽한 실패 (Total Failure)**.
- **원인**:
    1.  **Small Data**: 9,000개의 샘플로는 Transformer의 수만 개 파라미터를 학습시킬 수 없음.
    2.  **Noise**: 금융 데이터의 낮은 S/N 비율(Signal-to-Noise Ratio)이 딥러닝 모델을 혼란스럽게 함.
- **교훈**:
    - "공격적인 시도"가 항상 좋은 것은 아님.
    - 이 데이터셋에서는 **Feature Engineering + Tree Model** 조합이 최선임을 재확인.
    - Deep Learning을 쓰려면 훨씬 더 간단한 구조(MLP)나 강력한 Regularization이 필요함.
