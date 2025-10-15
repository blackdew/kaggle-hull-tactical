# EXP-015: Transformer with Residual Connections

## 결과 요약

| Model | Config | Avg Sharpe | vs Baseline | vs LSTM |
|-------|--------|------------|-------------|---------|
| **Baseline (EXP-007)** | XGBoost + 754 features | **0.749** | - | - |
| **LSTM (EXP-014)** | 2 layers, h=128, seq=40 | 0.471 | -37.1% | - |
| **Transformer Tiny** | d=64, 2 heads, 2 layers, seq=30 | 0.257 | -65.7% | -45.4% |
| **Transformer Medium** | d=96, 3 heads, 2 layers, seq=35 | 0.299 | -60.0% | -36.5% |

## 실패 원인 분석

### 1. **Fold별 성능 차이**

Transformer Tiny (3-fold):
- Fold 1 (earliest): -0.029 ❌
- Fold 2 (middle): 0.229
- Fold 3 (latest): 0.572 ✓

**문제**: 초기 fold에서 학습이 전혀 안됨. Transformer는 충분한 training data가 필요!

### 2. **모델 크기 증가가 역효과**

- Tiny (d=64): 0.257
- Medium (d=96): 0.299

더 큰 모델이 약간 나았지만, 여전히 LSTM(0.471)보다 훨씬 나쁨.

### 3. **단일 Fold 테스트 vs 3-Fold CV**

Transformer Tiny - Fold 3만 테스트: **0.708** ✅
Transformer Tiny - 3-Fold 평균: **0.257** ❌

**문제**: Fold 3에서는 잘 작동했지만, 초기 fold들이 평균을 크게 낮춤.

## 왜 Transformer가 LSTM보다 나쁜가?

### 1. **데이터 부족**
- Fold 1: 2,220 samples만으로 75K+ parameters 학습
- Transformer는 LSTM보다 훨씬 많은 데이터 필요
- 초기 fold에서는 overfitting

### 2. **시계열 특성**
- Transformer는 position encoding으로 순서 인식
- LSTM은 recurrent 구조로 자연스럽게 시계열 처리
- 금융 시계열에서는 LSTM의 inductive bias가 유리

### 3. **Attention의 한계**
- 30-40일의 짧은 sequence
- Attention이 장점을 발휘하기에는 너무 짧음
- LSTM의 hidden state가 더 효과적

## 시도한 최적화

1. ✅ **Pre-LN + Residual**: 구현 완료
2. ✅ **GELU activation**: ReLU 대신 사용
3. ✅ **AdamW + weight decay**: 정규화
4. ✅ **Gradient clipping**: 안정적 학습
5. ✅ **Cosine annealing LR**: 학습률 스케줄링

모든 최적화를 적용했지만 여전히 성능 부족!

## 결론

**Transformer + Residual 접근은 실패.**

- LSTM (0.471) < Transformer (0.257~0.299) < Baseline (0.749)
- 94개 feature를 multivariate time series로 보는 접근 자체에 문제 가능성
- 혹은 Transformer가 아닌 다른 아키텍처 필요

## 다음 방향

### Option 1: Temporal Convolutional Network (TCN)
- 1D Convolution으로 시계열 패턴 추출
- Dilated convolution으로 receptive field 확장
- Transformer보다 parameter 효율적

### Option 2: Hybrid Architecture
- CNN (feature extraction) + LSTM (temporal) + Attention
- 각 feature group별로 다른 처리
- Feature interaction 명시적 모델링

### Option 3: 근본적 재고
- **94개 feature를 모두 시계열로 볼 필요가 있나?**
- 일부는 시계열 (P, M, V groups)
- 일부는 static (D, E, I, S groups)?
- Feature group별 다른 전략 필요

### Option 4: Ensemble 강화
- XGBoost (0.749) + LSTM (0.471) 앙상블
- XGBoost가 여전히 최고 성능
- Deep learning은 보조 역할로 제한

## 교훈

1. **More parameters ≠ Better performance**
   - 200K parameters < 75K parameters

2. **Architecture matters**
   - Transformer가 항상 최고는 아님
   - 문제 특성에 맞는 아키텍처 선택 중요

3. **Data efficiency**
   - 초기 fold의 적은 데이터로는 복잡한 모델 학습 불가
   - XGBoost가 여전히 data-efficient

4. **Baseline is strong**
   - EXP-007의 0.749는 여전히 최고
   - 754개 engineered features의 힘
   - Deep learning이 이기려면 더 창의적인 접근 필요
