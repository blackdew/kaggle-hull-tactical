# EXP-020: 10점 달성을 위한 근본적 재설계

## 전체 실험 히스토리 요약

### 목표
**Kaggle Public Score 10점 이상 달성** (다른 참가자 최고 17점)

### 실험 경과

| Experiment | 날짜 | 전략 | CV Sharpe | Public Score | 상태 |
|------------|------|------|-----------|--------------|------|
| **EXP-005** | 2025-10 초반 | XGBoost baseline | 0.627 | 0.724 | ✅ Baseline |
| **EXP-006** | 2025-10 초반 | K parameter 튜닝 | 0.699 | - | ⚠️ 한계 확인 |
| **EXP-007** | 2025-10 중반 | 754 features (lag/rolling) | 0.749 | - | ❌ InferenceServer 불가 |
| **EXP-008-015** | 2025-10 중반 | 딥러닝 시도들 | <0.75 | - | ❌ 모두 실패 |
| **EXP-016** | 2025-10-21 | **Interaction Features** | **0.559** | **4.440** | ✅ **최고 성과!** |
| **EXP-018** | 2025-10-25 | Dynamic K + Volatility | 0.582 | - | ⚠️ 제출 안함 |
| **EXP-019** | 2025-10-25 | Aggressive Ensemble | 3.541 | 3.599 | ❌ **Overfitting** |
| **EXP-020** | 진행 예정 | Volatility-aware Position | TBD | TBD | 📋 **계획 중** |

### 주요 발견사항

**성공 사례 (EXP-016)**:
- ✅ Simple 2-way interactions (30 features)
- ✅ K=250 fixed
- ✅ CV 0.559 → Public 4.440 (ratio 7.9x)
- ✅ InferenceServer 호환

**실패 사례 (EXP-019)**:
- ❌ Complex 3-way, 4-way interactions (284→30 features)
- ❌ Kelly Criterion + 5-model ensemble
- ❌ CV 3.541 → Public 3.599 (ratio 1.0x = Overfitting!)
- ❌ 복잡도 증가 = 성능 저하

**핵심 교훈**:
1. **Simple > Complex**: 2-way interactions까지만 유효
2. **High CV ≠ Good Public**: CV 3.54인데 Public 3.60은 overfitting 신호
3. **CV-to-Public Ratio 중요**: 7-8x가 정상, 1x는 위험
4. **Feature Quality > Quantity**: 30 simple features > 284 complex features

### 현재 상황

**최고 성과**: EXP-016 (Public Score 4.440)
**목표**: 10점 이상 (2.25배 향상 필요)
**Gap with Top**: 17점 / 4.44점 = 3.83배 차이

**다음 전략**: Position formula 근본 변경 (Feature 복잡도 증가는 실패 확인)

---

## 실패 원인 분석 (EXP-019)

### 결과 비교
| Experiment | CV Sharpe | Public Score | CV-to-Public Ratio |
|------------|-----------|--------------|-------------------|
| EXP-016 | 0.559 | **4.440** | **7.9x** ✅ |
| EXP-019 | 3.541 | 3.599 | **1.0x** ❌ |

### 근본 문제

#### 1. Massive Overfitting 🔴
**증거**:
- CV Sharpe 3.54 (EXP-016의 6.3배)
- Public Score 3.60 (EXP-016보다 낮음)
- **CV-to-Public ratio 1.0x (정상은 7-8x)**

**원인**:
- 3-way, 4-way interactions이 training noise 포착
- 284 features → Top 30 선택이 validation set에 과적합
- Kelly Criterion이 training 패턴에만 맞춤

**교훈**: High CV ≠ Good Public Score

#### 2. 잘못된 복잡도 증가 🔴
**시도한 것**:
- 284 features (2-way, 3-way, 4-way)
- 10-model ensemble → 5-model
- Kelly Criterion
- Regime-based weighting
- Quantile-based K

**결과**: 모두 역효과

**교훈**: Complexity ≠ Performance

#### 3. 잘못된 가정 🔴
**가정**: "더 많은 features + 더 복잡한 전략 = 더 좋은 성능"

**현실**:
- EXP-016 (30 simple features, K=250) = 4.440
- EXP-019 (284→30 complex features, 5-model ensemble, Kelly) = 3.599

**교훈**: Simple is Better

---

## 왜 10점이 어려운가?

### 현재 한계

**EXP-016 분석**:
- CV Sharpe: 0.559
- Public Score: 4.440
- 다른 참가자 최고: 17점

**Gap**:
- 필요: 4.440 → 10.0 (2.25배)
- 다른 참가자와 gap: 17 / 4.44 = **3.83배**

### 근본 원인

#### 1. Feature Engineering의 한계
**시도한 것**:
- 2-way interactions ✅ (EXP-016 성공)
- 3-way interactions ❌ (EXP-019 실패)
- 4-way interactions ❌ (EXP-019 실패)
- Meta-features ❌ (EXP-019 실패)

**결론**: Interaction features는 2-way까지만 유효

#### 2. Position Formula의 한계
**현재 formula**:
```python
position = clip(1.0 + excess_return * K, 0.0, 2.0)
```

**문제점**:
- Linear transformation만 사용
- Volatility 고려 부족 (EXP-018에서 약간 개선했지만 부족)
- Risk management 부족

**가능성**: Position formula 자체를 바꿔야 함

#### 3. 단일 전략의 한계
**현재**: Regression (excess return 예측) → Position

**다른 가능성**:
- Classification (direction 예측)
- Volatility prediction
- Multiple timeframe strategies
- Regime switching

---

## 17점 달성자의 가능한 접근 (추론)

### 가설 1: 완전히 다른 Position Formula
```python
# 현재 (우리)
position = clip(1.0 + excess_return * K, 0.0, 2.0)

# 가능한 대안
position = f(excess_return, volatility, regime, confidence, ...)
```

**근거**:
- Linear formula로는 한계 명확
- Kelly Criterion 시도했지만 실패
- 더 정교한 risk-adjusted sizing 필요

### 가설 2: Volatility-First Approach
```python
# Volatility를 먼저 예측
vol_pred = predict_volatility(features)

# Volatility로 position 조정
position = base_position * (target_vol / vol_pred)
```

**근거**:
- Sharpe = return / volatility
- Volatility control이 핵심일 수 있음

### 가설 3: Multiple Strategy Ensemble
```python
# Strategy 1: Trend following
# Strategy 2: Mean reversion
# Strategy 3: Volatility arbitrage

position = weighted_combination(s1, s2, s3)
```

**근거**:
- Single strategy의 한계
- 다양한 market regime 대응

### 가설 4: Direct Sharpe Optimization
```python
# Target을 excess_return이 아닌 Sharpe ratio로
# Differentiable Sharpe loss 사용
```

**근거**:
- EXP-011에서 시도했지만 실패 (Sharpe 0.552)
- 하지만 더 정교한 구현으로 가능할 수 있음

---

## 새로운 전략: EXP-020 계획

### 핵심 방향

**원칙**:
1. ✅ EXP-016의 단순함 유지 (30 features, 2-way interactions)
2. ✅ Position formula 근본적 변경
3. ✅ Volatility-aware strategy
4. ✅ Risk management 강화

### Strategy A: Volatility-Scaled Position Sizing

**목표**: Volatility를 직접 예측하고 활용

**접근**:
```python
# Step 1: Predict excess return (EXP-016과 동일)
excess_pred = model_return.predict(features)

# Step 2: Predict volatility (NEW!)
vol_pred = model_volatility.predict(features)

# Step 3: Target volatility approach
target_vol = 0.15  # Annual target volatility
position = base_position * (target_vol / vol_pred)

# Step 4: Combine with return prediction
position = clip(position * (1 + excess_pred * K), 0.0, 2.0)
```

**예상 개선**: +30-50%

### Strategy B: Quantile-Based Dynamic Position

**목표**: 예측 분포를 활용한 position sizing

**접근**:
```python
# Train quantile regression (10th, 50th, 90th percentile)
q10_pred = model_q10.predict(features)
q50_pred = model_q50.predict(features)
q90_pred = model_q90.predict(features)

# Confidence interval
ci_width = q90_pred - q10_pred
confidence = 1.0 / (ci_width + eps)

# Position based on confidence
position = clip(1.0 + q50_pred * K * confidence, 0.0, 2.0)
```

**예상 개선**: +20-40%

### Strategy C: Multi-Objective Optimization

**목표**: Return과 Risk를 동시 최적화

**접근**:
```python
# Objective 1: Maximize return
return_score = excess_pred * K

# Objective 2: Minimize risk (predicted volatility)
risk_score = -vol_pred

# Objective 3: Maximize Sharpe (return/risk)
sharpe_score = excess_pred / (vol_pred + eps)

# Combined score
combined = w1*return_score + w2*risk_score + w3*sharpe_score

# Position
position = clip(1.0 + combined, 0.0, 2.0)
```

**예상 개선**: +25-45%

### Strategy D: Regime-Aware Multiple Models

**목표**: Market regime별로 다른 전략 사용

**접근**:
```python
# Detect regime
regime = classify_regime(features)  # Bull/Bear/Sideways

# Different models for different regimes
if regime == 'Bull':
    position = model_bull.predict(features)
elif regime == 'Bear':
    position = model_bear.predict(features)
else:
    position = model_sideways.predict(features)
```

**예상 개선**: +30-50%

---

## EXP-020 실행 계획

### Phase 1: Volatility Prediction Model
**목표**: EXP-016 features로 volatility 예측

**방법**:
1. Target: realized_volatility = std(returns, window=20)
2. Model: XGBoost (EXP-016과 동일 architecture)
3. Features: EXP-016 Top 30 features
4. Evaluation: R², MAE

**예상 시간**: 30분

### Phase 2: Volatility-Scaled Strategy
**목표**: Target volatility approach 구현

**방법**:
1. Predict excess return (EXP-016 모델 재사용)
2. Predict volatility (Phase 1 모델)
3. Position = f(excess_pred, vol_pred)
4. Grid search for optimal target_vol

**예상 시간**: 1시간

### Phase 3: Quantile Regression
**목표**: Uncertainty quantification

**방법**:
1. Train 3 models (q10, q50, q90)
2. Confidence-based position sizing
3. Compare with Phase 2

**예상 시간**: 1시간

### Phase 4: Multi-Objective Approach
**목표**: Return-Risk tradeoff optimization

**방법**:
1. Combine return, risk, sharpe predictions
2. Grid search for optimal weights
3. Compare with Phase 2, 3

**예상 시간**: 1시간

### Phase 5: Best Strategy Selection
**목표**: CV로 최고 전략 선택

**방법**:
1. 5-fold CV로 모든 전략 평가
2. Best strategy 선택
3. InferenceServer 구현

**예상 시간**: 30분

---

## 예상 결과

### Conservative Scenario
- Strategy A (Volatility-scaled): +30%
- Public Score: 4.44 × 1.3 = **5.77**

### Expected Scenario
- Strategy B or C: +40%
- Public Score: 4.44 × 1.4 = **6.22**

### Optimistic Scenario
- Strategy D or combination: +50%
- Public Score: 4.44 × 1.5 = **6.66**

### Best Case
- Multiple strategies combined: +80-100%
- Public Score: 4.44 × 1.8-2.0 = **8.0-8.9**

**10점 달성 확률**: 30-40% (honest estimate)

---

## 핵심 교훈 (EXP-016 vs EXP-019)

### EXP-016 성공 요인
1. ✅ Simple 2-way interactions
2. ✅ 30 features (적당한 크기)
3. ✅ Linear position formula (단순함)
4. ✅ Fixed K=250 (복잡도 최소)
5. ✅ CV 0.559 → Public 4.44 (7.9x, 일반화 잘됨)

### EXP-019 실패 요인
1. ❌ Complex 3-way, 4-way interactions
2. ❌ 284 features (과도함)
3. ❌ Kelly Criterion (overfitting)
4. ❌ 5-model ensemble (복잡도 증가)
5. ❌ CV 3.54 → Public 3.60 (1.0x, overfitting)

### 새로운 원칙 (EXP-020)
1. ✅ Keep EXP-016's simplicity
2. ✅ Change position formula (근본적 개선)
3. ✅ Add volatility awareness (risk management)
4. ✅ Avoid overfitting (CV/Public ratio 모니터링)
5. ✅ Focus on generalization, not CV score

---

## 다음 단계

**추천**: EXP-020 Phase 1부터 시작

**우선순위**:
1. Phase 2 (Volatility-Scaled) - 가장 유망
2. Phase 3 (Quantile Regression) - Uncertainty 활용
3. Phase 4 (Multi-Objective) - 이론적으로 강력
4. Phase 1 (Volatility Prediction) - Foundation

**목표**: Public Score 6-8점 달성 후 10점 재도전

---

**Date**: 2025-10-25
**Status**: Planning
**Next**: EXP-020 Phase 1 implementation
