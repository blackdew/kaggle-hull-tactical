# EXP-023: Profit Maximization Strategy

**목표**: Public Score 20점 달성 (현재 5.872 대비 3.4배)

**핵심 전략**: CV Sharpe 증가(불가능) 대신 **Profit 극대화**

---

## 근본적 문제 재정의

### Utility 공식
```
Utility = min(Sharpe, 6) × Σ(profits)
```

### 목표 달성 방정식
```
20.0 = min(Sharpe, 6) × Profit
```

**Case A**: Sharpe = 6 (cap) → Profit = 3.33 필요
**Case B**: Sharpe = 4 → Profit = 5.0 필요
**Case C**: Sharpe = 3 → Profit = 6.67 필요

### 현재 상황 (추정)
- CV Sharpe: 0.637
- Public Score: 5.872
- CV-to-Public Ratio: 9.2x
- **추정 Public Sharpe**: ~0.8~1.0
- **추정 Profit**: ~5.9~7.3

**결론**: Sharpe는 충분, **Profit을 1.5~2배 증가** 필요!

---

## 핵심 통찰

### 발견 1: CV Sharpe 정체
```
EXP-005: 0.627 → EXP-007: 0.749 → EXP-016: 0.559
→ EXP-020: 0.637 → EXP-021: 0.637
```
**결론**: 0.6~0.75 범위에서 정체, 더 높이기 매우 어려움

### 발견 2: CV-to-Public Ratio 증가 추세
```
EXP-016: 7.9x → EXP-019: 1.0x (overfit) → EXP-021: 9.2x ✅
```
**결론**: 단순한 모델이 ratio를 높임 (EXP-019 교훈)

### 발견 3: Position 분포 문제
```
현재: position 평균 ~1.0, 대부분 0.9~1.1 범위
문제: [0, 2] 범위 중 10%만 활용
해결: Position을 [0.2, 1.8] 범위로 확대 → Profit 2배+
```

---

## 실험 Phase

### Phase 1: Profit 분석 ⭐⭐⭐⭐⭐
**목표**: 현재 profit bottleneck 파악

**분석 내용:**
1. Historical position distribution
2. Profit per trade 분석
3. Position size vs Return 상관관계
4. High profit trades vs Low profit trades 특성

**Output**: `results/phase1_profit_analysis.csv`

**시간**: 30분

---

### Phase 2: Position Amplification ⭐⭐⭐⭐⭐
**목표**: [0, 2] 범위 전체 활용하여 profit 증대

**3가지 전략 테스트:**

#### Strategy A: Sigmoid Amplification
```python
signal = excess_pred * K
position = 2.0 / (1 + exp(-sigmoid_slope * signal))
```
- sigmoid_slope = [5, 10, 15, 20]

#### Strategy B: Quantile-based Binary
```python
threshold_high = 90th percentile of |excess_pred|
threshold_low = 10th percentile of |excess_pred|

if excess_pred > threshold_high:
    position = 2.0
elif excess_pred < -threshold_high:
    position = 0.0
else:
    position = 1.0 + excess_pred * K
```

#### Strategy C: Threshold Amplification
```python
if abs(excess_pred * K) > threshold:
    amplification = 2.0
else:
    amplification = 1.0

position = clip(1.0 + excess_pred * K * amplification, 0.0, 2.0)
```
- threshold = [0.2, 0.3, 0.5, 0.7]

**Output**: `results/phase2_position_strategies.csv`

**성공 조건:**
- Position std > 0.4 (현재 ~0.2)
- Position range [0.2, 1.8]
- Profit 증가 > 30%

**시간**: 1시간

---

### Phase 3: Market Regime Detection ⭐⭐⭐⭐
**목표**: 시장 상황별 차별화된 position sizing

**Regime Features (1-row calculable):**
1. Market trend proxy: M4 value (high/low)
2. Volatility regime: V13 value (high/low)
3. Sentiment: S2, S5 (positive/negative)
4. Economic: E19 (expansion/contraction)

**Strategy:**
```python
# Regime detection (1-row)
market_trend = 'bullish' if M4 > percentile_75 else \
               'bearish' if M4 < percentile_25 else 'neutral'
vol_regime = 'high' if V13 > percentile_75 else 'low'

# Position adjustment
if market_trend == 'bullish' and vol_regime == 'low':
    base = 1.3  # Aggressive long
elif market_trend == 'bearish' and vol_regime == 'high':
    base = 0.7  # Defensive
else:
    base = 1.0

position = clip(base + excess_pred * K, 0.0, 2.0)
```

**Output**: `results/phase3_regime_strategies.csv`

**시간**: 1시간

---

### Phase 4: Multi-K Ensemble ⭐⭐⭐
**목표**: 여러 K 값 사용하여 profit 다양성 확보

**Ensemble Strategy:**
```python
# 3 models with different K
model_conservative = XGBoost with K=50
model_balanced = XGBoost with K=250
model_aggressive = XGBoost with K=1000

# Weighted ensemble
if confidence > 0.8:
    weights = [0.1, 0.3, 0.6]  # Favor aggressive
elif confidence < 0.3:
    weights = [0.6, 0.3, 0.1]  # Favor conservative
else:
    weights = [0.2, 0.6, 0.2]  # Balanced
```

**Output**: `results/phase4_multi_k_ensemble.csv`

**시간**: 1.5시간

---

### Phase 5: Time-based Pattern ⭐⭐
**목표**: date_id 패턴 활용

**Cyclical Features (1-row calculable):**
```python
# Sin/Cos encoding for cyclicality
week_of_year = (date_id % 252) / 252  # Trading days
month_proxy = (date_id % 21) / 21     # ~21 trading days/month

week_sin = sin(2 * pi * week_of_year)
week_cos = cos(2 * pi * week_of_year)
month_sin = sin(2 * pi * month_proxy)
month_cos = cos(2 * pi * month_proxy)
```

**Strategy:**
```python
# Time-based K adjustment
seasonal_factor = predict_seasonal_multiplier(week_sin, week_cos, month_sin, month_cos)
K_adjusted = K * seasonal_factor  # [0.5, 2.0] range
```

**Output**: `results/phase5_time_based.csv`

**시간**: 1시간

---

### Phase 6: Integrated Best Strategy ⭐⭐⭐⭐⭐
**목표**: Phase 1~5에서 최고 조합 선택

**Integration:**
```python
# Phase 2 best: Sigmoid or Threshold
# Phase 3 best: Regime detection
# Phase 4: Multi-K ensemble if helpful
# Phase 5: Time adjustment if helpful

final_position = integrate_best_strategies()
```

**Output**:
- `results/phase6_integrated_cv.csv`
- `submissions/submission_exp023.py`

**시간**: 1시간

---

## 예상 결과

### Conservative (20% 확률)
- CV Sharpe: 0.5~0.6 (-15%)
- Position std: 0.35~0.45 (+75%)
- Profit: +50%
- **Public Score: 8~10**

### Expected (60% 확률)
- CV Sharpe: 0.55~0.65 (-5%)
- Position std: 0.45~0.55 (+125%)
- Profit: +80%
- **Public Score: 12~15**

### Optimistic (20% 확률)
- CV Sharpe: 0.6~0.7 (유지)
- Position std: 0.55~0.65 (+175%)
- Profit: +120%
- **Public Score: 18~22** ✅ 목표 달성!

---

## 성공 조건

### 필수 (Phase 2)
- ✅ Position distribution 변화: 0.9~1.1 → 0.3~1.7
- ✅ Position std 증가: 0.2 → 0.4+
- ✅ Profit proxy (position variation) 증가 > 50%

### 목표 (Phase 6)
- ✅ CV Sharpe > 0.5 (과도한 하락 방지)
- ✅ Public Score > 10 (현재 5.87 대비 1.7배)
- ✅ 목표 20점 달성 가능성 확인

---

## 리스크 관리

### Risk 1: Sharpe 과도 하락
**완화책**: Phase 2에서 Sharpe < 0.4면 중단, 덜 aggressive한 전략 채택

### Risk 2: Overfitting
**완화책**: EXP-019 교훈 - 단순한 접근 유지, 복잡한 interaction 피함

### Risk 3: Position 극단화
**완화책**: Position이 95% 이상 극단값(0 or 2)이면 조정

---

## 일정

**Day 1 (오늘)**
- [x] EXP-023 계획 수립
- [ ] Phase 1: Profit 분석 (30분)
- [ ] Phase 2: Position amplification (1시간)
- [ ] Phase 3: Regime detection (1시간)

**Day 2**
- [ ] Phase 4: Multi-K ensemble (1.5시간)
- [ ] Phase 5: Time-based (1시간)
- [ ] Phase 6: Integration (1시간)
- [ ] Kaggle 제출

**Total**: ~6시간

---

## 참고 실험

**성공 사례:**
- EXP-016: Simple 2-way interactions → Public 4.44 ✅
- EXP-021: Quantile regression → Public 5.87 ✅

**실패 사례:**
- EXP-019: 3-4way + Kelly + Ensemble → Public 3.60 (overfit)
- EXP-006: k 대폭 증가 → Sharpe 정체

**교훈:**
1. Simple > Complex
2. CV-to-Public ratio가 핵심
3. Position distribution이 profit에 직결
4. Sharpe는 0.6~0.7이면 충분

---

**Status**: Phase 1 준비 완료
**Next**: `phase1_profit_analysis.py` 실행
