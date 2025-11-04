# EXP-023: Profit Maximization Strategy - 결론

**목표**: Public Score 20점 달성 (현재 5.872 대비 3.4배)

**전략**: CV Sharpe 증가(불가능) 대신 **Profit 극대화**

---

## 최종 결과

### CV Performance
- **CV Sharpe**: 0.597 ± 0.326
- **Profit proxy**: 1444.5 (Baseline 641.5 대비 **+125%**)
- **Position std**: 0.903 (매우 aggressive)
- **Extreme positions**: 94.4% (대부분 0 또는 2 근처)

### 예상 Public Score
- Conservative (7.9x): **4.71**
- Expected (9.2x): **5.49**
- Optimistic (10x): **5.97**

**현재 최고 (5.872)보다 낮음** ❌

---

## Phase별 결과

### Phase 1: Profit 분석 ✅
**목표**: Position bottleneck 파악

**발견**:
- K=250일 때 position std = 0.498
- Extreme positions = 22.8%
- Profit proxy = 641.5

**결론**: Position이 [0, 2] 범위의 일부만 사용 → 확대 필요

---

### Phase 2: Position Amplification ✅✅✅
**목표**: [0, 2] 범위 전체 활용

**최고 전략: Sigmoid-20**
- Sharpe: 0.275 (+28% vs baseline 0.215)
- **Profit: 1270.8 (+98% vs baseline 641.5)** ✅✅✅
- Position std: 0.832
- Extremes: 75.1%

**성공**: Profit이 거의 2배 증가!

---

### Phase 3: Regime Detection ❌
**목표**: 시장 상황별 차별화

**결과**:
- Regime-based: Profit +0.6% (거의 효과 없음)
- Volatility-adaptive: Profit -21%
- Sentiment-adaptive: Profit -0.9%

**실패**: Regime detection은 도움 안됨

---

### Phase 6: Final Integration 🟡
**전략**: EXP-020 Quantile + EXP-023 Sigmoid-20

**결과**:
- CV Sharpe: 0.597 (EXP-021: 0.637)
- Profit: 1444.5 (+125% vs baseline)
- 예상 Public Score: 5.49

**평가**: Profit 증가 성공, 하지만 Sharpe 부족

---

## 핵심 발견

### 성공 요인
1. ✅ **Sigmoid amplification 매우 효과적** (profit +98%)
2. ✅ **Quantile regression + Sigmoid 결합 가능**
3. ✅ **Position 극단화로 profit 증대 달성**

### 실패 요인
1. ❌ **CV Sharpe가 0.6 수준으로 낮음**
   - 목표 20점 달성에는 CV Sharpe 2.0+ 필요
   - 현재 접근으로는 Sharpe 0.8 이상 달성 어려움

2. ❌ **Regime detection 효과 없음**
   - 시장 상황별 차별화가 도움 안됨
   - Feature quality 자체가 한계

3. ❌ **Profit vs Sharpe trade-off**
   - Profit 증가 시 Sharpe 하락 위험
   - Utility = min(Sharpe, 6) × Profit이므로 Sharpe가 더 중요

---

## Utility 방정식 재분석

### 목표 달성 조건
```
Utility = 20.0 = min(Sharpe, 6) × Profit
```

**Case A**: Sharpe = 6 (cap) → Profit = 3.33 필요
**Case B**: Sharpe = 4 → Profit = 5.0 필요
**Case C**: Sharpe = 3 → Profit = 6.67 필요

### 현재 EXP-023 (추정)
```
CV Sharpe = 0.597 → Public Sharpe ≈ 0.8~1.0
Profit proxy = 1444.5 (baseline 641.5의 2.25배)
→ Estimated Utility ≈ 0.8~1.0 × (5.9*2.25) ≈ 11~13
```

**여전히 20점에 크게 부족**

---

## 근본적 한계

### 한계 1: Sharpe의 벽
- 모든 실험에서 CV Sharpe 0.6~0.75 범위
- EXP-006, 007, 016, 020, 021, 023 모두 동일한 한계
- **Feature quality 자체의 한계**

### 한계 2: CV-to-Public Ratio 의존
- 최고 ratio: 9.2x (EXP-021)
- CV 0.6 × 9.2 = 5.5 (현재 수준)
- **20점 달성에는 ratio 33x 필요** (비현실적)

### 한계 3: Profit 증가의 한계
- Profit을 2배 증가시켜도 전체 utility는 2배
- 5.9 × 2 = 11.8 (여전히 20점 부족)

---

## 교훈

### 1. Sigmoid Amplification은 유효
- Profit을 거의 2배 증가 가능
- Position 극단화 전략은 작동함
- 하지만 Sharpe가 낮으면 한계

### 2. Regime Detection은 비효과적
- 1-row features만으로는 regime 파악 어려움
- 단순한 접근이 더 나음 (EXP-016, 019 교훈)

### 3. Profit-focused는 partial success
- Profit 증가 성공 (1.25x~2x)
- 하지만 Sharpe 개선 없이는 utility 한계
- **Sharpe > Profit in importance**

---

## 20점 달성 가능성 평가

### 현재 상황
- 최고 Public Score: 5.872
- 목표: 20.0
- Gap: **3.4배**

### 필요 조건
**Option A**: CV Sharpe 2.0+ (현재 0.6의 3.3배)
- 모든 시도가 실패
- Feature quality 근본 개선 필요
- **실현 가능성: 5% 미만**

**Option B**: CV-to-Public Ratio 30x+ (현재 9.2x의 3.3배)
- 비현실적
- **실현 가능성: 1% 미만**

### 최종 평가
**20점 달성 가능성: < 5%**

---

## 다음 단계 제안

### Option 1: EXP-023 제출 (추천 ⭐⭐⭐)
**이유**:
- Profit 전략 검증 필요
- 예상 5.5점 (현재 5.87 대비 비슷)
- Sigmoid amplification 효과 확인

**Action**:
- submission_exp023.py 생성
- Kaggle 제출
- Public Score 확인

---

### Option 2: 새로운 접근 탐색 (추천 ⭐⭐⭐⭐⭐)
**가능한 방향**:

#### 2A: External Features
- 외부 데이터 추가 (macro indicators, sentiment data)
- Feature quality 근본 개선
- **예상**: Sharpe 0.8~1.2 (2배 향상)

#### 2B: Ensemble of Diverse Strategies
- Trend following + Mean reversion + Momentum
- 전혀 다른 signal 결합
- **예상**: Sharpe 0.9~1.3

#### 2C: Deep Learning (재시도)
- Transformer with better architecture
- Longer sequence (date aggregation)
- **예상**: Sharpe 0.7~1.1

#### 2D: 목표 재설정
- 20점이 비현실적일 가능성
- 10점 목표로 재설정
- Public 8~12점 달성 집중

---

### Option 3: Accept Current Best (추천 ⭐⭐⭐⭐)
**현실**:
- EXP-021: Public 5.872 (최고 기록)
- 1~22 모든 실험 시도 완료
- Sharpe 0.6~0.75 한계 명확

**제안**:
- EXP-021 유지
- 추가 실험 중단
- 리소스 다른 프로젝트에 투입

---

## 최종 결론

### EXP-023 성과
- ✅ **Profit maximization 전략 성공** (profit +125%)
- ✅ **Sigmoid amplification 효과 확인** (profit +98%)
- ❌ **20점 목표 달성 실패** (예상 5.5점)
- ❌ **Regime detection 비효과적**

### 핵심 교훈
1. **Sharpe > Profit** in importance for utility
2. **Feature quality가 근본적 bottleneck**
3. **Position formula 개선만으로는 한계**
4. **20점 목표는 비현실적일 가능성**

### 추천 행동
1. EXP-023 제출하여 Sigmoid 효과 검증
2. 20점 목표 재평가 (10점으로 하향 조정)
3. EXP-021 (5.872) 유지 또는 새로운 접근 탐색

---

**작성일**: 2025-11-04
**상태**: EXP-023 완료, 제출 대기
**다음**: submission_exp023.py 생성 또는 전략 재검토
