# EXP-006 PIVOT: k 파라미터 접근의 한계와 방향 전환

## 요약

**결론**: k 파라미터 증가만으로는 목표 17.395 달성 불가능
**이유**: Sharpe 0.627 → 0.699 (+11.4%, k=200→3000), 목표 6.0의 1/6 미만
**Pivot**: 예측 정확도 근본 개선 + Kaggle 메트릭 이해 재점검 필요

---

## 실험 결과

### Phase 1a: k=200~800
| k | Sharpe | Vol Ratio | Pos Std | vs Baseline |
|---|--------|-----------|---------|-------------|
| 200 | 0.627 | 1.23 | 0.311 | baseline |
| 300 | 0.646 | 1.28 | 0.411 | +3.0% |
| 400 | 0.651 | 1.31 | 0.485 | +3.8% |
| 500 | 0.659 | 1.34 | 0.542 | +5.1% |
| 600 | 0.665 | 1.36 | 0.588 | +6.0% |
| 800 | 0.659 | 1.38 | 0.658 | +5.1% |

**결과**: k=600에서 peak, k=800에서 하락 시작

### Phase 1b: k=1000~3000 (Aggressive)
| k | Sharpe | Vol Ratio | Pos Std | vs Baseline |
|---|--------|-----------|---------|-------------|
| 1000 | 0.664 | 1.40 | 0.705 | +5.9% |
| 2000 | 0.690 | 1.44 | 0.814 | +10.0% |
| **3000** | **0.699** | 1.46 | 0.857 | **+11.4%** |

**결과**: k=3000에서도 Sharpe < 0.7, 목표 6.0의 1/8 미만

---

## 핵심 발견

### 1. k 증가의 한계
```
k × 15배 (200 → 3000)
Sharpe × 1.11배 (0.627 → 0.699)
```

**의미**: k를 무한대로 올려도 Sharpe는 수렴 (아마도 0.7~0.8 근처)

**원인**:
- 예측 정확도 자체가 낮음 (MSE 0.00015, corr ~0.03-0.06)
- k는 신호 증폭만 가능, 신호 자체는 생성 못함
- 약한 신호 × 큰 k = 노이즈 증폭

### 2. Kaggle 메트릭의 오해
**발견**: `utility = min(max(sharpe, 0), 6) × Σ profits`

**이전 이해**:
- Sharpe만 최적화하면 됨
- k 높이면 Sharpe 높아짐 → utility 높아짐

**올바른 이해**:
- **Sharpe는 [0, 6]으로 클램핑됨**
- Sharpe 6 이상은 의미 없음
- **Profit이 핵심 변수!**
- 목표 17.395 = sharpe(~6) × profit(~2.9)

### 3. 현재 Sharpe의 의미
- **현재**: 0.665~0.699
- **목표**: 6.0
- **격차**: 8.6~9.0배

**문제**: k를 15배 올려도 1.11배만 개선
- 선형 외삽: 목표 달성에 k × 115배 필요 (k=23,000) → 비현실적
- 실제: Sharpe는 0.7~0.8에서 수렴 예상

---

## 근본 원인 분석

### 원인 1: 예측 정확도 너무 낮음 (80% 확률)
**증거**:
- MSE: 0.00015
- Correlation: 0.03~0.06 (매우 약함)
- Feature importance: 상위 feature도 약한 신호

**해결책**:
1. 더 강력한 Feature Engineering
   - Lag [20, 40, 60] (현재 [1, 5, 10])
   - Cross-sectional features (date 내 상대 순위)
   - Market regime features
2. 더 복잡한 모델
   - Neural Network (MLP, LSTM, Transformer)
   - Stacking Ensemble
3. Target Engineering
   - Sign 예측 (classification) 후 크기 예측 (regression)
   - Quantile regression

### 원인 2: Profit 최적화 무시 (15% 확률)
**증거**:
- utility = sharpe × profit
- 현재는 Sharpe만 최적화
- Profit proxy (pos_std): 0.311 → 0.857 (+2.75배)

**해결책**:
1. Confidence-based weighting
   - 예측 신뢰도 높을 때만 큰 position
2. Asymmetric position
   - Long/Short 비대칭 설정
3. Dynamic leverage
   - 시장 상황별 k 조정

### 원인 3: Volatility 무시 (5% 확률)
**증거**:
- 현재 `position = 1 + excess * k` (vol 무시)
- Vol Ratio: 1.23~1.46 (변동성 증가)

**해결책**:
- Volatility scaling: `position = 1 + (excess * k) / rolling_vol`
- 예상 효과: Sharpe +10~20% (0.7 → 0.84)

---

## 의사결정 트리

### 현재 상황
```
Sharpe: 0.699 (best k=3000)
Target: 6.0
Gap: 8.6배
```

### Path A: k 접근 포기 ✅ (선택)
**근거**:
- k × 15배 → Sharpe × 1.11배 (비효율)
- k=3000도 실패 → 더 높은 k 의미 없음
- 근본적 예측 정확도 문제

**Action**:
- EXP-007: 예측 정확도 개선
- 목표: Sharpe 1.5~3.0 (현재 0.7 대비 2~4배)

### Path B: k 접근 계속 ❌ (기각)
**근거**:
- k=5000~10000 시도?
- 예상 Sharpe: 0.75~0.80 (수렴)
- 여전히 목표 6.0의 1/8

**Action**: 시간 낭비, 포기

### Path C: 문제 재정의 (보류)
**질문**:
1. Kaggle 메트릭을 정확히 이해했나?
2. 17.395는 정말 realistic한 목표인가?
3. 리더보드 상위권은 어떻게 달성했나?

**Action**:
- Kaggle discussion, public notebook 조사
- Winning solution 분석 (대회 종료 후)
- Metric 재확인

---

## EXP-007 계획 (Pivot)

### 목표
**Sharpe 1.5~3.0 달성** (현재 0.7 대비 2~4배)

### 전략 1: Feature Engineering 확장 (H1)
**가설**: 더 긴 시계열 + 다양한 features로 예측력 향상

**실험**:
- Lag [1, 5, 10, 20, 40, 60]
- EMA [10, 20, 40, 60]
- Momentum: return_5d, return_20d, return_60d
- Cross-sectional: rank, quantile within date
- Volatility features: rolling_vol [5, 20, 60]

**예상**: Sharpe 0.8~1.0 (+14~43%)

### 전략 2: Volatility Scaling (H2)
**가설**: Vol-aware positioning으로 Sharpe 개선

**실험**:
- `position = 1 + (excess * k) / rolling_vol_20`
- Vol targeting
- Dynamic k by volatility regime

**예상**: Sharpe 0.8~0.9 (+14~29%)

### 전략 3: Ensemble (H3)
**가설**: 여러 모델 결합으로 안정성 향상

**실험**:
- XGBoost + LightGBM + Lasso
- Stacking (meta-learner)
- 예측 신뢰도 가중평균

**예상**: Sharpe 0.85~1.1 (+21~57%)

### 전략 4: Neural Network (H4)
**가설**: DL로 복잡한 패턴 포착

**실험**:
- MLP (3~5 layers)
- LSTM (시계열)
- Attention mechanism

**예상**: Sharpe 1.0~1.5 (+43~114%)

### 전략 5: Target Engineering (H5) - Radical
**가설**: 다른 target 정의로 예측 용이성 향상

**실험**:
- Classification (sign 예측) + Regression (크기)
- Quantile regression (tail 예측)
- Multi-task learning

**예상**: Sharpe 1.2~2.0 (+71~186%)

### 순서
1. **H1 + H2** (2~3시간): Feature + Vol scaling
2. **H3** (1시간): Ensemble
3. **성과 평가**:
   - Sharpe > 1.5 → Kaggle 제출, k 재조정
   - Sharpe < 1.5 → H4, H5 시도
4. **반복**: 목표 달성까지

---

## 교훈

### 1. k는 증폭기일 뿐, 신호 생성기 아님
- 약한 신호 (corr 0.03) × 큰 k (3000) = 노이즈
- 강한 신호 (corr 0.3) × 작은 k (100) > 약한 신호 × 큰 k

### 2. Metric 이해 필수
- Sharpe clamping [0, 6] 간과
- Profit이 핵심 변수임을 늦게 깨달음
- 초기에 metric 정확히 이해했어야 함

### 3. 빠른 Pivot 중요
- k=200~800 실험에서 이미 한계 보임 (+6%)
- k=1000~3000 실험은 확인 차원, 결과는 예상대로
- 더 빨리 포기하고 pivot 했어야 함

### 4. 근본 원인 우선
- 파라미터 튜닝 < 데이터/모델 개선
- k 튜닝 1주일 < Feature Eng 1주일

---

## 다음 단계

### 즉시 (오늘)
1. ✅ PIVOT.md 작성
2. ⏭️ `experiments/007/HYPOTHESES.md` 작성
3. ⏭️ EXP-007 실험 시작 (H1: Feature Eng)

### 중기 (이번 주)
- H1~H3 실험
- Sharpe 1.5+ 달성
- Kaggle 제출 (1회)

### 장기 (목표 달성까지)
- H4~H5 필요 시 시도
- 17.395 달성 또는 realistic 목표 재설정

---

## 결론

**EXP-006 실패**: k 파라미터 접근으로는 Sharpe 0.7 한계
**EXP-007 방향**: 예측 정확도 근본 개선 (Feature, Model, Ensemble)
**목표 수정**: 단계별 목표 (1.5 → 3.0 → 6.0)
**핵심 교훈**: 파라미터 튜닝 < 근본적 개선

---

**작성일**: 2025-10-13
**상태**: EXP-006 종료, EXP-007 시작
**다음 작업**: Feature Engineering 확장 (Lag 20~60, Cross-sectional)
