# EXP-020: Position Formula 근본 변경 전략

**목표**: Public Score 10점 달성을 위한 Position formula 근본 변경
**결과**: **CV Sharpe 0.6368** (+13.91% vs EXP-016)
**예상 Public Score**: **5.0-5.5** (EXP-016: 4.44)

---

## 실험 배경

### 이전 실험 결과
- **EXP-016**: Public Score 4.440 (CV 0.559) ✅ 최고 성과
- **EXP-019**: Public Score 3.599 (CV 3.541) ❌ Massive overfitting

### 문제 인식
1. Feature engineering 복잡도 증가는 역효과 (3-way, 4-way interactions)
2. Position formula 자체를 변경해야 함
3. Volatility-aware strategy 필요

### 목표 설정
- EXP-016 단순함 유지 (30 features, 2-way interactions)
- Position formula 근본 변경으로 10점 달성 시도

---

## 실험 과정

### Phase 1: Volatility Prediction Model
**목표**: EXP-016 features로 realized volatility 예측

**방법**:
- Target: Forward 20-day rolling std
- Model: XGBoost (EXP-016과 동일 architecture)
- Features: EXP-016 Top 30 features

**결과**:
- R² = -0.2832 (매우 낮음)
- MAE = 0.003579 (평균 volatility의 39%)
- RMSE = 0.004535

**결론**: Volatility 예측이 매우 어려움 (예상된 결과)

---

### Phase 2: Volatility-Scaled Strategy
**목표**: Volatility prediction 활용한 position sizing

**전략 테스트**:
1. Baseline (EXP-016): position = clip(1.0 + excess_pred * K, 0.0, 2.0)
2. Target Volatility (0.01)
3. Target Volatility (0.015)
4. Vol-Adjusted K
5. Inverse Vol Scaling

**결과**:
| Strategy | Sharpe | Improvement |
|----------|--------|-------------|
| **Target Volatility (0.015)** | **0.5740** | **+2.69%** |
| Target Volatility (0.01) | 0.5723 | +2.38% |
| Baseline (EXP-016) | 0.5590 | - |
| Vol-Adjusted K | 0.5458 | -2.36% |
| Inverse Vol Scaling | 0.5256 | -5.98% |

**결론**: 미미한 개선 (+2.69%), 10점 달성 부족

---

### Phase 3: Quantile Regression ✅
**목표**: Uncertainty quantification 기반 position sizing

**방법**:
- Quantile regression: q10, q50, q90
- Confidence interval: CI = q90 - q10
- Position = 1.0 + q50_pred * K * confidence
- Confidence = 1.0 / (|CI| + 0.001), clipped to [0.5, 5.0]

**전략 테스트**:
1. Baseline (median only)
2. CI-based (q90-q10)
3. CI-based (scaled x2)
4. **CI-based (scaled x5)** ✅
5. Asymmetric (upside focus)

**결과**:
| Strategy | Sharpe | Improvement |
|----------|--------|-------------|
| **CI-based (scaled x5)** | **0.6368** | **+13.91%** ✅ |
| CI-based (q90-q10) | 0.6333 | +13.29% |
| CI-based (scaled x2) | 0.6262 | +12.02% |
| Baseline (median only) | 0.6262 | +12.02% |
| Asymmetric (upside focus) | 0.5899 | +5.53% |

**결론**: 의미 있는 개선! Quantile regression 효과 확인

---

### Phase 4: Multi-Objective Optimization
**목표**: Return과 Risk 동시 최적화

**전략 테스트**:
1. Baseline (return only)
2. Return/Risk ratio
3. Return/Risk (scaled x20)
4. Weighted (ret=0.7, risk=0.3)
5. Weighted (ret=0.5, risk=0.5)
6. Sharpe-focused

**결과**:
| Strategy | Sharpe |
|----------|--------|
| **Baseline (return only)** | **0.5590** |
| Weighted (ret=0.7, risk=0.3) | 0.4329 |
| Weighted (ret=0.5, risk=0.5) | 0.3409 |
| Return/Risk ratio | 0.3161 |

**결론**: 모두 baseline보다 나쁨. Volatility prediction 정확도 부족

---

## 최종 결과 (Phase 3)

### CV Performance
- **CV Sharpe**: 0.6368 ± 0.2834
- **Fold ranges**: [0.352, 1.176]
- **Best fold**: Fold 4 (1.176)
- **Worst fold**: Fold 3 (0.352)

### 개선 폭
- **vs EXP-016** (0.5590): **+13.91%**
- **vs EXP-019** (3.5410): -81.4% (하지만 EXP-019는 overfitting)

### 예상 Public Score
| Estimate | Calculation | Score |
|----------|-------------|-------|
| Conservative | 0.637 × 7.9 × 0.9 | **4.5** |
| Expected | 0.637 × 7.9 | **5.0** |
| Optimistic | 0.637 × 7.9 × 1.1 | **5.5** |

**EXP-016 CV-to-Public ratio**: 7.9x (0.559 → 4.44)

---

## 핵심 인사이트

### 성공 요인 (Phase 3)
1. ✅ **Quantile regression**으로 불확실성 측정
2. ✅ **Confidence interval 기반** position sizing
3. ✅ **Narrow CI = High confidence** → Larger position
4. ✅ **Wide CI = Low confidence** → Smaller position
5. ✅ **EXP-016 단순함 유지** (30 features, 2-way interactions)

### 실패 요인 (Phase 2, 4)
1. ❌ **Volatility prediction 정확도 부족** (R² = -0.28)
2. ❌ Volatility 기반 전략이 오히려 역효과
3. ❌ Multi-objective approach 실패

### 교훈
1. **Uncertainty quantification이 중요**: Quantile regression으로 예측 신뢰도 측정
2. **Direct volatility prediction보다 indirect approach**: Quantile interval이 더 유용
3. **단순함 유지**: EXP-016의 30 features, 2-way interactions 유지
4. **Confidence-based sizing**: 확신 있는 예측에 더 큰 포지션

---

## 파일 구조

```
experiments/020/
├── README.md                                  # This file
├── ANALYSIS.md                                # 사전 분석 및 계획
├── phase1_volatility_prediction.py            # Volatility 예측 모델
├── phase2_volatility_scaled_strategy.py       # Volatility-scaled 전략
├── phase3_quantile_regression.py              # Quantile regression (BEST ✅)
├── phase4_multi_objective.py                  # Multi-objective optimization
└── results/
    ├── phase1_volatility_prediction.csv       # Phase 1 결과
    ├── phase1_config.csv
    ├── phase2_strategy_comparison.csv         # Phase 2 결과
    ├── phase3_quantile_comparison.csv         # Phase 3 결과 (BEST)
    └── phase4_multi_objective_comparison.csv  # Phase 4 결과

submissions/
└── submission_exp020.py                       # InferenceServer (Phase 3 구현)
```

---

## 비교

| Experiment | Strategy | CV Sharpe | Public Score | Note |
|------------|----------|-----------|--------------|------|
| **EXP-016** | Interaction Features | **0.559** | **4.440** | Baseline ✅ |
| **EXP-019** | Aggressive Ensemble | 3.541 | 3.599 | Overfitting ❌ |
| **EXP-020 P1** | Volatility Prediction | - | - | R² = -0.28 |
| **EXP-020 P2** | Volatility-Scaled | 0.574 | - | +2.69% |
| **EXP-020 P3** | **Quantile Regression** | **0.637** | **TBD** | **+13.91%** ✅ |
| **EXP-020 P4** | Multi-Objective | 0.559 | - | No improvement |

---

## 다음 단계

### Option 1: Kaggle 제출 (추천 ⭐⭐⭐⭐⭐)
- `submissions/submission_exp020.py` 업로드
- Public Score 확인
- **예상**: 5.0-5.5 (EXP-016 4.44 대비 +13-24%)

### Option 2: 추가 개선 시도
**가능한 방향**:
1. Quantile regression hyperparameter 튜닝
2. 다른 confidence scaling 시도 (x3, x4, x7, x10)
3. Ensemble: 여러 quantile_alpha 조합
4. Asymmetric CI: Upside/downside 비대칭 처리

**예상 개선**: +5-10% 추가

### Option 3: 다른 전략 탐색
**근본적 변경**:
1. **Regime switching**: Bull/Bear/Sideways 시장 구분
2. **Multi-strategy ensemble**: Trend following + Mean reversion
3. **Direct Sharpe optimization**: Differentiable Sharpe loss (EXP-011 재시도)

**예상 개선**: +20-50% (불확실)

---

## 10점 달성 전망

### 현실적 평가
- **EXP-020 Phase 3**: CV 0.637, 예상 Public 5.0-5.5
- **목표**: 10점
- **Gap**: 2배 부족

### 10점 달성을 위한 필요 조건
1. **CV Sharpe 1.3 이상** (현재 0.637의 2배)
2. 또는 **CV-to-Public ratio 증가** (현재 7.9x → 15x 이상)

### 달성 확률
- **EXP-020 단독**: 10-15% (보수적)
- **추가 개선**: 20-30%
- **근본적 변경**: 30-40%

**결론**: 10점 달성은 어렵지만, 5-6점은 충분히 가능

---

## 핵심 교훈 (전체 실험)

### EXP-016 vs EXP-019 vs EXP-020

**EXP-016** (최고 Public Score):
- ✅ Simple 2-way interactions
- ✅ 30 features
- ✅ Fixed K=250
- ✅ CV 0.559 → Public 4.44 (ratio 7.9x)

**EXP-019** (Overfitting):
- ❌ Complex 3-way, 4-way interactions
- ❌ Kelly Criterion + 5-model ensemble
- ❌ CV 3.54 → Public 3.60 (ratio 1.0x)

**EXP-020** (개선):
- ✅ EXP-016 단순함 유지
- ✅ Quantile regression (uncertainty)
- ✅ Confidence-based sizing
- ✅ CV 0.637 → 예상 Public 5.0-5.5

### 핵심 원칙
1. **Simple > Complex**: 2-way interactions까지만 유효
2. **Uncertainty > Volatility**: Quantile CI > Direct volatility prediction
3. **Confidence-based sizing**: 확신 있을 때 크게 배팅
4. **CV-to-Public ratio 모니터링**: 7-8x가 정상, 1x는 overfitting

---

**작성일**: 2025-11-04
**상태**: Phase 1-5 완료, 제출 준비 완료
**추천**: Option 1 (Kaggle 제출 및 Public Score 확인)
