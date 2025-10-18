# Phase 2.1 Complete Summary - Interaction Features 실패

**완료일**: 2025-10-18
**소요 시간**: ~5분
**결과**: **실패** - 성능 하락 (-21.6%)

---

## 🎯 실험 목표 vs 달성

| 목표 | 달성 | 상태 |
|------|------|------|
| Top 20 baseline 재확인 | ✅ Sharpe 0.874 확인 | 완료 |
| Interaction features 추가 | ✅ 760개 생성 | 완료 |
| Sharpe 1.0+ 달성 | ❌ 0.874 → 0.686 (-21.6%) | **실패** |

---

## 📊 실험 결과

### Performance Comparison

| Feature Set | # Features | Sharpe | vs Baseline | 비고 |
|-------------|------------|--------|-------------|------|
| **Top 20 (baseline)** | **20** | **0.874** | baseline | **최고!** ✅ |
| Top 20 + Interactions | 780 | 0.686 | **-21.6%** | 실패 ❌ |

### Fold별 결과

**Baseline (Top 20):**
- Fold 1: 0.732
- Fold 2: 0.994
- Fold 3: 0.896
- **평균: 0.874 ± 0.133**

**With Interactions (780 features):**
- Fold 1: 0.626
- Fold 2: 0.735
- Fold 3: 0.696
- **평균: 0.686 ± 0.055**

**관찰:**
- 모든 fold에서 성능 하락
- 표준편차는 감소 (0.133 → 0.055) → 일관되게 나쁨
- Fold 2는 baseline에서 0.994로 매우 높았으나, interactions에서 0.735로 급락

---

## 💡 핵심 발견

### 1. **Feature Engineering ≠ 성능 향상**

Phase 1과 Phase 2.1을 종합하면:

| Approach | # Features | Sharpe | vs Top 20 | 전략 |
|----------|------------|--------|-----------|------|
| EXP-007 all features | 754 | 0.722 | -17.4% | Feature 많이 |
| **Top 20** | **20** | **0.874** | **baseline** | **Feature 선택** ✅ |
| Significant 57 | 57 | 0.689 | -21.2% | 통계적 선택 |
| Top 50 | 50 | 0.842 | -3.7% | Feature 선택 |
| Top 20 + Interactions | 780 | 0.686 | -21.5% | Feature 추가 |

**명확한 패턴:**
- Feature 추가 → 과적합 → 성능 하락
- Feature 줄임 → 과적합 제거 → 성능 향상
- **Top 20이 sweet spot**

### 2. **"Less is More" 확정**

- 20개 features가 최적
- 더 많이 = 더 나쁨
- 더 적게 = 더 나쁨 (50개까지는 OK, 20이 최고)

### 3. **Interaction Features의 문제**

760개 interaction features를 만들었지만:
- 대부분 noise
- 진짜 signal은 이미 Top 20에 포함
- 추가적인 interaction은 과적합만 유발

Top 20 interaction features (importance 기준):
1. E19_rolling_std_5_X_E19_vol_regime_60: 0.003105
2. V7_vol_norm_60_SUB_E19_vol_regime_60: 0.003038
3. V13_vol_norm_20_X_S2_ema_60: 0.002937
...

**문제:** importance가 매우 낮음 (0.003 수준)
- 원래 Top 20의 importance는 훨씬 높음
- Interaction features는 marginal contribution만

---

## 🔍 왜 실패했는가?

### 가설 1: 과적합 (Overfitting)
- 780 features는 8990 samples 대비 너무 많음
- XGBoost가 noise에 학습
- CV에서 일관되게 성능 하락

### 가설 2: Signal 이미 포함
- Top 20 features가 이미 충분한 signal
- Interaction은 redundant information
- 새로운 정보 추가 X

### 가설 3: 잘못된 Interaction
- 모든 pair의 multiply/divide/add/subtract는 너무 단순
- 금융 도메인 지식 없는 blind interaction
- 의미 없는 조합 대량 생성

---

## 📈 전체 실험 요약 (Phase 1 + 2.1)

### Phase 1: Feature Selection ✅ 성공
- 754 → 20 features
- Sharpe 0.722 → **0.874** (+21%)
- **Less is More 발견**

### Phase 2.1: Interaction Features ❌ 실패
- 20 → 780 features
- Sharpe 0.874 → 0.686 (-21.6%)
- **More is Worse 확인**

### 결론
**Feature Selection > Feature Engineering**
- 좋은 feature 고르기 > 많은 feature 만들기
- Simple > Complex

---

## 🚀 다음 전략

### ❌ 실패한 접근
- ~~Interaction features~~
- ~~더 많은 features 추가~~

### ✅ 가능한 접근

#### Option 1: Hyperparameter Tuning (추천 ⭐)
- Top 20 features 고정
- XGBoost hyperparameters 최적화
- 현재: n_estimators=300, lr=0.01, max_depth=5
- 목표: 0.874 → 0.95+
- 예상: +5~10% 가능

#### Option 2: Model Ensemble
- Top 20 features
- 여러 random_seed로 학습
- Averaging
- 예상: +3~5%

#### Option 3: 다른 Feature Selection 방법
- Top 15? Top 25?
- Forward selection
- Backward elimination
- 예상: +2~3%

#### Option 4: 한계 인정
- 0.874가 이 데이터의 limit일 수 있음
- 1.0은 unrealistic?
- EXP-007 0.749 대비 **+16.7% 이미 큰 성공**

---

## 🎯 추천 다음 단계

**Phase 3: Hyperparameter Tuning**
1. Top 20 features 고정
2. XGBoost hyperparameter grid search
3. 목표: Sharpe 0.95+
4. 예상 시간: 1~2시간
5. 성공 확률: 중~높음

**근거:**
- Feature engineering은 실패 (Phase 2.1)
- Feature selection은 성공 (Phase 1)
- 남은 개선 여지: 모델 최적화

---

## 📁 생성된 파일

```
experiments/016/results/
├── phase_2_1_results.csv          ← 실험 결과
├── phase_2_1_importance.csv       ← Feature importance
├── top_20_interactions.csv        ← Top 20 interaction features
└── PHASE_2_1_SUMMARY.md          ← 이 파일
```

---

**작성일**: 2025-10-18
**Phase 2.1 상태**: ❌ 실패 (성능 하락 -21.6%)
**현재 최고 Sharpe**: **0.874** (Top 20 features, Phase 1)
**다음**: Phase 3 - Hyperparameter Tuning (추천)

---

## 🔑 핵심 교훈

1. **Simple is Better**: 20 features > 780 features
2. **Feature Selection > Feature Engineering**: 좋은 것 고르기 > 많이 만들기
3. **Domain Knowledge 필요**: Blind interaction은 noise
4. **Overfitting 주의**: Feature 많으면 무조건 과적합
5. **실패도 가치**: "이건 안 된다"를 확인함

**Phase 2.1의 가치:**
- Interaction features가 안 되는 것을 확인
- Top 20이 정말 최적임을 재확인
- 다음 방향 명확해짐 (hyperparameter tuning)
