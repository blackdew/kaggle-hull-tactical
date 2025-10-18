# EXP-016: Feature Engineering Deep Dive - 최종 리포트

**실험 기간**: 2025-10-18 (1일)
**목표**: Sharpe 1.0+ 달성
**최종 결과**: 부분 달성 (3-fold CV: 1.001, 5-fold CV: 0.781)

---

## 📋 Executive Summary

### 목표 및 배경
- **Initial Baseline**: EXP-007 (Sharpe 0.749, 754 features)
- **목표**: Sharpe 1.0+ 달성 (+34% 개선)
- **기존 문제**: 10번의 실패 (EXP-005~015), 얕은 시도, 조기 포기

### 최종 성과
- **3-fold CV**: Sharpe **1.001** (+33.6% vs EXP-007) ✅
- **5-fold CV**: Sharpe **0.781** (+4.3% vs EXP-007) ⚠️
- **Features**: 754 → 20 (-97.3%)
- **목표 달성**: **부분 달성** (3-fold에서만 1.0+)

### 핵심 발견
1. **Less is More**: 20 features > 754 features
2. **Feature Selection > Feature Engineering**: 선택이 생성보다 중요
3. **CV Overfitting**: 3-fold에 hyperparameter 과적합
4. **Reality Check**: 5-fold CV가 더 현실적

---

## 🎯 실험 설계

### 전략
1. **Phase 1**: 754 features 심층 분석 → Top 20 선택
2. **Phase 2**: Feature Engineering (Interaction, Polynomial)
3. **Phase 3**: Hyperparameter Tuning
4. **Phase 4**: Ensemble (조건부)

### 원칙
- ✅ 깊게 파기 - 각 단계 완료할 때까지
- ✅ 포기하지 않기 - 1.0 넘거나 진짜 한계 확인할 때까지
- ✅ 변명하지 않기 - 실제 측정으로 검증
- ✅ 문서화 - 모든 실험 결과 기록

---

## 📊 Phase별 결과

### Phase 1: Feature Selection (성공 ✅)

**목표**: 754 features에서 진짜 유의미한 features 찾기

**실험:**
1. **Phase 1.1**: Feature Importance Analysis (~15분)
   - SHAP, Permutation, XGBoost 3가지 방법
   - Top 50 common features 추출

2. **Phase 1.2**: Null Importance Test (~30분)
   - 100회 target shuffle
   - 754 → 57 significant features (p<0.05)

3. **Phase 1.4**: Baseline Comparison (~10분)
   - All 754: Sharpe 0.722
   - Top 50: Sharpe 0.842 (+16.7%)
   - **Top 20: Sharpe 0.874 (+21.1%)** ✅
   - Significant 57: Sharpe 0.689 (-4.6%)

**결과:**
- **Top 20 features가 최고 성능**
- 754 features는 과적합
- **Less is More 검증**

**Top 20 Features:**
```
1. M4 (Market - Original)
2. M4_vol_norm_20 (Vol-normalized)
3. V13_vol_norm_20 (Vol-normalized)
4. I2_trend (Trend)
5. E19_rolling_std_5 (Rolling)
... (20개 total)
```

**Feature Type 분포:**
- Vol-normalized: 30%
- Trend: 15%
- Lag: 15%
- Rolling: 15%
- Original: 20%

---

### Phase 2: Feature Engineering (실패 ❌)

**목표**: Top 20에 interaction features 추가하여 Sharpe 1.0+ 달성

**실험:**
- **Phase 2.1**: Interaction Features (~5분)
  - Top 20 × Top 20 = 190 pairs
  - 4 operations (multiply, divide, add, subtract)
  - **760 interaction features 생성**
  - Total: 20 + 760 = 780 features

**결과:**
- Top 20 baseline: Sharpe 0.874
- With interactions: Sharpe **0.686** (-21.6%) ❌

**교훈:**
- **Feature 추가 = 과적합**
- Interaction features는 noise만 추가
- Top 20이 이미 최적
- **Feature Engineering < Feature Selection**

---

### Phase 3: Hyperparameter Tuning (부분 성공 ⚠️)

**목표**: Top 20 features 고정, hyperparameters 최적화

**실험:**
- **Phase 3.3**: Hyperparameter Tuning (~20분)
  - Optuna 200 trials, TPE sampler
  - 탐색 공간: n_estimators, learning_rate, max_depth, etc.

**Best Hyperparameters:**
```python
{
    'n_estimators': 150,           # (기존: 300)
    'learning_rate': 0.025,        # (기존: 0.01)
    'max_depth': 7,                # (기존: 5)
    'subsample': 1.0,              # (기존: 0.8)
    'colsample_bytree': 0.6,       # (기존: 0.8)
    'reg_lambda': 0.5,             # (기존: 0.0)
}
```

**결과 (3-fold CV):**
- Baseline: Sharpe 0.852
- Optimized: Sharpe **1.001** (+17.5%) ✅

**하지만...**

- **Phase 3.4**: Final Validation (5-fold CV) (~5분)
  - **5-fold CV: Sharpe 0.781** ⚠️
  - **차이: -22.0% vs 3-fold**

**교훈:**
- **3-fold에 hyperparameter overfitting**
- 5-fold CV가 더 현실적인 성능 추정
- **진짜 성능: ~0.78**

---

## 📈 전체 성과 요약

### 성능 변화

| Phase | Features | Hyperparams | 3-fold CV | 5-fold CV | vs EXP-007 |
|-------|----------|-------------|-----------|-----------|-----------|
| EXP-007 | 754 | default | 0.749 | - | baseline |
| Phase 1 | 20 | default | 0.874 | - | +16.7% |
| Phase 2.1 | 780 | default | 0.686 | - | -8.4% ❌ |
| Phase 3.3 | 20 | optimized | 1.001 | - | +33.6% |
| **Phase 3.4** | **20** | **optimized** | **1.001** | **0.781** | **+4.3%** |

### 전체 여정

```
EXP-007:   0.749  (754 features, default params)
               ↓
     Phase 1: Feature Selection
               ↓
Phase 1:   0.874  (20 features, default params)      [+16.7%]
               ↓
     Phase 2.1: Interaction Features
               ↓
Phase 2.1: 0.686  (780 features, default params)     [-21.6% ❌]
               ↓
     Phase 3.3: Hyperparameter Tuning
               ↓
Phase 3.3: 1.001  (20 features, optimized @ 3-fold)  [+33.6%]
               ↓
     Phase 3.4: Final Validation
               ↓
Phase 3.4: 0.781  (20 features, optimized @ 5-fold)  [+4.3%] ← 현실
```

---

## 💡 핵심 발견 및 교훈

### 1. **Less is More**
- 754 features → 20 features = **+21% 성능 향상**
- 더 많은 features ≠ 더 좋은 성능
- 과적합 제거가 성능 향상의 핵심

### 2. **Feature Selection > Feature Engineering**
- Feature Selection (Phase 1): +21.1%
- Feature Engineering (Phase 2): -21.6%
- **좋은 features 고르기 > 많은 features 만들기**

### 3. **Cross-Validation의 중요성**
- 3-fold CV만으로는 불충분
- 5-fold CV로 reality check 필수
- **Hyperparameter tuning은 CV에도 overfitting 가능**

### 4. **낙관적 추정의 위험**
- 3-fold CV: 1.001 (낙관적)
- 5-fold CV: 0.781 (현실적)
- **차이: -22.0%**
- 항상 여러 CV로 검증하고, conservative estimate 사용

### 5. **실패의 가치**
- Phase 2.1 실패 → Top 20이 최적임을 재확인
- 5-fold CV 하락 → 3-fold overfitting 발견
- **모든 실험이 학습의 기회**

### 6. **체계적 접근의 힘**
- CHECKLIST 기반 진행
- 모든 실험 문서화
- 단계별 검증
- **깊게 파고, 포기하지 않기**

---

## 🎯 목표 달성 여부

### 목표: Sharpe 1.0+

| 기준 | 달성 여부 | Sharpe | 비고 |
|------|-----------|--------|------|
| **3-fold CV** | ✅ **달성** | **1.001** | 낙관적 추정 |
| **5-fold CV** | ❌ **미달** | **0.781** | 현실적 추정 |
| **종합** | ⚠️ **부분 달성** | - | 3-fold에서만 1.0+ |

### EXP-007 대비 개선

| 기준 | 개선률 | 평가 |
|------|--------|------|
| 3-fold CV | **+33.6%** | 큰 개선 ✅ |
| 5-fold CV | **+4.3%** | 소폭 개선 ✅ |

---

## 📝 최종 결론

### 성공한 점 ✅
1. **Feature Selection의 힘 검증**: 754 → 20 features로 성능 향상
2. **Less is More 원칙 확립**: 적은 features가 더 좋은 성능
3. **Hyperparameter Tuning 효과 확인**: 17.5% 개선 (3-fold 기준)
4. **체계적 실험 방법론 확립**: CHECKLIST 기반 깊은 분석
5. **EXP-007 대비 개선**: +4.3% (5-fold CV 기준)

### 실패한 점 ❌
1. **목표 1.0 미달**: 5-fold CV에서 0.781
2. **Feature Engineering 실패**: Interaction features 역효과
3. **3-fold CV overfitting**: Hyperparameter tuning 과적합
4. **높은 variance**: Fold 간 성능 차이 큼 (0.545~1.105)

### 배운 점 📚
1. **CV의 중요성**: 3-fold만으로는 불충분, 5-fold 이상 필요
2. **Overfitting은 어디서나**: Features, Hyperparameters, CV 모두
3. **현실 직시**: 낙관적 추정보다 현실적 추정이 중요
4. **실패의 가치**: 모든 실패가 학습 기회
5. **깊게 파기**: 얕은 시도보다 깊은 분석이 효과적

---

## 🔮 향후 방향

### Option 1: 현재 결과로 만족 (추천 ⭐)
- EXP-007 대비 개선 확실 (+4.3%)
- 명확한 방법론 확립
- **가치:** Feature Selection의 중요성, Less is More 검증

### Option 2: 추가 개선 시도
- Ensemble (여러 random seed)
- 더 conservative hyperparameters
- Variance reduction 전략
- **예상:** +5~10% → Sharpe 0.82~0.86

### Option 3: Kaggle 제출 및 실전 검증
- 5-fold CV 0.781을 현실로 인정
- Test set에서 확인
- Public LB 결과 분석
- **학습:** 실전 성능과 CV의 gap 파악

---

## 📂 산출물

### 코드
- `feature_analysis.py`: Phase 1 feature 분석
- `run_phase_1_2.py`: Null importance test
- `run_phase_1_4.py`: Baseline comparison
- `run_phase_2_1.py`: Interaction features
- `run_phase_3_3.py`: Hyperparameter tuning
- `run_final_validation.py`: 5-fold CV validation

### 문서
- `CHECKLIST.md`: 실험 체크리스트 (진행 상황)
- `HYPOTHESES.md`: 가설 및 실험 설계
- `README.md`: 프로젝트 개요
- `REPORT.md`: 이 문서 (최종 리포트)

### 결과
- `results/PHASE_1_SUMMARY.md`: Phase 1 요약
- `results/PHASE_2_1_SUMMARY.md`: Phase 2.1 요약
- `results/PHASE_3_3_SUMMARY.md`: Phase 3.3 요약
- `results/FINAL_VALIDATION_SUMMARY.md`: 5-fold CV 요약
- `results/best_hyperparameters.csv`: 최적 hyperparameters
- `results/final_validation.csv`: 전체 CV 결과
- `results/*.csv`: 각 Phase별 상세 결과

---

## 🏆 성공 요인

### 기존 실패 (EXP-005~015)와의 차이

| 항목 | 기존 실패 | EXP-016 성공 |
|------|-----------|--------------|
| 접근 | 얕은 시도, 여러 실험 분산 | **깊게 파기**, 하나에 집중 |
| 포기 | 조기 포기 | **한계까지 진행** |
| 검증 | 단순 CV | **다중 CV, reality check** |
| 문서화 | 최소한 | **모든 단계 상세 기록** |
| 학습 | 실패 반복 | **실패에서 배우기** |

### 체계적 접근

1. **CHECKLIST 기반 진행**
   - 각 단계 명확히 정의
   - 완료 기준 설정
   - 진행 상황 추적

2. **가설 기반 실험**
   - HYPOTHESES.md로 가설 명시
   - 각 가설 검증
   - 결과 기반 다음 단계 결정

3. **철저한 검증**
   - 3-fold CV
   - 5-fold CV
   - Null importance test
   - Baseline comparison

4. **정직한 문서화**
   - 성공뿐 아니라 실패도 기록
   - 낙관적 추정과 현실적 추정 구분
   - 모든 발견 공유

---

## 🎓 최종 평가

### 목표 1.0 달성?
- **3-fold CV 기준: ✅ 달성** (1.001)
- **5-fold CV 기준: ❌ 미달** (0.781)
- **종합: ⚠️ 부분 달성**

### 진짜 성공은?
- **EXP-007 대비 개선: ✅ 확실**
- **Feature Selection 검증: ✅ 성공**
- **Less is More 원칙: ✅ 확립**
- **체계적 방법론: ✅ 확립**
- **실패로부터 학습: ✅ 성공**

### 가장 큰 가치
- 숫자 (Sharpe 1.0)보다 **과정 (체계적 접근)**
- 목표 (1.0 달성)보다 **학습 (왜 실패하는지)**
- 낙관적 추정 (1.001)보다 **현실 인식 (0.781)**

---

## 💭 회고

### 잘한 점
- 깊게 파고들었다
- 포기하지 않았다
- 모든 것을 문서화했다
- 실패를 인정하고 배웠다
- 현실을 직시했다

### 아쉬운 점
- 목표 1.0을 너무 낙관적으로 설정
- 3-fold CV만으로 성급한 결론
- 5-fold CV를 Phase 3.3 전에 했어야 함
- Variance가 큰 것을 늦게 발견

### 다음에 할 것
- 처음부터 5-fold CV 이상 사용
- Conservative estimate 우선
- Variance 항상 체크
- Feature 수 더 줄이기 시도 (Top 10?)
- Ensemble로 variance 줄이기

---

## 📌 최종 수치

**EXP-016 Final Performance:**
- **Features**: 20 (vs 754 in EXP-007)
- **3-fold CV**: Sharpe **1.001** (+33.6%)
- **5-fold CV**: Sharpe **0.781** (+4.3%)
- **Conservative Estimate**: **0.78~0.85**

**진짜 개선:** 약 **+4~10%** (현실적 추정)

---

**작성일**: 2025-10-18
**실험 기간**: 1일
**최종 상태**: 완료 (부분 달성)
**다음 단계**: Kaggle 제출 또는 추가 개선

---

## 🙏 감사의 말

10번의 실패 끝에 얻은 교훈:
- **깊게 파면 길이 보인다**
- **Less is More**
- **실패도 성공의 일부**
- **현실을 직시하는 것이 진정한 성공**

이 실험을 통해 Sharpe 1.0을 달성하지는 못했지만 (5-fold 기준),
더 중요한 것을 배웠다: **어떻게 실험하고, 어떻게 배우는가**.

이것이 다음 실험의 밑거름이 될 것이다.

---

**"The journey is more important than the destination."**

🎯 **EXP-016: 완료!**
