# 실험 전체 회고 및 결론 (EXP-005~016)

## 최종 결과 🏆

**Public Score: 4.440** (2025-10-21, EXP-016 v2)
- 이전 최고 (Version 9): 0.724
- **개선: 6.1배 향상** ✅
- 접근: InferenceServer 호환 + Interaction Features

## 목표
- **최종 목표**: Kaggle utility 17+ (목표치)
- **필요 조건**: Sharpe > 1.0 (최소), Sharpe > 3.0 (이상적)
- **달성**: Public Score 4.440 (목표치는 아니지만 큰 성과)

---

## 실험 경과 (전체)

### EXP-005: 모델 전환 (Lasso → XGBoost)
**접근**: 비선형 모델로 예측력 향상
- Lasso → XGBoost/LightGBM + Feature Engineering
- **결과**: CV Sharpe 0.627, **Kaggle 0.724** ✅
- **개선**: 이전 0.441 대비 1.64배

**평가**: 성공 (모델 전환 효과 확인)

### EXP-006: k 파라미터 최적화
**접근**: k를 대폭 증가시켜 Sharpe 향상
- k=200 → 3000 (15배 증가)
- **결과**: CV Sharpe 0.627 → 0.699 (+11.4%)
- **실패 원인**: k는 신호 증폭만 가능, 신호 생성 불가

**평가**: 실패 (근본적 한계 확인)

### EXP-007: Feature Engineering 확장
**접근**: 예측 정확도 근본 개선
- 234 features → 754 features (3.2배)
- Lag 60, Cross-sectional, Volatility, Momentum 추가
- **결과**: CV Sharpe 0.749 (+19.5% vs 0.627)
- MSE 0.000150 → 0.000126 (-16%)

**평가**: 제한적 성공 (개선되었지만 목표에 크게 부족)

### EXP-008~010: 딥러닝 시도 (삭제됨)
**접근**: Classification, Autoencoder, Transformer
- EXP-008: Binary classification
- EXP-009: Autoencoder multi-task learning
- EXP-010: Temporal Transformer
- **결과**: 모두 0.75 이하로 실패

**평가**: 실패 (삭제됨)

### EXP-011: Direct Utility Optimization
**접근**: Utility 함수 직접 최적화
- Differentiable Sharpe loss 사용
- **결과**: Sharpe 0.552 (오히려 하락)

**평가**: 실패 (목표 함수 직접 최적화도 효과 없음)

### EXP-012: Missing Pattern Recognition
**접근**: E features 등장 패턴 활용
- Missing indicator features
- Period-aware features
- **결과**: Sharpe 0.647

**평가**: 실패 (baseline 대비 하락)

### EXP-013: Technical Analysis
**접근**: 차트 분석 (RSI, MACD, Bollinger Bands)
- Rule-based strategy
- ML with technical indicators
- **결과**: Sharpe 0.483 (rule-based)

**평가**: 실패 (기술적 지표만으로는 부족)

### EXP-014: Multi-variate LSTM
**접근**: 94개 feature를 시계열로 동시 학습
- SimpleLSTM: 2 layers, hidden=128
- **결과**: Sharpe 0.471 (3-fold CV)

**평가**: 실패 (딥러닝 첫 시도, 매우 낮은 성능)

### EXP-015: Transformer + Residual Connections
**접근**: LSTM보다 강력한 Transformer 적용
- Pre-LN architecture + Residual connections
- Tiny: d=64, 2 heads, 2 layers → Sharpe 0.257
- Medium: d=96, 3 heads, 2 layers → Sharpe 0.299
- **결과**: LSTM(0.471)보다도 훨씬 나쁨

**평가**: 실패 (데이터 부족, 짧은 sequence로 Transformer 비효율)

### EXP-016 v2: InferenceServer + Interaction Features ✅ 🏆
**접근**: 완전 재설계 - 1-row 계산 가능한 features만 사용
- **문제 발견**: 기존 lag/rolling features는 InferenceServer에서 사용 불가 (row-by-row 예측)
- **해결책**:
  1. Top 20 원본 features 선택
  2. Interaction features 120개 생성 (곱셈, 나눗셈, 다항식)
  3. XGBoost로 Top 30 선택
  4. K=250 최적화
- **결과**: **Public Score 4.440** (이전 0.724 대비 6.1배 향상)
- CV Sharpe: 0.559 (낮아 보였지만 Public에서 폭발)

**평가**: 대성공 🎉
- ✅ InferenceServer 제약 극복
- ✅ Interaction features의 강력한 효과
- ✅ 완전 재설계의 용기
- ✅ 최고 성과 달성

---

## 현재 상황 (EXP-016 종료 기준, 2025-10-21)

### 전체 실험 성과 요약
| Experiment | Method | CV Sharpe | Public Score | Note |
|------------|--------|-----------|--------------|------|
| **EXP-005** | XGBoost (234 feat) | 0.627 | 0.724 | baseline |
| **EXP-006** | XGBoost + k튜닝 | 0.699 | - | +11.5% |
| **EXP-007** | XGBoost (754 feat) | 0.749 | - | +19.5% (CV 최고) |
| **EXP-008~010** | Deep Learning | <0.75 | - | 실패 (삭제) |
| **EXP-011** | Direct Utility Opt | 0.552 | - | -26.2% |
| **EXP-012** | Missing Patterns | 0.647 | - | -13.6% |
| **EXP-013** | Technical Analysis | 0.483 | - | -35.5% |
| **EXP-014** | Multi-variate LSTM | 0.471 | - | -37.1% |
| **EXP-015** | Transformer+Residual | 0.257~0.299 | - | -60% ~ -66% |
| **EXP-016 v2** | **Interaction Features** | **0.559** | **4.440** | **🏆 +514% (Public)** |

### 돌파구: EXP-016 v2의 성공
- **CV Sharpe**: 0.559 (EXP-007보다 낮음)
- **Public Score**: **4.440** (이전 0.724 대비 **6.1배**)
- **핵심**: InferenceServer 제약 이해 + Interaction Features

### 핵심 발견
1. **InferenceServer 제약이 결정적** - lag/rolling features 사용 불가
2. **Interaction Features의 힘** - 곱셈, 나눗셈, 다항식이 비선형 관계 포착
3. **완전 재설계의 용기** - 기존 접근 포기하고 처음부터 다시
4. **CV ≠ Public Score** - CV 0.559 → Public 4.440 (예상 밖 성공)
5. **XGBoost가 여전히 최강** - 딥러닝 모두 실패, XGBoost가 답
6. **Quality > Quantity** - 30 features > 754 features

---

## 근본 원인 분석

### 1. 예측 정확도의 한계 (핵심 문제)

**증거:**
```
MSE: 0.000126 (754 features)
MSE: 0.000150 (234 features)
개선: 16% (520 features 추가로)
```

**해석:**
- Features를 3배 늘렸지만 MSE는 16%만 감소
- **신호 자체가 매우 약함** (SNR 낮음)
- 더 많은 feature 추가해도 한계 명확

**결론**: 이 데이터로 excess return 예측은 근본적으로 어려움

### 2. Sharpe 0.7~0.8이 실질적 상한

**증거:**
- EXP-005 (simple): 0.627
- EXP-006 (k↑): 0.699
- EXP-007 (features↑): 0.749
- k 재조정: 0.741 (k=600이 최적)

**패턴:**
- 다양한 접근으로도 0.75 이상 넘기 어려움
- 0.7~0.8 구간에서 수렴하는 양상

**결론**: Sharpe 0.75가 현재 접근의 실질적 상한

### 3. Kaggle Metric의 특성

**재확인:**
```
utility = min(max(sharpe, 0), 6) × Σ profits
```

- Sharpe 0.75 << 6.0 (목표의 1/8)
- Sharpe를 6까지 올리는 것이 핵심
- 현재 접근으로는 불가능

---

## 시도한 방법들과 한계

### ✅ 시도한 것
1. **모델 전환**: Lasso → XGBoost/LightGBM
2. **k 파라미터**: 200 → 3000
3. **Feature Engineering**:
   - Longer lags (60일)
   - Cross-sectional (rank, zscore)
   - Volatility features
   - Momentum & Trend
4. **Feature 수**: 234 → 754

### ❌ 시도하지 않은 것
1. **Volatility Scaling**: position = (1 + excess*k) / rolling_vol
2. **Ensemble**: XGBoost + LightGBM + Lasso
3. **Neural Network**: LSTM, Transformer
4. **Target Re-engineering**: Classification + Regression
5. **전략 변경**: Regime switching, Dynamic leverage
6. **외부 데이터**: 경제 지표, 다른 시장 데이터

### 왜 시도하지 않았는가?
- Volatility Scaling: 예상 +10~15% (0.75 → 0.86, 여전히 부족)
- Ensemble: 예상 +5~10% (0.75 → 0.82, 부족)
- Neural Network: 과적합 위험 높음, 성공 확률 5%
- Target Re-engineering: 근본적 재설계 필요, 시간 많이 소요
- 전략 변경: position formula 자체 변경, 검증 어려움

**공통점**: 모두 Sharpe를 6.0까지 올리기에는 부족

---

## 왜 Sharpe 6.0이 어려운가?

### 가설 1: 데이터 자체의 예측 가능성 한계 (90% 확률)

**근거:**
- Excess return의 autocorrelation 매우 낮음
- Feature와 target의 correlation 0.03~0.06 (매우 약함)
- 시장 효율성 (EMH): 초과 수익 예측 어려움

**의미:**
- 이 데이터로 Sharpe 6.0은 **이론적으로 불가능할 수 있음**
- 어떤 모델을 써도 0.7~1.0이 상한

**검증 필요:**
- Target의 예측 가능성 upper bound 계산
- 다른 참가자들의 성과 확인 (대회 종료 후)

### 가설 2: 접근 방식 자체가 잘못됨 (10% 확률)

**가능성:**
- Regression으로 excess return 예측하는 것 자체가 비효율
- Classification (sign 예측)이 더 나을 수 있음
- 다른 target 정의 필요 (volatility, drawdown 등)
- Portfolio optimization 접근 필요

**검증 필요:**
- Top solution 분석 (대회 종료 후)
- 다른 접근 방식 시도

---

## 현실적 평가

### 달성 가능한 것
- ✅ CV Sharpe 0.85~1.0 (Volatility Scaling + Ensemble)
- ✅ Kaggle utility 1.5~3.0 (현재 0.724 대비 2~4배)
- ✅ 체계적 실험 및 문서화

### 달성 불가능한 것
- ❌ Sharpe 6.0 (현재 0.75의 8배)
- ❌ Kaggle utility 17.395
- ❌ 현재 접근으로는 근본적 한계

---

## 다음 실험 방향 제시

### Option 1: 현재 최선으로 마무리 (추천 ⭐⭐⭐⭐⭐)

**행동:**
1. EXP-007 결과 정리 및 커밋
2. 전체 회고 문서 작성
3. 목표를 현실적으로 재설정
   - 기존: Sharpe 6.0, Kaggle 17.395
   - 수정: Sharpe 0.85~1.0, Kaggle 1.5~3.0
4. 필요 시 Volatility Scaling + Ensemble 시도
5. 최종 제출 (optional)

**소요 시간**: 1~2시간 (문서 작업)

**장점:**
- 명확한 마무리
- 배운 점 정리
- 현실적 목표 설정

### Option 2: 근본적 접근 변경 (도전 ⭐⭐⭐)

**행동:**
1. **Problem Re-definition**
   - Regression → Classification
   - Target 재정의
   - Portfolio optimization 접근

2. **EXP-008: Classification Approach**
   - Target: sign(excess_return)
   - Model: Binary classifier
   - Strategy: Long if prob > 0.6, Short if prob < 0.4
   - 예상: Sharpe 1.2~2.0 (초낙관적)

3. **EXP-009: Portfolio Optimization**
   - Markowitz Mean-Variance
   - Risk parity
   - Black-Litterman
   - 예상: Sharpe 1.5~2.5 (매우 낙관적)

**소요 시간**: 10~20시간

**리스크:**
- 성공 확률 10~20%
- 시간 많이 소요
- 여전히 6.0 달성 불가능할 가능성

### Option 3: 대회 종료 후 Top Solution 분석 (학습 ⭐⭐⭐⭐⭐)

**행동:**
1. 대회 종료 기다리기
2. Winning solution 분석
3. 17.395가 realistic했는지 확인
4. 다른 참가자들의 접근 학습
5. 필요 시 재실험

**소요 시간**: 대회 종료 후

**장점:**
- 정답 확인
- 효율적 학습
- 시간 낭비 방지

---

## 핵심 교훈

### 1. 파라미터 튜닝 < 근본적 개선
- k 튜닝 (EXP-006): +11.4%
- Feature Engineering (EXP-007): +19.5%
- 하지만 둘 다 목표에는 부족

### 2. 데이터 품질이 성능 상한 결정
- 약한 신호 (corr 0.03~0.06)는 어떤 모델로도 극복 불가
- MSE 0.000126이 현실적 하한

### 3. 목표 설정의 중요성
- 17.395가 비현실적일 가능성
- 초기에 realistic 여부 검증 필요

### 4. 빠른 Pivot의 중요성
- EXP-006에서 이미 한계 보였음
- 더 빠르게 접근 전환했어야 함

### 5. 체계적 실험의 가치
- HYPOTHESES → 실험 → REPORT → PIVOT
- 실패해도 배움이 있음
- 문서화로 재현성 확보

---

## 최종 결론 (EXP-016 기준, 2025-10-21)

### 현실
- **달성**: **Public Score 4.440** (EXP-016 v2) 🏆
- **CV Sharpe**: 0.559 (낮았지만 Public에서 폭발)
- **이전 최고**: 0.724 (Version 9)
- **개선**: **6.1배 향상**
- **결론**: InferenceServer 제약을 이해하고 완전 재설계로 돌파구 찾음

### 주요 교훈

**1. 제약 조건이 설계를 결정한다**
- InferenceServer = row-by-row 예측
- lag/rolling features 사용 불가 (과거 데이터 필요)
- 제약을 초기에 이해했어야 10~13번의 실패 제출 방지 가능

**2. Interaction Features의 놀라운 효과**
- 곱셈: `P8*S2`, `M4*V7` (비선형 관계)
- 나눗셈: `P8/P7`, `M4/S2` (상대적 변화)
- 다항식: `M4²`, `V13²` (비선형 패턴)
- 120개 생성 → Top 30 선택 = **6.1배 성능 향상**

**3. 완전 재설계의 용기**
- EXP-016 초기 버전 (CV Sharpe 1.001) 포기
- Sunk cost fallacy 극복
- 처음부터 다시 설계 → 최고 성과 달성

**4. XGBoost의 압도적 우위**
- 딥러닝(LSTM, Transformer)은 XGBoost의 절반 수준
- Feature Engineering이 핵심
- 작은 데이터셋에서는 전통적 ML이 강력

**5. CV Score ≠ Public Score**
- CV Sharpe 0.559 (보통)
- Public Score 4.440 (최고!)
- Metric 차이, Test set 특성 고려 필요

**6. Quality > Quantity**
- 754 features (EXP-007): CV 0.749, Public -
- 30 features (EXP-016): CV 0.559, **Public 4.440**
- 많은 features보다 의미 있는 features

### 성과
- ✅ 12개 실험 완료 (EXP-005~016)
- ✅ 체계적 실험 프로세스 확립
- ✅ **Public Score 4.440 달성** (최고 기록!) 🏆
- ✅ InferenceServer 제약 극복
- ✅ Interaction Features 효과 입증
- ✅ 딥러닝 실패 원인 파악
- ✅ 완전 재설계 경험
- ✅ 전체 문서화 완료

### 돌파구
- ✅ **EXP-016 v2**: 완전 재설계로 6.1배 향상
- ✅ InferenceServer 호환 설계
- ✅ Interaction Features의 힘 입증
- ✅ CV와 Public Score 차이 경험

### 여전히 미달성
- ❌ 목표 utility 17+ (하지만 4.440은 큰 성과)
- ❌ Sharpe 6.0 (Public Score와 Sharpe는 다른 metric)
- ❌ 딥러닝으로 XGBoost 능가 (여전히 XGBoost가 최강)

### 다음 방향 제안

**Option 1: 현재 결과로 마무리** ⭐⭐⭐⭐⭐ (추천)
- Public Score 4.440은 충분히 좋은 성과
- 문서화 완료
- 다른 대회로 이동

**Option 2: 추가 개선 시도** ⭐⭐⭐
- Interaction features 추가 탐색
- Ensemble (여러 K 값)
- Hyperparameter fine-tuning
- 예상: Public Score 5~7

**Option 3: 대회 종료 후 분석** ⭐⭐⭐⭐⭐ (추천)
- Winning solution 학습
- Private Score 확인
- Top 참가자들의 접근 분석

---

**작성일**: 2025-10-21 (업데이트)
**상태**: EXP-005~016 완료, **Public Score 4.440 달성** 🏆
**추천**: Option 1 (마무리) or Option 3 (Top Solution 분석 대기)

---

## 부록: 실험 산출물 (전체)

### 문서
- `experiments/005/REPORT.md`: EXP-005 전체 결과
- `experiments/006/PIVOT.md`: k 접근 실패 분석
- `experiments/007/HYPOTHESES.md`: Feature Engineering 계획
- `experiments/007/ANALYSIS.md`: 현실적 가능성 평가
- `experiments/016/README.md`: **EXP-016 v2 성공 사례** 🏆
- `docs/retrospectives/2025-10-21.md`: EXP-016 v2 회고
- `experiments/CONCLUSION.md`: 이 문서

### 주요 코드
- `experiments/005/run_experiments.py`: H1, H2, H3 실험
- `experiments/006/run_experiments.py`: k-grid search
- `experiments/007/feature_engineering.py`: 754 features
- `experiments/007/run_experiments.py`: Feature Eng 실험
- `experiments/016/phase1_analyze_features.py`: Top 20 원본 features 선택
- `experiments/016/phase2_feature_engineering.py`: Interaction features 생성
- `experiments/016/phase3_sharpe_evaluation.py`: K 최적화 및 평가
- `submissions/submission.py`: **InferenceServer 구현** (최종 제출)

### 결과 데이터
- `experiments/005/results/`: H1~H3 결과
- `experiments/006/results/`: k 최적화 결과
- `experiments/007/results/`: Feature Eng 결과 (CV 최고)
- `experiments/016/results/`: **Interaction features 결과** (🏆 **Public 최고**)

### 총 실험 시간
- EXP-005: 6~8시간
- EXP-006: 3~4시간
- EXP-007: 4~5시간
- EXP-008~010: 3~4시간 (삭제됨)
- EXP-011: 2시간
- EXP-012: 2시간
- EXP-013: 3시간
- EXP-014: 2시간
- EXP-015: 2시간
- **EXP-016**: 2~3시간 (완전 재설계)
- **총**: 29~35시간

### 얻은 것
- ✅ **Public Score 4.440** (최고 기록!) 🏆
- ✅ InferenceServer 제약 이해 및 극복
- ✅ Interaction Features 효과 입증
- ✅ 완전 재설계 경험
- ✅ 12개 실험 완료 및 문서화
- ✅ 딥러닝 실패 원인 파악
- ✅ 체계적 실험 프로세스 확립
- ✅ 문제의 근본 이해 및 한계 인식
- ✅ CV와 Public Score 차이 경험

### 실패한 접근들 (배움)
- ❌ Classification (EXP-008)
- ❌ Autoencoder (EXP-009)
- ❌ Temporal Transformer (EXP-010)
- ❌ Direct Utility Optimization (EXP-011)
- ❌ Missing Pattern Recognition (EXP-012)
- ❌ Technical Analysis (EXP-013)
- ❌ Multi-variate LSTM (EXP-014)
- ❌ Transformer + Residual (EXP-015)

### 성공한 접근
- ✅ **EXP-016 v2**: InferenceServer + Interaction Features
  - 완전 재설계로 6.1배 향상
  - Public Score 4.440 달성
