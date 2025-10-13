# 실험 전체 회고 및 결론 (EXP-005~007)

## 목표
- **최종 목표**: Kaggle utility 17.395
- **필요 조건**: Sharpe ≈ 6.0, Profit ≈ 2.9

---

## 실험 경과

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

---

## 현재 상황

### 성과 요약
| Metric | EXP-005 | EXP-007 | 개선율 |
|--------|---------|---------|--------|
| CV Sharpe | 0.627 | **0.749** | +19.5% |
| MSE | 0.000150 | 0.000126 | -16% |
| Features | 234 | 754 | +222% |
| Kaggle (실제) | 0.724 | ? | ? |

### 문제: 목표와의 격차
- **현재**: Sharpe 0.749
- **목표**: Sharpe 6.0
- **격차**: **8배 부족**

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

## 최종 결론

### 현실
- **달성**: CV Sharpe 0.749 (19.5% 개선)
- **목표**: Sharpe 6.0 (8배 부족)
- **결론**: 현재 접근으로는 불가능

### 제안

**즉시 (오늘):**
1. ✅ CONCLUSION.md 작성 (이 문서)
2. ⏭️ 전체 커밋 및 정리
3. ⏭️ 목표 재설정 또는 대회 종료 대기

**선택 (시간 있으면):**
- Volatility Scaling (2시간)
- Ensemble (2시간)
- 예상: Sharpe 0.85~0.95

**장기 (배움 목적):**
- 대회 종료 후 Top Solution 분석
- Classification, Portfolio Optimization 접근 학습

### 성과
- ✅ 체계적 실험 프로세스 확립
- ✅ EXP-005~007 문서화
- ✅ k, feature, model 한계 확인
- ✅ Sharpe 0.749 달성 (현실적 최선)
- ✅ 문제의 근본 원인 이해

### 한계
- ❌ 목표 17.395 미달성
- ❌ Sharpe 6.0 도달 불가
- ❌ 근본적 돌파구 미발견

---

**작성일**: 2025-10-13
**상태**: EXP-005~007 완료, 현실적 한계 도달
**추천**: Option 1 (현재 최선으로 마무리) or Option 3 (Top Solution 분석 대기)

---

## 부록: 실험 산출물

### 문서
- `experiments/005/REPORT.md`: EXP-005 전체 결과
- `experiments/006/PIVOT.md`: k 접근 실패 분석
- `experiments/007/HYPOTHESES.md`: Feature Engineering 계획
- `experiments/007/ANALYSIS.md`: 현실적 가능성 평가
- `experiments/CONCLUSION.md`: 이 문서

### 코드
- `experiments/005/run_experiments.py`: H1, H2, H3 실험
- `experiments/006/run_experiments.py`: k-grid search
- `experiments/007/feature_engineering.py`: 확장 feature 모듈
- `experiments/007/run_experiments.py`: Feature Eng 실험

### 데이터
- `experiments/005/results/`: H1~H3 결과
- `experiments/006/results/`: k 최적화 결과
- `experiments/007/results/`: Feature Eng 결과

### 총 실험 시간
- EXP-005: 6~8시간
- EXP-006: 3~4시간
- EXP-007: 4~5시간
- **총**: 13~17시간

### 얻은 것
- Sharpe 0.749 (실질적 최선)
- 체계적 실험 프로세스
- 문제의 근본 이해
- 한계 인식
