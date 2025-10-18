# EXP-016 가설 및 실험 설계

## 배경

### 이전 실험 결과
- **EXP-007 최고 성능**: Sharpe 0.749 (XGBoost + 754 features)
- **목표**: Sharpe 1.0+ (현재 대비 +34%)
- **총 10개 실험 완료** (EXP-005~015)

### 핵심 발견
1. **XGBoost가 압도적 1위**
   - LSTM: 0.471 (XGBoost의 63%)
   - Transformer: 0.257 (XGBoost의 34%)
   - 딥러닝은 데이터 부족으로 실패

2. **Feature Engineering이 핵심**
   - 234 features (EXP-005): Sharpe 0.627
   - 754 features (EXP-007): Sharpe 0.749 (+19%)
   - **Feature 수 증가가 가장 효과적**

3. **이전 실패의 원인**
   - 얕은 시도 (각 실험 2~3시간)
   - 754개에서 멈춤 (더 확장 시도 안함)
   - Feature importance 분석 안함
   - Hyperparameter tuning 부족
   - Ensemble 시도 안함

---

## 문제 정의

### 왜 0.749에서 멈췄는가?

**가설 1: Feature가 충분하지 않다** (확률 60%)
- EXP-007에서 754 features 생성
- 하지만 **진짜 효과적인 features인지 검증 안함**
- Interaction, polynomial features 시도 안함
- Domain knowledge 기반 features 부족

**가설 2: Feature에 noise가 많다** (확률 30%)
- 754개 중 일부만 진짜 유의미
- 불필요한 features가 과적합 유발
- Feature selection으로 성능 향상 가능

**가설 3: Hyperparameter가 최적이 아니다** (확률 40%)
- XGBoost default parameters 거의 그대로 사용
- Grid search / Bayesian optimization 안함
- 새 feature set에 맞는 재튜닝 필요

**가설 4: Single model의 한계** (확률 20%)
- XGBoost 단일 모델 사용
- Ensemble로 분산 감소 가능
- LightGBM, CatBoost 추가로 성능 향상

---

## 핵심 가설

### H1: 754 features 중 일부만 진짜 유의미하다

**근거:**
- Feature importance 분석 안함
- Null importance test 안함
- 과적합 가능성

**검증 방법:**
1. SHAP analysis
2. Permutation importance
3. Null importance test (target shuffle)
4. Top 100/50/20만으로 재실험

**예상 결과:**
- 754 → 300~500 (유의미한 features)
- Sharpe 유지 or 약간 향상
- 과적합 감소

---

### H2: Interaction features로 성능 향상 가능

**근거:**
- XGBoost는 feature interaction 자동 학습
- 하지만 명시적 interaction features가 도움될 수 있음
- 특히 Top features 간 interaction

**검증 방법:**
1. Top 20 features × Top 20 = 400 interactions
   - Multiply, Divide, Difference, Ratio
2. Baseline + Interaction 실험
3. Feature importance 재분석

**예상 결과:**
- Sharpe +3~8% (0.749 → 0.77~0.81)
- 일부 interaction features가 Top 100 진입

---

### H3: Polynomial features로 비선형성 강화

**근거:**
- XGBoost는 piecewise linear
- Polynomial features로 복잡한 비선형 관계 포착 가능

**검증 방법:**
1. Top 10 features: 제곱, 세제곱
2. Top 20 features: cross terms (degree=2)
3. Baseline + Polynomial 실험

**예상 결과:**
- Sharpe +2~5% (0.749 → 0.76~0.79)
- 과적합 위험 있음 (주의 필요)

---

### H4: Domain-specific features로 추가 정보 획득

**근거:**
- 현재 features는 대부분 통계적 (lag, rolling, cross-sectional)
- Market microstructure, order flow 관련 features 부족
- Domain knowledge 활용 부족

**검증 방법:**
1. Market microstructure features
   - Order imbalance proxies
   - Spread proxies
2. Volatility clustering features
3. Return autocorrelation features
4. Time-based features

**예상 결과:**
- Sharpe +3~7% (0.749 → 0.77~0.80)
- 특정 feature group이 특히 효과적일 가능성

---

### H5: Feature selection으로 과적합 감소

**근거:**
- 754 features (현재) or 1500+ features (확장 후)
- 모든 features가 유의미하지 않음
- Redundant features 제거 필요

**검증 방법:**
1. Recursive Feature Elimination (RFE)
2. L1 regularization (Lasso)
3. Correlation-based selection
4. 최적 subset으로 재실험

**예상 결과:**
- Feature 수: 1500+ → 300~800
- Sharpe 유지 or 향상
- Train-CV gap 감소 (과적합 감소)

---

### H6: Hyperparameter tuning으로 추가 개선

**근거:**
- 현재 XGBoost parameters 거의 default
- 새 feature set에 맞는 최적화 필요
- Optuna로 체계적 탐색

**검증 방법:**
1. Optuna Bayesian optimization
2. 200+ trials
3. max_depth, learning_rate, subsample 등 전체 탐색

**예상 결과:**
- Sharpe +3~10% (feature 개선 후)
- 최적 parameters 확정

---

### H7: Ensemble로 최종 boost (조건부)

**근거:**
- 현재 XGBoost만 사용
- LightGBM, CatBoost 추가 가능
- Ensemble로 분산 감소

**검증 방법:**
1. LightGBM, CatBoost 각각 최적화
2. Simple average, Weighted average
3. Stacking (meta-learner)

**예상 결과:**
- Sharpe +5~10% (single model 대비)
- 최종 Sharpe 1.0+ 달성 가능성

---

## 실험 전략

### Phase 1: Feature 품질 검증 (H1)
**목표: 754 features 중 진짜 효과적인 것만 선택**

1. SHAP, Permutation, Null importance 분석
2. Top 100/50/20 재실험
3. Feature group 분석 (D/E/I/M/P/S/V)
4. 최적 baseline 확립

**예상 Sharpe: 0.75~0.80** (유지 or 약간 향상)

---

### Phase 2: Feature 확장 (H2, H3, H4)
**목표: 1500+ features로 확장, 한계까지**

1. Interaction features (400+)
2. Polynomial features (100+)
3. Domain-specific features (100+)
4. Progressive testing

**예상 Sharpe: 0.80~0.90** (cumulative effect)

---

### Phase 3: Feature Selection & Tuning (H5, H6)
**목표: 최적화로 Sharpe 1.0+ 달성**

1. Feature selection (1500+ → 300~800)
2. Hyperparameter tuning (Optuna 200+ trials)
3. 최종 검증

**예상 Sharpe: 0.90~1.05** (목표 달성)

---

### Phase 4: Ensemble (H7, 조건부)
**조건: Phase 3에서 Sharpe < 1.0인 경우만**

1. LightGBM, CatBoost 추가
2. Ensemble 전략 (5~10가지)
3. 최종 boost

**예상 Sharpe: 0.95~1.10** (최종)

---

## 성공 시나리오

### 낙관적 (20% 확률)
- Phase 2에서 Sharpe 0.90 달성
- Phase 3에서 Sharpe 1.05+ 달성
- Phase 4 불필요
- **최종 Sharpe: 1.05~1.15**

### 현실적 (50% 확률)
- Phase 2에서 Sharpe 0.85 달성
- Phase 3에서 Sharpe 0.95 달성
- Phase 4 필수
- **최종 Sharpe: 1.00~1.08**

### 비관적 (30% 확률)
- Phase 2에서 Sharpe 0.80 달성
- Phase 3에서 Sharpe 0.90 달성
- Phase 4에서 Sharpe 0.95~1.00
- **최종 Sharpe: 0.95~1.00** (목표 미달)

---

## 실패 시나리오 및 대응

### Scenario A: Phase 2에서 Sharpe 향상 없음 (0.75 유지)
**가능성**: Feature 추가가 도움 안됨, noise만 증가

**대응:**
1. Feature selection 더 aggressive하게
2. Domain features 재검토
3. 다른 접근 (target re-engineering) 고려

---

### Scenario B: Phase 3에서 과적합 발생
**가능성**: 1500+ features로 train-CV gap 증가

**대응:**
1. Regularization 강화 (reg_alpha, reg_lambda ↑)
2. Feature 수 감소 (500 이하)
3. Cross-validation fold 수 증가 (5-fold)

---

### Scenario C: 최종 Sharpe < 1.0
**가능성**: 0.749가 진짜 한계

**대응:**
1. 회고 작성 (진짜 한계 인정)
2. 다른 대회로 이동
3. Top solution 분석 (대회 종료 후)

---

## 측정 지표

### Primary
- **3-fold CV Sharpe** (목표 > 1.0)

### Secondary
- MSE (낮을수록 좋음)
- Train-CV Sharpe gap (< 0.15)
- Feature importance distribution
- Position distribution (mean, std)

---

## 실험 원칙

### 1. 깊게 파기
- 각 Phase 완료할 때까지 진행
- 1~2번 시도로 포기 금지
- 한계까지 밀어붙이기

### 2. 포기하지 않기
- Sharpe 1.0 달성 or 진짜 한계 확인할 때까지
- "예상 효과 +10%"가 아니라 실제 측정
- 변명 만들지 않기

### 3. 체계적 문서화
- 모든 실험 결과 저장
- CHECKLIST.md 실시간 업데이트
- 재현 가능하도록 코드 정리

### 4. 과적합 경계
- Train-CV gap 지속적 모니터링
- Feature 수 vs 성능 curve 확인
- Regularization 적절히 사용

---

## 예상 타임라인

### Week 1
- Day 1-3: Phase 1 (Feature 분석)
- Day 4-7: Phase 2 시작 (Interaction, Polynomial)

### Week 2
- Day 8-10: Phase 2 완료 (Domain features, Testing)
- Day 11-14: Phase 3 (Selection, Tuning)

### Week 3 (조건부)
- Day 15-17: Phase 4 (Ensemble)
- Day 18-21: 최종 검증, REPORT 작성

**총 예상 시간: 2~3주**

---

## 최종 목표

### Minimum Success
- ✅ CV Sharpe > 0.90
- ✅ Feature 품질 검증 완료
- ✅ 체계적 실험 프로세스

### Target Success
- ✅ CV Sharpe > 1.0
- ✅ Kaggle utility > 2.0
- ✅ 재현 가능한 코드 & 문서

### Stretch Success
- ✅ CV Sharpe > 1.10
- ✅ Kaggle utility > 3.0
- ✅ 진짜 한계 확인 or 돌파

---

**작성일**: 2025-10-18
**상태**: Phase 1 시작 전
**현재 baseline**: Sharpe 0.749 (EXP-007)
**목표**: Sharpe 1.0+
