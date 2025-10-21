# EXP-016: InferenceServer-Compatible Feature Engineering

**목표**: Sharpe 1.0+ 달성 (현실적 목표: 0.8+)
**제약**: InferenceServer 호환 (1 row씩 예측)
**접근**: 원본 features + interaction features (lag/rolling 제외)

---

## 🎯 핵심 전략

### 문제 인식
- 기존 EXP-007: lag/rolling features 사용 → InferenceServer 불가
- Kaggle Code Competition: row-by-row 예측 필요
- **1 row에서 계산 가능한 features만 사용**

### 해결책
1. **원본 features 선택**
   - lag/rolling/ema 제외
   - 즉시 계산 가능한 features만

2. **Feature Engineering**
   - Interaction features (A × B, A / B)
   - Polynomial features (A², A³)
   - Ratio features (A / (B + ε))

3. **Model**
   - XGBoost Regressor
   - Hyperparameter tuning
   - K parameter optimization

---

## 📋 실험 계획

### Phase 1: 원본 Features 분석
- [ ] 1.1: 전체 features 목록 확인
- [ ] 1.2: 1 row 계산 가능 features 필터링
- [ ] 1.3: Feature importance 분석
- [ ] 1.4: Top N features 선택

### Phase 2: Feature Engineering
- [ ] 2.1: Interaction features 생성 (곱셈, 나눗셈)
- [ ] 2.2: Polynomial features 생성
- [ ] 2.3: Feature selection (중요도 기반)

### Phase 3: Model Training
- [ ] 3.1: Baseline model (원본 features만)
- [ ] 3.2: Engineered features 추가
- [ ] 3.3: Hyperparameter tuning
- [ ] 3.4: Cross-validation (5-fold)

### Phase 4: InferenceServer 구현
- [ ] 4.1: Server 코드 작성
- [ ] 4.2: 로컬 테스트
- [ ] 4.3: Kaggle 제출

---

## 🚫 제약사항

### 사용 불가 Features
- `*_lag*`: lag features (과거 데이터 필요)
- `*_rolling_*`: rolling window features (여러 row 필요)
- `*_ema_*`: exponential moving average (과거 데이터 필요)

### 사용 가능 Features
- 원본 features: `M4`, `V13`, `P7`, `E19` 등
- Interaction: `M4 * V13`, `M4 / (V13 + 1e-8)`
- Polynomial: `M4²`, `V13³`

---

## 📊 성공 기준

- **Minimum**: Sharpe 0.75+ (EXP-007 수준)
- **Target**: Sharpe 0.85+
- **Stretch**: Sharpe 1.0+
- **필수**: InferenceServer 정상 작동
- **필수**: Kaggle 제출 성공

---

## 📝 진행 상황

- [x] 기존 EXP-016 백업
- [ ] Phase 1: 원본 Features 분석
- [ ] Phase 2: Feature Engineering
- [ ] Phase 3: Model Training
- [ ] Phase 4: InferenceServer 구현
