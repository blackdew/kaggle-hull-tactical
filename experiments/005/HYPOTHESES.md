# EXP-005 가설 및 실험 계획

## 배경

**EXP-004 실패 분석**:
- Lasso + Top-20 + k 조정 접근의 한계 명확
- CV Sharpe 0.836 (k=500) → Kaggle 0.150 (실패)
- k=50 (0.441) vs 상위권 17.333 → **38배 차이**

**근본 원인**:
1. **모델 예측력 부족**: Lasso의 excess return 예측 상관계수 0.03~0.06 수준
2. **선형 모델의 한계**: Feature interaction, 비선형 관계 포착 못함
3. **분포 이동**: 훈련 데이터(2020~2023) vs 테스트(2024+) 환경 차이

**결론**: k 파라미터 조정이 아닌 **모델 자체를 바꿔야 함**

---

## 실험 목표

**Primary**: XGBoost/LightGBM 등 Gradient Boosting으로 예측력 향상
- Lasso 대비 10~30% 성능 개선 목표
- Feature interaction 자동 학습
- 비선형 관계 포착

**Secondary**: Feature Engineering으로 신호 강화
- Lag features (시계열 패턴)
- Rolling statistics (추세/변동성)
- Interaction features (M4×V13 등)

**Ultimate**: Kaggle 점수 5+ 달성 (현재 0.44 대비 10배+)

---

## 가설 (Hypotheses)

### H1: XGBoost Baseline (기본 성능 확인)

**가설**: XGBoost가 Lasso보다 excess return 예측력이 높다

**설정**:
- 모델: XGBRegressor
- Features: 전체 98개 (exclude: date_id, targets, lagged 등)
- Target: market_forward_excess_returns
- Hyperparameters:
  ```python
  n_estimators=500
  learning_rate=0.01
  max_depth=6
  subsample=0.8
  colsample_bytree=0.8
  ```
- k 값: 50, 100, 200 테스트

**예상 결과**:
- CV Sharpe: 0.7~0.9 (Lasso 0.604 대비 +15~50%)
- Feature importance에서 Top-20과 다른 패턴 발견
- Interaction features 자동 학습 효과

**성공 기준**:
- CV Sharpe > 0.7 (Lasso 대비 +15%)
- k=100~200에서 안정적 성능

### H2: LightGBM Baseline (비교)

**가설**: LightGBM이 XGBoost와 유사하거나 더 나은 성능

**설정**:
- 모델: LGBMRegressor
- Features: 전체 98개
- Hyperparameters:
  ```python
  n_estimators=500
  learning_rate=0.01
  num_leaves=31
  subsample=0.8
  colsample_bytree=0.8
  ```
- k 값: 50, 100, 200

**예상 결과**:
- XGBoost와 유사한 성능 (Sharpe ±0.05 범위)
- 훈련 속도 더 빠름
- Feature importance 패턴 비교

**성공 기준**:
- CV Sharpe > 0.7
- XGBoost 대비 성능 차이 < 5%

### H3: Feature Engineering (Lag + Rolling)

**가설**: 시계열 feature 추가로 예측력 향상

**새 Features**:
```python
# 1. Lag features (과거 값)
for col in top_features:  # M4, V13, M1, S5, etc.
    df[f'{col}_lag1'] = df[col].shift(1)
    df[f'{col}_lag5'] = df[col].shift(5)
    df[f'{col}_lag10'] = df[col].shift(10)

# 2. Rolling statistics (추세/변동성)
for col in top_features:
    df[f'{col}_rolling_mean_5'] = df[col].rolling(5).mean()
    df[f'{col}_rolling_mean_10'] = df[col].rolling(10).mean()
    df[f'{col}_rolling_std_5'] = df[col].rolling(5).std()
    df[f'{col}_rolling_std_10'] = df[col].rolling(10).std()

# 3. Difference (변화량)
for col in top_features:
    df[f'{col}_diff1'] = df[col].diff(1)
```

**모델**: XGBoost + 추가 features

**예상 결과**:
- CV Sharpe: 0.75~0.95 (H1 대비 +5~10%)
- 시계열 패턴 포착으로 안정성 증가

**성공 기준**:
- CV Sharpe > H1 결과 + 0.05
- Feature importance에서 lag/rolling features 상위 진입

### H4: Interaction Features

**가설**: Feature 간 곱셈/나눗셈 상호작용이 예측력 향상

**새 Features**:
```python
# Top features 간 interaction
df['M4_x_V13'] = df['M4'] * df['V13']
df['M1_x_S5'] = df['M1'] * df['S5']
df['V13_x_V10'] = df['V13'] * df['V10']
df['M4_div_M1'] = df['M4'] / (df['M1'].abs() + 1e-6)

# Volatility regime indicators
df['high_vol'] = (df['V13'] > df['V13'].rolling(20).quantile(0.8)).astype(int)
df['low_vol'] = (df['V13'] < df['V13'].rolling(20).quantile(0.2)).astype(int)
```

**모델**: XGBoost + H3 features + interaction features

**예상 결과**:
- CV Sharpe: 0.8~1.0 (H3 대비 +5%)
- Regime indicator가 중요 feature로 부상

**성공 기준**:
- CV Sharpe > H3 결과 + 0.03
- Interaction features가 상위 10 feature에 진입

### H5: Ensemble (XGBoost + LightGBM + Lasso)

**가설**: 여러 모델 앙상블이 단일 모델보다 안정적

**설정**:
```python
# 각 모델 예측
pred_xgb = xgb_model.predict(X)
pred_lgbm = lgbm_model.predict(X)
pred_lasso = lasso_model.predict(X)

# Weighted average (CV Sharpe 기반 가중치)
weights = {
    'xgb': 0.5,     # 최고 성능 가정
    'lgbm': 0.3,
    'lasso': 0.2    # 다양성 확보
}
pred_ensemble = (
    weights['xgb'] * pred_xgb +
    weights['lgbm'] * pred_lgbm +
    weights['lasso'] * pred_lasso
)
```

**예상 결과**:
- CV Sharpe: H4와 유사하거나 약간 높음
- Sharpe Std 감소 (안정성 증가)

**성공 기준**:
- CV Sharpe ≥ max(H1, H2, H3, H4)
- Sharpe Std < min(H1, H2, H3, H4) - 0.05

### H6: Regime-Based Model (Advanced)

**가설**: Volatility regime별 모델이 분포 이동 문제 해결

**Regime 정의**:
```python
# V13 (volatility feature) 기준
vol_20q_low = df['V13'].rolling(20).quantile(0.33)
vol_20q_high = df['V13'].rolling(20).quantile(0.67)

regime = pd.cut(
    df['V13'],
    bins=[-np.inf, vol_20q_low, vol_20q_high, np.inf],
    labels=['low_vol', 'mid_vol', 'high_vol']
)
```

**모델**: Regime별 XGBoost + k 값도 regime별 조정
```python
models = {
    'low_vol': XGBRegressor(...),   # k=200 (공격적)
    'mid_vol': XGBRegressor(...),   # k=100
    'high_vol': XGBRegressor(...)   # k=50 (방어적)
}
```

**예상 결과**:
- CV Sharpe: 0.85~1.05
- Regime 전환 시 안정성 증가
- 테스트 데이터 분포 이동 대응력 향상

**성공 기준**:
- CV Sharpe > 0.85
- Regime별 Sharpe 편차 < 0.2

---

## 실험 설계

### 데이터 분할
```python
from sklearn.model_selection import TimeSeriesSplit

# 5-fold TimeSeriesSplit (EXP-004와 동일)
tscv = TimeSeriesSplit(n_splits=5)

# Hold-out test (최신 20% 데이터)
# 분포 이동 테스트용
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
holdout_data = df[train_size:]
```

### 평가 메트릭
```python
# 1. Sharpe Ratio (주 메트릭)
sharpe = (mean(excess_returns) / std(strategy_returns)) * sqrt(252)

# 2. Volatility Ratio (제약 조건)
vol_ratio = std(strategy_returns) / std(market_returns)
# 목표: vol_ratio ≤ 1.2

# 3. Position Stats
pos_mean = mean(positions)
pos_std = std(positions)
pos_range = max(positions) - min(positions)

# 4. Sharpe Std (안정성)
sharpe_std = std([sharpe_fold1, sharpe_fold2, ...])
# 낮을수록 안정적

# 5. Feature Importance (해석성)
# XGBoost/LightGBM의 gain-based importance
```

### Missing Data 처리
```python
# 현재: 단순 median imputation
# 개선 옵션:
# 1. Forward-fill (시계열 특성 활용)
# 2. Group-wise median (feature group별)
# 3. MICE (iterative imputation)
# 4. Missing indicator (결측 여부 자체를 feature로)

# EXP-005에서는 일단 median 유지 (빠른 검증)
# 나중에 H7로 imputation 비교 실험
```

---

## 실행 계획

### Phase 1: Baseline Models (Day 1)

**오전**:
- H1: XGBoost Baseline (k=50, 100, 200)
- H2: LightGBM Baseline (k=50, 100, 200)

**오후**:
- 결과 분석
- Feature importance 비교
- Best k 값 선정

**예상 소요**: 4~6시간

### Phase 2: Feature Engineering (Day 1~2)

**Day 1 저녁**:
- H3: Lag + Rolling features 생성
- XGBoost 재훈련

**Day 2 오전**:
- H4: Interaction features 추가
- XGBoost 재훈련
- Feature importance 분석

**예상 소요**: 4~6시간

### Phase 3: Advanced (Day 2~3)

**Day 2 오후**:
- H5: Ensemble (XGBoost + LightGBM + Lasso)

**Day 3 (Optional)**:
- H6: Regime-Based Model
- Hold-out test 검증

**예상 소요**: 6~8시간

---

## 성공 기준

### Minimum Viable (최소 성공)
- CV Sharpe > 0.7 (Lasso 0.604 대비 +15%)
- Kaggle 제출 점수 > 1.0 (현재 0.44 대비 2배+)

### Target (목표)
- CV Sharpe > 0.85 (+40%)
- Kaggle 제출 점수 > 3.0 (7배+)

### Stretch (도전)
- CV Sharpe > 1.0 (+65%)
- Kaggle 제출 점수 > 5.0 (11배+)
- 상위 50% 진입

---

## 리스크 및 대응

### Risk 1: Overfitting
- **증상**: CV 높은데 Kaggle 낮음 (EXP-004와 동일)
- **대응**:
  - Hold-out test 도입
  - Early stopping (validation loss 기준)
  - Max depth 제한 (4~6)
  - Subsample 0.8 이하

### Risk 2: Feature 폭발
- **증상**: Lag + Rolling → Feature 수 300~500개
- **대응**:
  - Top-20 features만 lag/rolling 적용
  - Feature importance 기반 필터링
  - L1 regularization (XGBoost reg_alpha)

### Risk 3: 시간 부족
- **증상**: 실험 너무 많음
- **대응**:
  - H1, H2, H3만 필수 실행
  - H4, H5, H6는 선택적
  - 병렬 실행 (H1, H2 동시)

### Risk 4: 여전히 점수 낮음
- **증상**: XGBoost도 Kaggle 1점 미만
- **대응**:
  - Kaggle discussion/notebooks 확인
  - 상위권 접근 참고
  - EXP-006에서 Deep Learning 시도

---

## 산출물

### 코드
- `experiments/005/run_experiments.py` - 메인 실험 스크립트
- `experiments/005/feature_engineering.py` - Feature 생성 함수
- `experiments/005/models.py` - 모델 정의 (XGBoost, LightGBM, Ensemble)

### 결과
- `experiments/005/results/h1_xgboost_folds.csv` - H1 폴드별 결과
- `experiments/005/results/h2_lightgbm_folds.csv` - H2 폴드별 결과
- `experiments/005/results/h3_feature_eng_folds.csv` - H3 결과
- `experiments/005/results/summary.csv` - 전체 요약
- `experiments/005/results/feature_importance.csv` - Feature importance

### 문서
- `experiments/005/HYPOTHESES.md` - 가설 및 계획 (이 파일)
- `experiments/005/REPORT.md` - 실험 결과 리포트
- `experiments/005/README.md` - 실행 가이드

### 제출 파일
- `experiments/005/submissions/best_model_submission.csv`
- `kaggle_kernel/kaggle_inference_xgboost.py` - Kaggle 제출용

---

## 참고

- EXP-000: Feature 분석 → Top-20 features, 결측 현황
- EXP-002: Lasso 실험 → Sharpe 0.604 baseline
- EXP-004: k 조정 실험 → Sharpe 0.836 (CV), 0.150 (Kaggle 실패)
- **EXP-005: 모델 전환** → XGBoost/LightGBM + Feature Engineering

**핵심 차이**: Lasso(선형) → Gradient Boosting(비선형, interaction 학습)
