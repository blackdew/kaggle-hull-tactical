# EXP-005: Fundamental Strategy Pivot

## 현실 직시: k 조정으로는 17점 불가능

### 증거
- **우리 최고**: CV Sharpe 0.836 → Kaggle 0.15~0.44
- **상위권**: 17.333 (우리의 **38~115배**)
- **k=500 실패**: CV에서 최고였지만 실제로는 최악

### 근본 원인
1. **모델 예측력 부족**: Lasso의 excess return 예측 상관계수 0.03~0.06 수준
2. **k는 증폭기일 뿐**: 약한 신호 × 500 = 큰 노이즈
3. **선형 모델의 한계**: 시장은 비선형, 시변적, 복잡계

**결론**: 현재 Lasso + Top-20 + k 조정 접근은 한계 명확. **전략 전환 필요.**

---

## 🎯 새로운 접근 방향

### Option 1: 시계열 모델 (Regime-Aware) ⭐⭐⭐⭐⭐

**핵심 아이디어**: 시장은 regime이 바뀜 (bull/bear/sideways). 단일 모델이 아닌 regime별 모델

**구현**:
```python
# 1. Regime 탐지 (HMM, Clustering, Rule-based)
regime = detect_regime(features)  # 0=bear, 1=neutral, 2=bull

# 2. Regime별 모델
if regime == 0:  # Bear market
    position = 0.0~0.8  # 방어적
elif regime == 1:  # Neutral
    position = 0.8~1.2  # 중립
else:  # Bull
    position = 1.2~2.0  # 공격적

# 3. Ensemble
models = {
    'bear': train_model(data[bear_periods]),
    'neutral': train_model(data[neutral_periods]),
    'bull': train_model(data[bull_periods])
}
```

**Feature candidates for regime detection**:
- V13, V10, V9 (volatility features) - 상위 상관
- M4, M1, M2 (macro features)
- D1~D9 (date/calendar features) - seasonality

**장점**:
- 분포 이동 문제 해결 (regime별 모델)
- 비선형성 포착
- 상위권이 사용할 가능성 높음

**단점**:
- 복잡도 증가
- Regime 오분류 리스크

### Option 2: Gradient Boosting (XGBoost/LightGBM) ⭐⭐⭐⭐

**왜 Lasso가 아닌가**:
- Lasso: 선형, feature interaction 포착 못함
- XGBoost: 비선형, feature interaction 자동 학습

**구현**:
```python
import xgboost as xgb

# 전체 98개 feature 사용 (Lasso는 Top-20만)
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror'
)

# TimeSeriesSplit CV
# Target: market_forward_excess_returns (not position)
model.fit(X_train, y_train)

# Feature importance 분석
# Top features가 Top-20과 다를 가능성 (interaction 고려)
```

**장점**:
- Feature interaction 자동 학습
- 비선형 관계 포착
- Kaggle competition에서 검증된 강력함

**단점**:
- Overfitting 리스크 (TimeSeriesSplit으로 방어)
- 해석성 낮음

### Option 3: Ensemble (Multiple Models + Multiple k) ⭐⭐⭐

**아이디어**: 서로 다른 모델들의 예측을 결합

```python
models = {
    'lasso_top20': Lasso(alpha=1e-4) on top-20,
    'lasso_top50': Lasso(alpha=1e-4) on top-50,
    'ridge': Ridge(alpha=1.0) on all features,
    'xgb': XGBRegressor(...),
    'lgbm': LGBMRegressor(...)
}

# 각 모델의 예측을 앙상블
excess_pred = weighted_average([
    models['lasso_top20'].predict(X) * 0.2,
    models['xgb'].predict(X) * 0.4,
    models['lgbm'].predict(X) * 0.4
])

# k 값도 동적 조정 (confidence-weighted)
k_effective = base_k * confidence_score(excess_pred)
```

**장점**:
- 모델 다양성으로 robustness 증가
- 단일 모델 실패 리스크 분산

**단점**:
- 복잡도 매우 높음
- 과적합 위험

### Option 4: Feature Engineering 대공사 ⭐⭐⭐⭐

**현재 문제**: Top-20 features는 단순 correlation 기반
- M4, V13, M1 등 개별 feature만 사용
- Feature interaction 미고려

**새 Feature 생성**:
```python
# 1. Interaction features
X['M4_x_V13'] = X['M4'] * X['V13']
X['M1_x_S5'] = X['M1'] * X['S5']

# 2. Lag features (시계열 패턴)
for col in ['M4', 'V13', 'M1']:
    X[f'{col}_lag1'] = X[col].shift(1)
    X[f'{col}_lag5'] = X[col].shift(5)
    X[f'{col}_rolling_mean_10'] = X[col].rolling(10).mean()
    X[f'{col}_rolling_std_10'] = X[col].rolling(10).std()

# 3. Ratio/Diff features
X['M4_M1_ratio'] = X['M4'] / (X['M1'] + 1e-6)
X['V13_diff'] = X['V13'].diff()

# 4. Regime indicators
X['high_vol'] = (X['V13'] > X['V13'].rolling(50).quantile(0.8)).astype(int)
X['low_vol'] = (X['V13'] < X['V13'].rolling(50).quantile(0.2)).astype(int)

# 5. Calendar features (D1~D9 활용)
# D4=1 (월요일), D5=1 (금요일) 등 → calendar effect
X['is_monday'] = X['D4']
X['is_friday'] = X['D5']
```

**장점**:
- Lasso/Ridge도 성능 향상 가능
- XGBoost와 조합 시 시너지

**단점**:
- Feature 수 폭발 (curse of dimensionality)
- Overfitting 위험

### Option 5: Direct Position Classification ⭐⭐

**아이디어**: Regression 대신 Classification
- Class 0: position = 0~0.5 (defensive)
- Class 1: position = 0.5~1.0 (neutral-defensive)
- Class 2: position = 1.0~1.5 (neutral-aggressive)
- Class 3: position = 1.5~2.0 (aggressive)

```python
# Target 변환
y_class = pd.cut(
    positions_train,  # [0, 2]
    bins=[0, 0.5, 1.0, 1.5, 2.0],
    labels=[0, 1, 2, 3]
)

# Multi-class classifier
clf = xgb.XGBClassifier(...)
clf.fit(X_train, y_class)

# Prediction
class_pred = clf.predict(X_test)
position = class_to_position_mapping[class_pred]
```

**장점**:
- 극단 포지션(0, 2)을 명시적으로 학습
- Softmax probability로 confidence 측정 가능

**단점**:
- H1a 실험에서 실패했음 (Sharpe 0.631~0.671)
- Continuous target을 discretize하면 정보 손실

---

## 🔬 추천 실험 계획 (EXP-005)

### Phase 1: Quick Wins (1~2일)

**H1: XGBoost Baseline**
- 전체 98 features 사용
- XGBRegressor(n_estimators=500, max_depth=6)
- Target: market_forward_excess_returns
- k=50, 100, 200 테스트
- **예상**: Lasso 대비 10~20% 성능 향상

**H2: Feature Engineering + Lasso**
- Lag features (1, 5, 10일)
- Rolling statistics (mean, std, 10일)
- Interaction features (M4×V13, M1×S5 등)
- Lasso Top-50 features
- **예상**: Lasso 대비 5~15% 향상

### Phase 2: Advanced (3~5일)

**H3: Regime-Based Ensemble**
- Volatility regime 탐지 (V13 기준)
- High/Medium/Low vol 별 모델
- k 값도 regime별 조정
- **예상**: 분포 이동 문제 해결, 20~40% 향상

**H4: XGBoost + LightGBM Ensemble**
- XGBoost, LightGBM, CatBoost 앙상블
- Weighted average (Sharpe 기반 가중치)
- **예상**: 단일 모델 대비 5~10% 향상

### Phase 3: Moonshot (1주)

**H5: Deep Learning (LSTM/Transformer)**
- 시계열 특화 모델
- Attention mechanism으로 중요 시점 학습
- **리스크**: 데이터 부족(8990행), Overfitting 위험 높음
- **예상**: 성공 시 30~50% 향상, 실패 가능성 50%

---

## 📊 데이터 분석 재검토

### 현재 사용 중인 Top-20 features (EXP-000)
```
M4 (-0.066), V13 (0.062), M1 (0.046), S5 (0.040), S2 (-0.038),
D1 (0.034), D2 (0.034), M2 (0.033), V10 (0.033), E7 (-0.032), ...
```

**문제**:
1. **상관계수가 너무 약함** (max 0.066)
2. **선형 상관만 고려** (비선형 관계 놓침)
3. **Feature interaction 무시**

### 새로운 Feature 분석 필요

**그룹별 전략**:
- **V (Volatility)**: V13, V10, V9 → Regime 탐지 핵심
- **M (Macro)**: M4, M1, M2 → 경제 환경
- **S (Sentiment)**: S5, S2 → 투자 심리
- **D (Date)**: D1~D9 → Calendar effects (요일, 월말 등)
- **P (Price)**: 가격 모멘텀 feature들
- **E (Events)**, **I (Index)**: 보조 신호

**Missing data 전략**:
- E7 (77.5%), V10 (67.3%), S3 (63.8%), M1 (61.7%) → 결측 심각
- 현재: 단순 median imputation
- 개선: MICE, Forward-fill (시계열), 또는 Missing indicator

---

## 🎯 최우선 실행 권장

**제안**: **H1 (XGBoost Baseline) + H2 (Feature Engineering)** 동시 진행

**이유**:
1. **XGBoost는 Kaggle 표준**: 거의 모든 상위권이 사용
2. **빠른 검증**: 1~2일 내 결과 확인
3. **Feature Engineering은 범용**: 모든 모델에 도움
4. **Lasso는 너무 약함**: 선형 모델로는 한계

**구체적 실행**:
```bash
# EXP-005 생성
mkdir -p experiments/005
touch experiments/005/HYPOTHESES.md
touch experiments/005/run_xgboost.py
touch experiments/005/feature_engineering.py
touch experiments/005/README.md

# H1: XGBoost baseline
# H2: Lasso + feature engineering
# H3: Regime-based XGBoost
# H4: Ensemble (XGBoost + LightGBM + Lasso)
```

---

## 💡 핵심 인사이트

### Why k adjustment failed
1. **Garbage In, Garbage Out**: Lasso 예측 상관 0.03~0.06 수준
2. **k는 확대경**: 약한 신호를 키워도 여전히 약함
3. **분포 이동**: 훈련(2020~2023) vs 테스트(2024+)는 다른 세계

### What top performers likely do
1. **Gradient Boosting**: XGBoost/LightGBM/CatBoost 앙상블
2. **Feature Engineering**: Lag, Rolling, Interaction features
3. **Regime Detection**: Volatility/Market regime별 전략
4. **Sophisticated Ensembling**: 10+ models 결합
5. **Advanced Imputation**: 결측치 처리 최적화

### Reality check
- **우리**: Lasso + Top-20 → 0.15~0.44
- **상위권**: 아마도 XGBoost + Feature Engineering + Regime → 17+
- **Gap**: **모델 자체가 다름**, k만 바꿔서는 못 따라감

---

## 다음 단계

**즉시**:
1. ✅ k=200 제출 (이미 준비됨) → 점수 확인
2. ⏭️ EXP-005 시작: XGBoost + Feature Engineering
3. ⏭️ 1~2일 내 새 접근으로 제출

**중기**:
1. Regime-based modeling
2. Ensemble (XGBoost + LightGBM)
3. Advanced feature engineering (lag, rolling, interaction)

**장기**:
- Deep Learning 시도 (LSTM/Transformer)
- Kaggle discussion/notebooks 참고 (상위권 공유 대기)

---

**작성일**: 2025-10-03
**현재 상태**: Lasso + k 조정 한계 명확, 전략 전환 필요
**목표**: 17+ 점수 달성 위해 근본적 접근 변경
