# EXP-012: BREAKTHROUGH - Missing Pattern Recognition

## 발견 (Breakthrough!)

### 데이터 구조 분석

```
Feature Groups & Missing Rates:
- D (Discrete?): 0% null ← ALWAYS available
- E (Economic?): 15.3% null, starts at date 1784
- I (Industry?): 11.2% null
- M (Market?): 25.5% null
- P (Price?): 12.7% null
- S (Sentiment?): 20.2% null
- V (Volatility?): 19.8% null
```

**핵심 발견:**
1. Features가 **시간에 따라 점진적으로 추가됨**
2. E features는 date_id 1784 (전체의 20%)부터 시작
3. Missing pattern 자체가 **중요한 signal**

## 제가 놓친 것들

### 1. Missing Pattern이 Signal

**현재 접근 (틀림):**
```python
X = X.fillna(X.median())  # ❌ 정보 손실!
```

**올바른 접근:**
```python
# Missing indicator features
for col in features:
    X[f'{col}_missing'] = X[col].isnull().astype(int)

# Group-level missing counts
X['E_missing_count'] = X[[c for c in X if c.startswith('E')]].isnull().sum(axis=1)
X['M_missing_count'] = X[[c for c in X if c.startswith('M')]].isnull().sum(axis=1)
```

### 2. 시간 Period가 다름

**Period 1 (date 0~1783): Early period**
- E, M, S, V features 없음
- D, I, P features만
- 예측이 어려움

**Period 2 (date 1784+): Late period**
- 모든 features 있음
- 더 나은 예측 가능
- **여기서 수익 집중**

**전략:**
```python
# Period별 다른 모델
if date_id < 1784:
    model = early_period_model  # Conservative
else:
    model = late_period_model   # Aggressive
```

### 3. Feature Group의 의미

**D features (항상 있음):**
- 날짜/시간 정보?
- Categorical (0, 1, -1)
- 매우 stable

**E features (나중에 추가):**
- Economic indicators
- 중요도가 높을 가능성
- **17점 달성의 핵심?**

### 4. Sharpe 6.0 달성 방법

**기존 실수:**
```python
position = clip(1 + pred * k, 0, 2)
# k=600으로 고정 → Sharpe 0.7 수준
```

**올바른 방법:**
```python
# Sharpe 6.0 → 변동성 매우 낮게 유지
# 즉, 매우 확실한 날짜만 포지션

if confidence > 0.95 and pred > 0:
    position = 2.0  # All-in
elif confidence > 0.95 and pred < 0:
    position = 0.0  # All-out
else:
    position = 1.0  # Neutral (risk-free)

# Sharpe = mean / std
# std를 낮추려면 → 대부분 neutral, 확실한 날만 포지션
```

## 새로운 전략

### Strategy 1: Period-Aware Model

```python
# 1. Identify period
early_mask = df['date_id'] < 1784
late_mask = df['date_id'] >= 1784

# 2. Different models
model_early = train_on(df[early_mask], features=['D*', 'I*', 'P*'])
model_late = train_on(df[late_mask], features=all_features)

# 3. Different strategies
if early_period:
    position = 1.0  # Stay neutral (can't predict well)
else:
    position = aggressive_position(model_late.predict())
```

### Strategy 2: Confidence-Based Position

```python
# 1. Predict with confidence
pred_mean = model.predict(X)
pred_std = estimate_uncertainty(X)  # Bootstrap, ensemble, etc

confidence = 1 - (pred_std / abs(pred_mean))

# 2. High Sharpe strategy
if confidence > threshold:  # Only trade when confident
    position = 2.0 if pred_mean > 0 else 0.0
else:
    position = 1.0  # Stay neutral

# This gives:
# - Low volatility (few trades)
# - High Sharpe (only trade when sure)
# - High profit (all-in when trading)
```

### Strategy 3: Missing Pattern Features

```python
# 1. Create missing indicators
for col in features:
    df[f'{col}_is_missing'] = df[col].isnull()

# 2. Group-level missing
df['n_missing'] = df[features].isnull().sum(axis=1)
df['pct_missing'] = df['n_missing'] / len(features)

# 3. Feature group availability
for group in ['E', 'M', 'P', 'S', 'V']:
    group_cols = [c for c in features if c.startswith(group)]
    df[f'{group}_available'] = df[group_cols].notna().any(axis=1)

# 4. Model with these features
# Missing pattern itself is predictive!
```

### Strategy 4: Ensemble with Regime Detection

```python
# 1. Detect regime by data availability
regime = detect_regime(
    n_features_available,
    date_id,
    recent_volatility
)

# 2. Regime-specific models
if regime == 'early_limited_data':
    return neutral_position()
elif regime == 'late_full_data':
    return aggressive_position()
elif regime == 'high_volatility':
    return conservative_position()
```

## Expected Results

### Conservative (보수적)
- Sharpe: 2.0~3.0 (Period-aware만으로)
- Utility: 5~10

### Realistic (현실적)
- Sharpe: 3.0~4.5 (Confidence-based position)
- Utility: 10~15

### Optimistic (낙관적)
- Sharpe: 5.0~6.0 (Perfect execution)
- Utility: 15~20+ ✅ **목표 달성!**

## 구현 우선순위

### Phase 1: Missing Pattern Features (1시간) ⭐⭐⭐⭐⭐
- Missing indicators
- Period detection
- Group-level features

### Phase 2: Period-Aware Model (1시간) ⭐⭐⭐⭐⭐
- Early vs Late period
- Different models
- Conservative early, aggressive late

### Phase 3: Confidence-Based Position (1시간) ⭐⭐⭐⭐
- Uncertainty estimation (ensemble)
- High-confidence only trading
- Target Sharpe 6.0

### Phase 4: Full Integration (30분) ⭐⭐⭐
- All strategies combined
- Final tuning

## This is it!

**Missing pattern recognition이 17점 달성의 핵심입니다!**

시작합니다!
