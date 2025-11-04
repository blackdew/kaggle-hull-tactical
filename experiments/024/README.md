# EXP-024: Data-Driven Feature Engineering

**목표**: 데이터 분석 기반 고품질 feature 생성으로 Sharpe 0.6 → 1.0+ 달성

**핵심 전략**: Mutual Information, Nonlinear correlation 분석 + InferenceServer 호환 feature engineering

---

## 연구 배경 (Web Search 결과)

### 1. Feature Engineering Best Practices (2024)
**출처**: ResearchGate, Medium, Analytics Vidhya

**핵심 원칙**:
- **Lag features**: 과거 값 활용 (❌ InferenceServer 불가)
- **Rolling window**: 이동 평균, 표준편차 (❌ InferenceServer 불가)
- **DateTime features**: 요일, 월, 주기성 (✅ 가능!)
- **Volatility indexes**: 변동성 측정 (✅ 1-row 계산 가능!)
- **Feature interactions**: 2-way, 3-way (✅ 가능, 단 EXP-019 교훈 - 단순함 유지)

**주의사항**:
- Look-ahead bias 방지 (미래 정보 사용 금지)
- Normalization 필수
- Feature selection (correlation, PCA, RFE)

---

### 2. Automated Feature Engineering
**출처**: AutoInt (arXiv), Pecan AI, TurinTech

**핵심 방법**:
- **Interaction features**: 곱셈, 나눗셈, 합
- **H-statistics**: Feature interaction strength 측정
- **Deep neural networks**: Feature interaction 자동 학습
- **Featuretools**: 자동 feature engineering 라이브러리

**우리 적용**:
- ✅ Interaction features (EXP-016 이미 사용)
- ✅ H-statistics로 interaction strength 측정
- ❌ Deep networks (EXP-014, 015 실패)
- ❌ Featuretools (time series lag 필요, InferenceServer 불가)

---

### 3. Feature Selection: Mutual Information vs Correlation
**출처**: Applied Intelligence, MDPI, Kaggle

**핵심 발견**:
- **Correlation**: 선형 관계만 포착
- **Mutual Information (MI)**: 비선형 관계 포착 ✅
- **CCMI**: Correlation + MI 결합 방법
- **Financial data**: MI가 correlation보다 더 많은 관계 발견

**우리 적용**:
- ✅ Mutual Information 기반 feature selection
- ✅ Nonlinear relationship 탐지
- ✅ Redundancy 제거 (correlation + MI)

---

## 근본 문제 분석

### 현재 상황
- **CV Sharpe 정체**: 0.6~0.75 (EXP-005~023 모두 동일)
- **Feature quality 한계**: 기존 94 features + 2-way interactions까지만 효과적
- **3-4way interactions**: Overfitting (EXP-019)

### 가설
**Feature quality 부족의 원인**:
1. 비선형 관계를 충분히 포착 못함
2. 카테고리별(M/V/P/S/I/E) 정보 통합 부족
3. Temporal patterns (date_id) 활용 부족
4. Volatility/Uncertainty 정보 부족

---

## EXP-024 전략

### Phase 1: 심화 데이터 분석 ⭐⭐⭐⭐⭐
**목표**: Feature 간 관계 심층 분석

**분석 항목**:
1. **Mutual Information 분석**
   - 각 feature와 target 간 MI score
   - Feature 간 MI (redundancy 탐지)
   - Nonlinear correlation 발견

2. **Nonlinear Correlation 탐지**
   - Spearman correlation (rank-based)
   - Distance correlation
   - Maximal Information Coefficient (MIC)

3. **Feature Group 분석**
   - M/V/P/S/I/E 카테고리별 특성
   - 카테고리 간 상호작용
   - Missing pattern 분석

**Output**: `results/phase1_data_analysis.csv`

**예상 시간**: 1시간

---

### Phase 2: Advanced Feature Engineering ⭐⭐⭐⭐⭐
**목표**: 연구 기반 고품질 feature 생성

#### 2A: DateTime Cyclical Features
```python
# date_id는 trading days (252 per year)
day_of_year = (date_id % 252) / 252
week_of_year = (date_id % 252) / (252 / 52)
month_of_year = (date_id % 252) / (252 / 12)

# Cyclical encoding
day_sin = sin(2π * day_of_year)
day_cos = cos(2π * day_of_year)
week_sin = sin(2π * week_of_year)
week_cos = cos(2π * week_of_year)
month_sin = sin(2π * month_of_year)
month_cos = cos(2π * month_of_year)
```
**근거**: Seasonality, cyclical patterns 포착 (Web Search)

---

#### 2B: Category-wise Statistics
```python
# Current row의 카테고리별 통계 (1-row 계산 가능!)
M_features = [M2, M3, M4, M12, ...]
V_features = [V7, V9, V10, V13, ...]
P_features = [P5, P7, P8, P10, ...]
S_features = [S2, S5, S8, ...]
I_features = [I2, ...]
E_features = [E12, E19, ...]

# Statistics per category
M_mean = mean(M_features)
M_std = std(M_features)
M_min = min(M_features)
M_max = max(M_features)
M_range = M_max - M_min
M_skew = skewness(M_features)
M_kurt = kurtosis(M_features)

# Repeat for V, P, S, I, E
```
**근거**: Category 정보 통합, higher-level features (EXP-019 meta-features)

---

#### 2C: Volatility & Uncertainty Proxies
```python
# Volatility proxies (1-row 계산)
vol_composite = sqrt(V13² + V7² + V9² + V10²)
vol_mean = mean([V13, V7, V9, V10])
vol_max = max([V13, V7, V9, V10])

# Cross-category volatility
market_vol = sqrt(M4² + M2² + M3²)
price_dispersion = std([P5, P7, P8, P10])
sentiment_dispersion = abs(S2 - S5)

# Economic uncertainty
econ_uncertainty = sqrt(E12² + E19²)
```
**근거**: Volatility features (Web Search, EXP-018)

---

#### 2D: Nonlinear Transformations
```python
# Log transforms (handle negative values)
M4_log = sign(M4) * log(1 + abs(M4))
V13_log = log(1 + V13)  # V always positive

# Square root (dampen large values)
M4_sqrt = sign(M4) * sqrt(abs(M4))
V13_sqrt = sqrt(V13)

# Exponential (amplify small values)
M4_exp = sign(M4) * (exp(abs(M4) / 10) - 1)  # scaled

# Polynomial (2nd, 3rd order only)
M4_squared = M4²
M4_cubed = M4³

# Inverse (1/x)
V13_inv = 1 / (V13 + eps)
```
**근거**: Nonlinear relationships (Web Search, MI analysis)

---

#### 2E: Ratio & Relative Features
```python
# Cross-category ratios
market_to_vol = M4 / (V13 + eps)
price_to_vol = P8 / (V13 + eps)
sentiment_to_econ = S2 / (E19 + eps)

# Relative to category mean
M4_rel = M4 / (M_mean + eps)
V13_rel = V13 / (V_mean + eps)

# Z-score within category
M4_z = (M4 - M_mean) / (M_std + eps)
```
**근거**: Relative features, normalization (EXP-001)

---

#### 2F: Interaction Features (Selected)
```python
# Top interactions from Phase 1 MI analysis
# Only create interactions with high MI score
# Example (will be determined by data):
M4_V13_interaction = M4 * V13 * (1 if MI(M4,V13) > threshold else 0)
```
**근거**: Data-driven interaction selection (Web Search)

---

### Phase 3: Mutual Information Feature Selection ⭐⭐⭐⭐
**목표**: 고품질 feature만 선택

**방법**:
1. **MI Score Calculation**
   - Each feature vs target MI
   - Feature vs feature MI (redundancy)

2. **CCMI (Correlation + MI)**
   - Remove high correlation features first
   - Then apply MI for nonlinear redundancy

3. **Top-N Selection**
   - Select top 50~100 by MI score
   - Remove redundant features (correlation > 0.9 or MI > threshold)

**Output**: `results/phase3_selected_features.csv`

**예상 시간**: 1시간

---

### Phase 4: Model Training & Evaluation ⭐⭐⭐⭐⭐
**목표**: 새 features로 성능 향상 확인

**모델**:
- XGBoost (EXP-016 baseline)
- Quantile regression (EXP-020)
- K = 250

**평가**:
- 5-fold TimeSeriesSplit
- CV Sharpe, Public Score 예측

**성공 조건**:
- ✅ CV Sharpe > 0.8 (현재 0.6 대비 +33%)
- ✅ CV Sharpe > 1.0 (최종 목표)

**Output**: `results/phase4_cv_results.csv`

**예상 시간**: 2시간

---

## 예상 Feature 수

### Baseline (EXP-016)
- Base: 20
- Interactions: 10
- **Total: 30**

### EXP-024 (추정)
- Base: 20
- DateTime: 6 (sin/cos for day, week, month)
- Category stats: 42 (6 categories × 7 stats)
- Volatility: 10
- Nonlinear transforms: 40 (top 20 features × 2 transforms)
- Ratio: 20
- **Subtotal: 138**

After MI selection:
- **Final: 50~100 features**

---

## 예상 결과

### Conservative (40% 확률)
- CV Sharpe: 0.75~0.85 (+25~42%)
- Public Score: 7~8 (현재 5.87)
- **평가**: 개선 있지만 목표 부족

### Expected (40% 확률)
- CV Sharpe: 0.85~1.0 (+42~67%)
- Public Score: 8~10
- **평가**: 의미 있는 개선

### Optimistic (20% 확률)
- CV Sharpe: 1.0~1.3 (+67~117%)
- Public Score: 10~13
- **평가**: 목표 근접 또는 달성

---

## 리스크 & 완화

### Risk 1: Overfitting
**완화책**:
- MI selection으로 redundancy 제거
- 단순한 transformations만 사용
- Cross-validation 철저히

### Risk 2: InferenceServer 호환성
**완화책**:
- 모든 features 1-row 계산 가능 확인
- Lag/rolling 절대 사용 안함
- 로컬 테스트 필수

### Risk 3: 계산 복잡도
**완화책**:
- Feature 수를 50~100으로 제한
- MI selection으로 불필요한 features 제거

---

## 일정

**Day 1 (오늘)**
- [x] EXP-024 계획 수립
- [ ] Phase 1: 데이터 분석 (1시간)
- [ ] Phase 2: Feature engineering (2시간)

**Day 2**
- [ ] Phase 3: MI feature selection (1시간)
- [ ] Phase 4: Model training (2시간)
- [ ] 결과 분석 및 제출 결정

**Total**: ~6시간

---

## 핵심 차별점 (vs 이전 실험)

**EXP-016~023**:
- Feature selection: Correlation, XGBoost importance
- Features: Base + 2-way interactions
- CV Sharpe: 0.6~0.75

**EXP-024**:
- ✅ **Mutual Information 기반 선택** (nonlinear 포착)
- ✅ **Category-wise statistics** (higher-level info)
- ✅ **DateTime cyclical features** (temporal patterns)
- ✅ **Volatility proxies** (uncertainty)
- ✅ **Nonlinear transformations** (log, sqrt, exp)
- ✅ **Data-driven approach** (연구 기반)

---

**Status**: 계획 완료, Phase 1 준비 완료
**Next**: `phase1_data_analysis.py` 실행
