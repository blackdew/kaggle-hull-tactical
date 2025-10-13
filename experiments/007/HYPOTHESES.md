# EXP-007 가설 및 실험 계획: 예측 정확도 근본 개선

## 배경

### EXP-006 실패 분석
- **시도**: k 파라미터 증가 (200→3000, 15배)
- **결과**: Sharpe 0.627→0.699 (+11.4%, 1.11배)
- **실패 원인**: 예측 정확도 자체가 너무 낮음
  - MSE: 0.00015
  - Correlation: 0.03~0.06
  - k는 신호 증폭만 가능, 신호 생성 불가

### Kaggle 메트릭 재이해
```
utility = min(max(sharpe, 0), 6) × Σ profits
```

**핵심 통찰**:
1. Sharpe는 [0, 6]으로 클램핑
2. Sharpe 6 이상은 의미 없음
3. **Profit이 핵심 변수**
4. 목표 17.395 = sharpe(6) × profit(~2.9)

**현재 상황**:
- Sharpe: 0.699 (best k=3000)
- 목표: 6.0
- 격차: 8.6배

---

## 목표

### Primary Goal
**Sharpe 1.5~3.0 달성** (현재 0.7 대비 2~4배)

### Milestone
- Milestone 1: Sharpe 1.0+ (현재 대비 1.4배)
- Milestone 2: Sharpe 1.5+ (2.1배)
- Milestone 3: Sharpe 2.0+ (2.9배)
- Stretch: Sharpe 3.0+ (4.3배)

### Success Criteria
- **Minimum**: Sharpe 1.0+ (CV), Kaggle utility 3.0+
- **Target**: Sharpe 1.5+ (CV), Kaggle utility 6.0+
- **Stretch**: Sharpe 2.0+ (CV), Kaggle utility 10.0+

---

## 가설 (우선순위별)

### 🥇 H1: Feature Engineering 확장
**가설**: 더 긴 시계열 + 다양한 features로 예측력 향상

**배경**:
- 현재 Lag [1, 5, 10]: 너무 짧음 (2주 이내)
- 시장 사이클은 20~60일 가능성
- Cross-sectional features 없음 (date 내 상대 비교)
- Volatility features 없음

**실험 내용**:
- **H1a: Longer Lags**
  - Lag [1, 5, 10, 20, 40, 60]
  - 예상: 중장기 패턴 포착, Sharpe +10~20%

- **H1b: Cross-Sectional Features**
  - Rank within date: feature별 순위 (0~1 normalize)
  - Quantile: feature별 분위수 (0.2, 0.4, 0.6, 0.8)
  - Z-score: (value - date_mean) / date_std
  - 예상: 상대적 강도 포착, Sharpe +15~30%

- **H1c: Volatility Features**
  - Rolling vol [5, 20, 60]
  - Volatility regime: 고변동 vs 저변동
  - Vol-normalized returns
  - 예상: 변동성 고려, Sharpe +10~25%

- **H1d: Momentum & Trend**
  - Return [5d, 20d, 60d]
  - EMA [10, 20, 40, 60]
  - MACD-like features
  - 예상: 트렌드 포착, Sharpe +5~15%

**측정 지표**:
- CV Sharpe (primary)
- Feature importance (top 20)
- Train vs CV 차이 (과적합 체크)
- Position distribution

**예상 결과**:
- Best case: Sharpe 1.2~1.5 (+71~114%)
- Realistic: Sharpe 0.9~1.1 (+29~57%)
- Worst case: Sharpe 0.75~0.85 (+7~21%)

**리스크**:
- 과적합: feature 수 증가 → overfitting
- 연산 시간: feature engineering 느려짐
- Cold start: lag features의 초기 NaN

**성공 기준**:
- CV Sharpe > 1.0
- Feature importance에 새 features 진입
- CV-Train Sharpe 차이 < 0.15 (과적합 방지)

---

### 🥈 H2: Volatility Scaling
**가설**: Vol-aware positioning으로 변동성 높은 구간에서 손실 감소

**배경**:
- 현재 `position = 1 + excess * k` (volatility 무시)
- Vol Ratio: 1.23~1.46 (시장 변동성 대비 높음)
- 고변동성 구간에서 과도한 position → 손실 증폭

**실험 내용**:
- **H2a: Simple Vol Scaling**
  ```python
  position = 1 + (excess * k) / rolling_vol_20
  ```
  - 고변동성 시 position 감소
  - 예상: Sharpe +10~15%, Vol Ratio 감소

- **H2b: Vol Targeting**
  ```python
  target_vol = 0.01  # daily
  position = (1 + excess * k) * (target_vol / realized_vol)
  ```
  - 일정 변동성 유지
  - 예상: Sharpe +15~25%, Vol Ratio → 1.0

- **H2c: Dynamic k by Vol Regime**
  ```python
  if rolling_vol > high_threshold:
      k_adjusted = k * 0.5
  elif rolling_vol < low_threshold:
      k_adjusted = k * 1.5
  else:
      k_adjusted = k
  ```
  - 고변동성: 보수적 (k↓)
  - 저변동성: 공격적 (k↑)
  - 예상: Sharpe +20~30%

**측정 지표**:
- CV Sharpe
- Vol Ratio (목표: < 1.2)
- Max position, Pos std

**예상 결과**:
- Best case: Sharpe 1.0~1.2 (+43~71%)
- Realistic: Sharpe 0.85~0.95 (+21~36%)

**성공 기준**:
- CV Sharpe > 0.85
- Vol Ratio < 1.2
- vs H1: 독립적 효과 확인 (H1+H2 > H1)

---

### 🥉 H3: Ensemble
**가설**: 여러 모델 결합으로 예측 안정성 향상

**배경**:
- EXP-005: XGBoost (0.627) vs LightGBM (0.611)
- 각 모델이 다른 패턴 포착 가능성
- Ensemble로 분산 감소

**실험 내용**:
- **H3a: Simple Average**
  ```python
  pred = (xgb_pred + lgbm_pred + lasso_pred) / 3
  ```
  - 동일 가중치
  - 예상: Sharpe 0.8~0.9

- **H3b: Weighted Average**
  ```python
  pred = 0.5*xgb_pred + 0.3*lgbm_pred + 0.2*lasso_pred
  ```
  - CV Sharpe 기반 가중치
  - 예상: Sharpe 0.85~0.95

- **H3c: Stacking**
  - Meta-learner (Linear Regression)
  - Base: XGBoost, LightGBM, Lasso
  - 예상: Sharpe 0.9~1.1

**측정 지표**:
- CV Sharpe
- 각 모델별 기여도
- Correlation between models (다양성 확인)

**예상 결과**:
- Best case: Sharpe 1.0~1.2 (+43~71%)
- Realistic: Sharpe 0.85~0.95 (+21~36%)

**성공 기준**:
- CV Sharpe > best single model + 0.05
- Model correlation < 0.9 (다양성 확보)

---

### H4: Neural Network (선택)
**가설**: DL로 복잡한 비선형 관계 포착

**배경**:
- XGBoost/LightGBM: tree-based, feature interaction 제한적
- Neural Network: 임의 비선형 함수 근사

**실험 내용**:
- **H4a: MLP**
  - 3~5 layers, ReLU, Dropout
  - BatchNorm
  - 예상: Sharpe 0.9~1.2

- **H4b: LSTM** (시계열)
  - Sequence length: 10~20
  - Bi-directional
  - 예상: Sharpe 1.0~1.4

- **H4c: Transformer** (advanced)
  - Attention mechanism
  - 예상: Sharpe 1.2~1.8

**리스크**:
- 훈련 시간 길어짐
- 과적합 위험 높음
- Hyperparameter 민감

**조건**: H1~H3 실패 시 (Sharpe < 1.0)

---

### H5: Target Engineering (Radical, 선택)
**가설**: 다른 target 정의로 예측 용이성 향상

**배경**:
- 현재 target: market_forward_excess_returns (regression)
- Regression은 magnitude 예측 어려움
- Classification (sign)이 더 쉬울 수 있음

**실험 내용**:
- **H5a: Classification + Regression**
  1. Classifier: sign(excess) 예측 (binary)
  2. Regressor: |excess| 예측 (magnitude)
  3. Final: sign × magnitude
  - 예상: Sharpe 1.0~1.5

- **H5b: Quantile Regression**
  - P10, P50, P90 예측
  - Tail events 중점
  - 예상: Sharpe 1.2~1.8

- **H5c: Multi-task Learning**
  - Task 1: Excess return
  - Task 2: Volatility
  - Task 3: Sign
  - 예상: Sharpe 1.3~2.0

**조건**: H1~H4 실패 시 (Sharpe < 1.2)

---

## 실험 계획

### Phase 1: Feature Engineering (H1)
**목표**: Sharpe 1.0+ 달성

**순서**:
1. H1a: Longer Lags (30분)
2. H1b: Cross-Sectional (45분)
3. H1c: Volatility Features (30분)
4. H1d: Momentum & Trend (30분)
5. 조합 테스트: H1a+b+c+d (1시간)

**예상 시간**: 3~4시간

**의사결정**:
```
H1 결과:
  ├─ Sharpe > 1.2 ✅ → Phase 2 (H2), 목표 달성 근접
  ├─ Sharpe 1.0~1.2 📊 → Phase 2 (H2), 추가 개선 필요
  ├─ Sharpe 0.85~1.0 📉 → Phase 2 (H2) + Phase 3 (H3) 필수
  └─ Sharpe < 0.85 ❌ → H4 (Neural Network) 고려
```

---

### Phase 2: Volatility Scaling (H2)
**조건**: Phase 1 완료 후

**목표**: Sharpe +0.15~0.25 추가 개선

**순서**:
1. H2a: Simple Vol Scaling (30분)
2. H2b: Vol Targeting (30분)
3. H2c: Dynamic k (45분)
4. 최고 조합 선정

**예상 시간**: 2시간

**의사결정**:
```
H1 + H2 결과:
  ├─ Sharpe > 1.5 🎉 → Kaggle 제출, Phase 3 (H3) 선택
  ├─ Sharpe 1.2~1.5 📈 → Phase 3 (H3) 진행
  ├─ Sharpe 1.0~1.2 📊 → Phase 3 (H3) 필수
  └─ Sharpe < 1.0 ⚠️ → H4 (Neural Network) 필수
```

---

### Phase 3: Ensemble (H3)
**조건**: Phase 2 완료 후 Sharpe < 1.5

**목표**: Sharpe +0.1~0.2 추가 개선

**순서**:
1. H3a: Simple Average (30분)
2. H3b: Weighted Average (15분)
3. H3c: Stacking (1시간)

**예상 시간**: 2시간

**의사결정**:
```
H1 + H2 + H3 결과:
  ├─ Sharpe > 1.5 🎯 → Kaggle 제출, k 재조정
  ├─ Sharpe 1.2~1.5 📊 → Kaggle 제출, H4 고려
  └─ Sharpe < 1.2 ❌ → H4 (Neural Network) 필수
```

---

### Phase 4: Neural Network (H4) - Conditional
**조건**: Phase 3 완료 후 Sharpe < 1.5

**목표**: Sharpe 1.5~2.0 달성

**순서**:
1. H4a: MLP (2시간)
2. H4b: LSTM (3시간)
3. (H4c: Transformer는 시간 있을 때만)

**예상 시간**: 5~8시간

---

## 실험 산출물

### 필수 파일
```
experiments/007/
├── HYPOTHESES.md           # 이 파일
├── run_experiments.py      # 실험 실행 스크립트
├── feature_engineering.py  # 확장된 feature 생성
├── volatility_scaling.py   # Vol scaling 로직
├── ensemble.py             # Ensemble 구현
├── results/
│   ├── h1_feature_eng.csv
│   ├── h2_vol_scaling.csv
│   ├── h3_ensemble.csv
│   └── summary.csv
└── REPORT.md               # 실험 결과 (완료 후)
```

---

## 리스크 관리

### Risk 1: 과적합
**증상**: CV Sharpe 높지만 Kaggle 낮음
**확률**: 40%
**대응**:
- Train vs CV Sharpe 차이 모니터링
- Early stopping
- Feature 수 제한

### Risk 2: 개선폭 여전히 부족
**증상**: H1~H3 후에도 Sharpe < 1.0
**확률**: 30%
**대응**: H4 (Neural Network) 또는 H5 (Target Engineering)

### Risk 3: 연산 시간 폭발
**증상**: Feature engineering 너무 느림
**확률**: 20%
**대응**:
- Feature 수 제한 (top 50~100)
- Sampling

### Risk 4: Cold start 문제
**증상**: Lag 60 features로 인한 초기 NaN
**확률**: 30%
**대응**:
- Forward fill
- Feature별 lag 차별화 (중요 feature만 lag 60)

---

## 성공 기준 (Exit Criteria)

### Minimum Success
- ✅ CV Sharpe > 1.0
- ✅ Kaggle utility > 3.0
- ✅ CV-Train Sharpe 차이 < 0.2

### Target Success
- ✅ CV Sharpe > 1.5
- ✅ Kaggle utility > 6.0
- ✅ Feature importance 분석 완료

### Stretch Success
- ✅ CV Sharpe > 2.0
- ✅ Kaggle utility > 10.0
- ✅ 17.395 달성 로드맵 명확화

---

## 다음 단계

1. ✅ HYPOTHESES.md 작성
2. ⏭️ `run_experiments.py` 작성 (Phase 1: H1)
3. ⏭️ `feature_engineering.py` 작성 (확장 features)
4. ⏭️ Phase 1 실행 (H1a~d)
5. ⏭️ 결과 분석 및 Phase 2 진행 여부 결정

---

**작성일**: 2025-10-13
**전제**: EXP-006 k 접근 실패 (Sharpe 0.7 한계)
**목표**: 예측 정확도 근본 개선으로 Sharpe 1.5~3.0 달성
**핵심**: Feature Engineering + Volatility Scaling + Ensemble
