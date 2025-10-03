# EXP-005: Gradient Boosting + Feature Engineering

## 개요

**목표**: Lasso를 XGBoost/LightGBM으로 교체하여 예측력 향상

**배경**:
- EXP-004 실패: Lasso + k 조정으로는 한계 명확 (Kaggle 0.15~0.44)
- 상위권 17.333점과 38~115배 차이
- 근본 원인: 모델 자체의 예측력 부족 (상관 0.03~0.06 수준)

**접근**:
- Lasso (선형) → XGBoost/LightGBM (비선형, feature interaction)
- Top-20 features → 전체 98 features + engineered features
- Feature engineering: Lag, Rolling, Interaction

---

## 실험 목록

| 실험 | 모델 | Features | k 값 | 예상 Sharpe | 우선순위 |
|------|------|----------|------|------------|---------|
| **H1** | XGBoost | 전체 98개 | 50, 100, 200 | 0.7~0.9 | ⭐⭐⭐⭐⭐ |
| **H2** | LightGBM | 전체 98개 | 50, 100, 200 | 0.7~0.9 | ⭐⭐⭐⭐ |
| **H3** | XGBoost | + Lag/Rolling | 50, 100, 200 | 0.75~0.95 | ⭐⭐⭐⭐⭐ |
| **H4** | XGBoost | + Interaction | 50, 100, 200 | 0.8~1.0 | ⭐⭐⭐⭐ |
| **H5** | Ensemble | XGB+LGBM+Lasso | 50, 100, 200 | 0.8~1.0 | ⭐⭐⭐ |
| **H6** | XGBoost | Regime-based | regime별 | 0.85~1.05 | ⭐⭐⭐ |

**필수**: H1, H2, H3
**선택**: H4, H5, H6 (시간 있으면)

---

## 빠른 실행

### 전체 실험 (Phase 1~3)
```bash
# H1: XGBoost Baseline
uv run python experiments/005/run_experiments.py --hypothesis H1

# H2: LightGBM Baseline
uv run python experiments/005/run_experiments.py --hypothesis H2

# H3: Feature Engineering
uv run python experiments/005/run_experiments.py --hypothesis H3

# H4~H6: Advanced (선택)
uv run python experiments/005/run_experiments.py --hypothesis H4
uv run python experiments/005/run_experiments.py --hypothesis H5
uv run python experiments/005/run_experiments.py --hypothesis H6

# 전체 실행 (시간 오래 걸림, 4~6시간+)
uv run python experiments/005/run_experiments.py --all
```

### Phase별 실행
```bash
# Phase 1: Baseline models (H1, H2)
uv run python experiments/005/run_experiments.py --phase 1

# Phase 2: Feature engineering (H3, H4)
uv run python experiments/005/run_experiments.py --phase 2

# Phase 3: Advanced (H5, H6)
uv run python experiments/005/run_experiments.py --phase 3
```

---

## 상세 실행 가이드

### 1. 환경 확인
```bash
# 필요한 패키지 설치
uv pip install xgboost lightgbm scikit-learn pandas numpy

# 데이터 확인
ls data/train.csv  # 있어야 함
ls data/test.csv   # 있어야 함
```

### 2. H1: XGBoost Baseline 실행

**목적**: Lasso 대비 XGBoost 성능 확인

```bash
uv run python experiments/005/run_experiments.py --hypothesis H1
```

**예상 출력**:
```
[INFO] Starting H1: XGBoost Baseline
[INFO] Loading data... (8990 rows)
[INFO] Features: 98 (exclude date_id, targets, lagged)
[INFO] Target: market_forward_excess_returns

[INFO] Cross-validation (TimeSeriesSplit, 5 folds)
Fold 1/5: Sharpe 0.72, Vol Ratio 1.08, k=100
Fold 2/5: Sharpe 0.68, Vol Ratio 1.12, k=100
Fold 3/5: Sharpe 0.75, Vol Ratio 1.05, k=100
Fold 4/5: Sharpe 0.71, Vol Ratio 1.10, k=100
Fold 5/5: Sharpe 0.69, Vol Ratio 1.09, k=100

[RESULT] H1: XGBoost k=100
  Sharpe: 0.71 ± 0.03
  Vol Ratio: 1.09 ± 0.03
  vs Lasso (0.604): +17.5%

[INFO] Feature importance saved: results/h1_feature_importance.csv
[INFO] Results saved: results/h1_xgboost_folds.csv
```

**확인 포인트**:
- Sharpe > 0.7? → 성공!
- Feature importance 상위 10개 확인
- Lasso Top-20과 비교

### 3. H2: LightGBM Baseline 실행

```bash
uv run python experiments/005/run_experiments.py --hypothesis H2
```

**확인 포인트**:
- XGBoost와 성능 비교 (±0.05 범위)
- 훈련 속도 비교
- Feature importance 패턴 비교

### 4. H3: Feature Engineering 실행

**목적**: Lag/Rolling features 추가 효과 확인

```bash
uv run python experiments/005/run_experiments.py --hypothesis H3
```

**생성되는 Features**:
- Lag features: M4_lag1, M4_lag5, V13_lag10 등
- Rolling mean: M4_rolling_mean_5, V13_rolling_mean_10 등
- Rolling std: M4_rolling_std_5 등
- Diff: M4_diff1 등

**확인 포인트**:
- Sharpe > H1 + 0.05?
- Feature importance에 lag/rolling 진입?
- 안정성(Sharpe Std) 개선?

### 5. 결과 분석

```bash
# 전체 요약 확인
cat experiments/005/results/summary.csv

# Feature importance 확인
cat experiments/005/results/h1_feature_importance.csv | head -20

# 폴드별 상세 결과
cat experiments/005/results/h1_xgboost_folds.csv
```

---

## 예상 결과 해석

### Case A: Sharpe 0.7~0.85 (목표 달성) ✅
- **해석**: XGBoost가 Lasso 대비 15~40% 향상
- **다음 단계**:
  1. Best model (H1~H3 중)로 Kaggle 제출
  2. k 값 미세 조정 (best_k ± 50)
  3. H4~H6 선택적 시도

### Case B: Sharpe 0.85~1.0 (대성공) 🎉
- **해석**: Feature engineering 효과 탁월
- **다음 단계**:
  1. 즉시 Kaggle 제출
  2. Ensemble (H5) 시도
  3. Regime-based (H6) 고려

### Case C: Sharpe < 0.7 (실패) ⚠️
- **해석**: XGBoost도 예측력 부족
- **다음 단계**:
  1. Feature importance 재분석
  2. Hyperparameter tuning
  3. 다른 접근 고려 (Deep Learning, 외부 데이터 등)

---

## 제출 파일 생성

### Best model 선택
```bash
# Summary에서 최고 Sharpe 모델 확인
cat experiments/005/results/summary.csv | sort -t',' -k2 -rn | head -1

# 예: H3_k100이 최고 (Sharpe 0.82)
```

### Kaggle 제출용 파일 생성
```bash
# 방법 1: 제출 스크립트 자동 생성
uv run python experiments/005/generate_submission.py --model H3 --k 100

# 방법 2: 수동으로 kaggle_inference 업데이트
# kaggle_kernel/kaggle_inference_xgboost.py 생성
```

### 제출
```bash
# Kaggle notebook에 kaggle_inference_xgboost.py 복사
# 실행 후 submission.parquet 생성 확인
# Submit to Competition
```

---

## 디버깅 가이드

### 문제: ModuleNotFoundError
```bash
# 해결: uv로 패키지 설치
uv pip install xgboost lightgbm
```

### 문제: Memory Error
```bash
# 해결: Feature engineering 범위 축소
# feature_engineering.py에서 top_n=10으로 변경 (기본 20)
```

### 문제: 실행 시간 너무 김
```bash
# 해결 1: n_estimators 줄이기 (500 → 200)
# 해결 2: 폴드 수 줄이기 (5 → 3)
# 해결 3: H1, H3만 실행
uv run python experiments/005/run_experiments.py --hypothesis H1
uv run python experiments/005/run_experiments.py --hypothesis H3
```

### 문제: XGBoost가 Lasso보다 낮음
```bash
# 체크리스트:
# 1. Feature scaling 필요? (XGBoost는 불필요하지만 확인)
# 2. Hyperparameter 조정 필요?
# 3. Overfitting? (max_depth 줄이기, subsample 줄이기)
# 4. Data leakage? (lagged features 제외 확인)
```

---

## 파일 구조

```
experiments/005/
├── README.md                    # 이 파일
├── HYPOTHESES.md               # 가설 및 실험 계획
├── REPORT.md                   # 실험 결과 (실행 후 생성)
├── STRATEGY_PIVOT.md           # 전략 전환 배경
├── run_experiments.py          # 메인 실험 스크립트
├── feature_engineering.py      # Feature 생성 함수
├── models.py                   # 모델 정의
├── generate_submission.py      # 제출 파일 생성
├── results/
│   ├── h1_xgboost_folds.csv
│   ├── h2_lightgbm_folds.csv
│   ├── h3_feature_eng_folds.csv
│   ├── h1_feature_importance.csv
│   └── summary.csv
└── submissions/
    └── best_model_submission.csv
```

---

## 일정 (추천)

### Day 1 오전 (2~3시간)
- [ ] H1: XGBoost Baseline 실행
- [ ] H2: LightGBM Baseline 실행
- [ ] 결과 분석, Feature importance 확인

### Day 1 오후 (2~3시간)
- [ ] H3: Feature Engineering 실행
- [ ] 결과 비교 (H1 vs H2 vs H3)
- [ ] Best model 선정

### Day 1 저녁 (1~2시간)
- [ ] Kaggle 제출 파일 생성
- [ ] 제출 및 점수 확인

### Day 2 (선택, 시간 있으면)
- [ ] H4: Interaction features
- [ ] H5: Ensemble
- [ ] H6: Regime-based model

---

## 성공 기준

### Minimum (최소)
- ✅ H1, H2, H3 실행 완료
- ✅ CV Sharpe > 0.7 (Lasso 대비 +15%)
- ✅ Kaggle 제출 1회

### Target (목표)
- ✅ CV Sharpe > 0.85 (+40%)
- ✅ Kaggle 점수 > 3.0 (현재 0.44 대비 7배)

### Stretch (도전)
- ✅ CV Sharpe > 1.0 (+65%)
- ✅ Kaggle 점수 > 5.0 (11배)
- ✅ 상위 50% 진입

---

## 참고

- EXP-000: Feature 분석
- EXP-002: Lasso baseline (Sharpe 0.604)
- EXP-004: k 조정 실패 (Kaggle 0.15~0.44)
- **EXP-005: 모델 전환** (XGBoost/LightGBM)

**핵심**: k 파라미터가 아닌 **모델 자체를 바꿔야 17+ 달성 가능**

---

## 질문/이슈

문제 발생 시:
1. GitHub Issues에 리포트
2. 또는 실험 로그 공유
