# EXP-021: Quantile Regression Grid Search

**목표**: Quantile alpha와 Confidence scaling의 최적 조합 찾기
**전략**: 모든 조합을 병렬 실험 → 최고 성능 1개 선택

---

## 실험 설계

### Grid Search Space

**Quantile Alpha 조합** (7가지):
1. `(0.05, 0.50, 0.95)` - Wide
2. `(0.10, 0.50, 0.90)` - EXP-020 baseline
3. `(0.15, 0.50, 0.85)` - Moderate
4. `(0.20, 0.50, 0.80)` - Narrow
5. `(0.05, 0.25, 0.50, 0.75, 0.95)` - 5-quantile
6. `(0.10, 0.30, 0.50, 0.70, 0.90)` - 5-quantile moderate
7. `(0.25, 0.50, 0.75)` - IQR

**Confidence Scaling** (6가지):
1. `x1.0` - No scaling
2. `x2.0` - Conservative
3. `x3.0` - Moderate
4. `x5.0` - EXP-020 baseline
5. `x7.0` - Aggressive
6. `x10.0` - Very aggressive

**총 조합**: 7 × 6 = **42 configurations**

### Asymmetric Confidence (추가 실험)

**8가지 추가**:
1. Upside x2, Downside x1 (asymmetric conservative)
2. Upside x5, Downside x3 (asymmetric moderate)
3. Upside x7, Downside x5 (asymmetric aggressive)
4. Upside x10, Downside x5 (very asymmetric)
5. Upside x3, Downside x5 (defensive)
6. Upside x5, Downside x7 (risk-averse)
7. Upside x1, Downside x2 (bearish)
8. Upside x2, Downside x3 (moderately bearish)

**총 실험**: 42 + 8 = **50 configurations**

---

## 구현 계획

### Phase 1: 실험 코드 작성 (30분)
**파일**: `experiments/021/grid_search.py`

```python
# Configuration space
quantile_alphas = [
    [0.05, 0.50, 0.95],
    [0.10, 0.50, 0.90],
    [0.15, 0.50, 0.85],
    [0.20, 0.50, 0.80],
    [0.05, 0.25, 0.50, 0.75, 0.95],
    [0.10, 0.30, 0.50, 0.70, 0.90],
    [0.25, 0.50, 0.75],
]

confidence_scalings = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

asymmetric_configs = [
    {'upside': 2, 'downside': 1},
    {'upside': 5, 'downside': 3},
    {'upside': 7, 'downside': 5},
    {'upside': 10, 'downside': 5},
    {'upside': 3, 'downside': 5},
    {'upside': 5, 'downside': 7},
    {'upside': 1, 'downside': 2},
    {'upside': 2, 'downside': 3},
]

# For each configuration:
#   - Train quantile models
#   - 5-fold CV
#   - Calculate Sharpe
#   - Save results
```

**출력**: `results/grid_search_results.csv` (50 rows)

### Phase 2: 실행 및 결과 수집 (2-3시간)
- 모든 50 configurations 실행
- 각 configuration마다 5-fold CV
- CV Sharpe, Std 기록

### Phase 3: 최적 설정 선택 (10분)
- Best configuration 선택 (highest CV Sharpe)
- Top 5 configurations 분석
- 결과 시각화

### Phase 4: InferenceServer 구현 (30분)
- Best configuration으로 `submissions/submission_exp021.py` 작성
- Kaggle 제출

---

## 예상 결과

### 시간 배분
- Phase 1 (코드): 30분
- Phase 2 (실행): 2-3시간 (50 configs × 3-4분)
- Phase 3 (분석): 10분
- Phase 4 (제출): 30분
- **총**: 3.5-4.5시간

### 예상 개선
- **Conservative**: +15-20% (5.87 → 6.8-7.0)
- **Expected**: +20-30% (5.87 → 7.0-7.6)
- **Optimistic**: +30-40% (5.87 → 7.6-8.2)

### 성공 기준
- CV Sharpe > 0.70 (EXP-020: 0.637)
- Public Score > 7.0 (EXP-020: 5.872)

---

## 실행 순서

1. `experiments/021/grid_search.py` 작성
2. `python experiments/021/grid_search.py` 실행
3. `results/grid_search_results.csv` 확인
4. Best config로 submission 파일 작성
5. Kaggle 제출 및 Public Score 확인

---

**상태**: 계획 완료, 구현 준비됨
**다음**: Phase 1 코드 작성 시작
