# EXP-006 가설 및 검증 계획

## 현재 상황 분석

### 성능 추이
- **EXP-002**: Lasso k=50, CV 0.604 → Kaggle 0.441
- **EXP-004**: Lasso k=500, CV 0.836 → Kaggle 0.150 (실패, 과적합)
- **EXP-005**: XGBoost+FeatEng k=200, CV 0.627 → Kaggle **0.724** ✅

### 문제 정의
- **현재 점수**: 0.724
- **목표 점수**: **17.395** (현재 대비 24배)
- **격차**: 16.671 (절대값)
- **전략**: 단계적 접근으로 17.395 달성

---

## 핵심 질문: 17.395 달성을 위한 방법은?

### 분석 1: k 파라미터는 얼마나 높여야 하는가?
**현재 상황**:
- k=200, Kaggle 0.724
- k를 24배 높이면? k=4800 → 점수도 24배? (매우 위험)
- EXP-004 k=500 실패 사례 (Lasso)

**가설**:
- k를 점진적으로 높여서 한계점 찾기
- XGBoost는 Lasso보다 안정적이므로 k=500~2000 가능성
- k=1000~3000 범위에서 10.0+ 달성 목표

**리스크**:
- k 과다 → 분산 폭발 → 점수 하락 (EXP-004 재현)
- 최적 k를 넘으면 급격히 하락

### 분석 2: 예측력 자체를 높여야 하는가?
**현재 모델 한계**:
- excess return 예측 정확도가 낮음 (correlation 0.03~0.06)
- 약한 신호로는 k를 높여도 노이즈만 증폭

**가설**:
- Feature Engineering으로 예측 정확도 향상
- Ensemble로 신호 안정화
- 더 복잡한 모델 (Neural Network)로 비선형 포착

**목표**:
- 예측 정확도 2~3배 향상 시 k를 낮춰도 고득점 가능

### 분석 3: 전략 자체를 바꿔야 하는가?
**현재 전략**:
- `position = 1 + excess * k` (단순 선형)
- Volatility 무시, Regime 무시

**가설**:
- Volatility scaling으로 Sharpe 2배 향상
- Regime switching (bull/bear 구분)
- Dynamic leverage (상황별 k 조정)

**목표**:
- 전략 개선으로 k=200에서도 5.0+ 달성

---

## 가설 (우선순위별)

### 🥇 H1: k 파라미터 최적화
**가설**: k=200 → k=400~800로 증가 시 Sharpe 및 Kaggle 점수 향상

**근거**:
- EXP-005에서 k↑ → Sharpe↑ 트렌드
- CV→Kaggle 성능 안정적 (과적합 없음)

**실험**:
- Phase 1a: k = [300, 400, 500, 600, 800] CV 테스트
- Phase 1b: k = [1000, 1500, 2000, 3000] CV 테스트 (aggressive)
- 각 k별 Sharpe, vol_ratio, position_std 측정
- TimeSeriesSplit 5-fold

**예상 결과**:
- k=500~800: Kaggle 1.5~3.0
- k=1000~2000: Kaggle 5.0~10.0 (or 폭망 < 0.5)
- k=3000: Kaggle 15.0+ (or 폭망 < 0.1)

**리스크 관리**:
- k=500 결과 < 0.5 → 중단, Phase 2로
- k=1000 결과 < 1.0 → 중단, 예측력 개선 우선
- k=2000 결과 > 10.0 → k=3000, 4000 추가 시도

**성공 기준**:
- CV Sharpe > 0.70
- vol_ratio < 2.0 (k 높은 경우 허용)
- Kaggle 제출 시 5.0+ 목표, 최종 17.395 달성

---

### 🥈 H2: Volatility Scaling
**가설**: Vol-aware positioning으로 Sharpe 개선 (특히 고변동 구간)

**근거**:
- 현재 공식은 vol 무시 → 변동성 높을 때 과다 포지션
- EXP-002에서 vol-aware 효과 확인

**실험**:
- H2a: `position = 1 + (excess * k) / rolling_vol_20`
- H2b: `position = 1 + (excess * k) / sqrt(rolling_vol_60)`
- H2c: Vol targeting: `position * (target_vol / realized_vol)`

**측정**:
- Sharpe, vol_ratio, max_position, position_std

**예상 결과**:
- Sharpe +5~10% (0.627 → 0.66~0.69)
- vol_ratio 감소 (1.23 → 1.1 이하)

**성공 기준**:
- CV Sharpe > 0.65 (H1 최고값과 비교)
- vol_ratio < 1.2

---

### 🥉 H3: Feature Engineering 확장
**가설**: 더 긴 시계열 feature (Lag 20~60, EMA, Momentum)로 예측력 향상

**근거**:
- 현재 Lag [1,5,10]은 단기
- 시장 사이클은 20~60일 가능성

**실험**:
- H3a: Lag [1,5,10,20,40,60] (기존 [1,5,10])
- H3b: EMA [10,20,40,60]
- H3c: Momentum: return_5d, return_20d, return_60d

**측정**:
- Sharpe, feature importance, 과적합 여부 (CV vs train 차이)

**예상 결과**:
- Sharpe +3~7% (0.627 → 0.65~0.67)
- 과적합 리스크 (feature 증가로)

**성공 기준**:
- CV Sharpe > 0.65
- CV-train Sharpe 차이 < 0.1 (과적합 방지)

---

### H4: Ensemble
**가설**: XGBoost + LightGBM 결합으로 안정성 향상

**근거**:
- H3 XGBoost: Sharpe 0.627
- H2 LightGBM: Sharpe 0.611
- Ensemble로 분산 감소

**실험**:
- H4a: Simple average: (XGBoost + LightGBM) / 2
- H4b: Weighted: 0.7*XGBoost + 0.3*LightGBM

**예상 결과**:
- Sharpe 0.62~0.64 (소폭 개선)

**성공 기준**:
- CV Sharpe > 0.63

---

## 실험 계획

### Phase 1: k 파라미터 최적화 (H1)
**목표**: k 최적값 찾기

**실험 순서**:
1. k=[200, 300, 400, 500, 600, 800, 1000] CV 5-fold
2. 각 k별 Sharpe, vol_ratio, position_std 측정
3. 최적 k 선정

**예상 시간**: 1~2시간 (모델 학습은 동일, k만 변경)

**의사결정**:
```
Phase 1a (k=300~800) 결과:
  ├─ Kaggle k=800 > 3.0 ✅ → Phase 1b (k=1000~3000) 진행
  ├─ Kaggle k=800: 1.5~3.0 📊 → Phase 2 (Vol Scaling) + 재시도
  └─ Kaggle k=800 < 1.0 ❌ → k↑는 한계, Phase 2/3 pivot

Phase 1b (k=1000~3000) 결과:
  ├─ Kaggle > 15.0 🎯 → k 미세 조정으로 17.395 달성
  ├─ Kaggle 10.0~15.0 📈 → k 더 높이기 or Phase 2 결합
  ├─ Kaggle 5.0~10.0 📊 → Phase 2/3 결합 필요
  └─ Kaggle < 3.0 ❌ → k↑ 실패, 예측력 개선 우선
```

---

### Phase 2: Volatility Scaling (H2)
**조건**: Phase 1 완료 후

**목표**: Vol-aware positioning으로 Sharpe 추가 개선

**실험 순서**:
1. Phase 1 최적 k 사용
2. H2a, H2b, H2c CV 테스트
3. 최고 성능 조합 선정

**예상 시간**: 2~3시간

**의사결정**:
```
Vol Scaling + 최적 k 조합:
  ├─ Kaggle > 15.0 🎯 → 17.395 근접, 미세 조정
  ├─ Kaggle 10.0~15.0 📈 → Phase 3 결합 or k 추가 조정
  ├─ Kaggle 5.0~10.0 📊 → Phase 3 필수
  └─ Kaggle < 5.0 → Phase 3 + Ensemble (H4)
```

---

### Phase 3: Feature Engineering 확장 (H3)
**조건**: Phase 2 완료 후 목표 미달 시

**목표**: 더 긴 시계열 feature로 예측력 향상

**실험 순서**:
1. H3a, H3b, H3c 각각 CV 테스트
2. 과적합 체크 (train vs CV Sharpe)
3. 최고 성능 조합 선정

**예상 시간**: 4~6시간 (feature 재생성, 모델 재학습)

**의사결정**:
```
Feature Eng + Vol Scaling + 최적 k:
  ├─ Kaggle > 17.0 🎉 → 목표 달성! 17.395 도전
  ├─ Kaggle 12.0~17.0 📈 → 미세 조정 계속
  ├─ Kaggle 8.0~12.0 📊 → Phase 4 (Ensemble) 필수
  └─ Kaggle < 8.0 → 전략 재검토 (새로운 접근 필요)
```

---

### Phase 4: Ensemble (H4) - 최후 수단
**조건**: Phase 3 실패 시

**목표**: 여러 모델 결합으로 안정성 확보

**예상 시간**: 2~3시간

---

## 실험 산출물

### 필수 파일
```
experiments/006/
├── run_experiments.py      # 실험 실행 스크립트
├── results/
│   ├── h1_k_grid.csv       # Phase 1 결과
│   ├── h2_vol_scaling.csv  # Phase 2 결과
│   ├── h3_feature_eng.csv  # Phase 3 결과
│   ├── h4_ensemble.csv     # Phase 4 결과
│   └── summary.csv         # 전체 요약
└── REPORT.md               # 최종 결과 리포트
```

### Kaggle 제출 파일
- **단 1개만**: 최고 성능 조합
- 예: `kaggle_kernel/kaggle_inference_20251010_exp006_best.py`

---

## 의사결정 트리 (목표: 17.395)

```
시작: EXP-005 CV 0.627, Kaggle 0.724

Phase 1a: k=300~800
  ├─ Kaggle > 3.0 → Phase 1b (k=1000~3000)
  ├─ Kaggle 1.5~3.0 → Phase 2 + 재조정
  └─ Kaggle < 1.0 → 예측력 개선 우선 (Phase 2/3)

Phase 1b: k=1000~3000
  ├─ Kaggle > 15.0 🎯 → 17.395 도전 (k 미세 조정)
  ├─ Kaggle 10.0~15.0 → Phase 2 결합
  ├─ Kaggle 5.0~10.0 → Phase 2 + 3 결합
  └─ Kaggle < 5.0 → 전면 재검토

Phase 2: Vol Scaling + Phase 1 최적 k
  ├─ Kaggle > 17.0 🎉 → 목표 달성
  ├─ Kaggle 12.0~17.0 → Phase 3 추가
  └─ Kaggle < 12.0 → Phase 3 필수

Phase 3: Feature Eng + Vol Scaling + 최적 k
  ├─ Kaggle > 17.0 🎉 → 목표 달성
  ├─ Kaggle 12.0~17.0 → Phase 4 or 미세 조정
  └─ Kaggle < 12.0 → Phase 4 (Ensemble)

Phase 4: Ensemble + 모든 최적화
  ├─ Kaggle > 17.0 🎉 → 목표 달성
  └─ Kaggle < 17.0 → 새로운 전략 필요 (Regime, Neural, etc)

반복: 17.395 달성까지 계속
```

---

## 성공 기준

### Milestone 1: 초기 개선
- Kaggle 3.0+ (현재 0.724 대비 4배)

### Milestone 2: 중간 목표
- Kaggle 8.0+ (현재 대비 11배)

### Milestone 3: 고급 목표
- Kaggle 12.0+ (현재 대비 16배)

### Final Goal: 목표 달성
- **Kaggle 17.395** (현재 대비 24배)

---

## 핵심 원칙

1. **목표 주도**: 17.395 달성이 최우선 목표
2. **가설 검증**: CV로 먼저 검증 후 Kaggle 제출
3. **단계적 접근**: Phase 1 → 2 → 3 → 4 순차, 목표 달성까지 반복
4. **리스크 관리**: 각 Phase별 실패 시 pivot 전략 명확화
5. **1개만 제출**: 각 Phase당 최고 성능 1개만 Kaggle 제출
6. **과적합 체크**: CV vs Kaggle 성능 차이 모니터링
7. **지속 실험**: 17.395 미달성 시 새로운 가설 추가 (Phase 5, 6, ...)

---

## 다음 단계

1. ✅ HYPOTHESES.md 작성
2. ⏭️ `run_experiments.py` 작성 (Phase 1~4 실험 코드)
3. ⏭️ Phase 1 실행 (H1: k-grid search)
4. ⏭️ 결과 분석 및 Phase 2 진행 여부 결정
5. ⏭️ 최종 모델 선정 및 Kaggle 제출 (1회)

**내일 시작할 때**: 이 문서 + `run_experiments.py`로 충분
