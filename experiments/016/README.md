# EXP-016 v2: InferenceServer-Compatible Feature Engineering

**결과**: 🏆 **Public Score 4.440** (최고 기록!)
**제약**: InferenceServer 호환 (1 row씩 예측)
**접근**: 원본 features + interaction features (lag/rolling 제외)

---

## 🎉 최종 결과

- **Public Score**: **4.440** (Version 9의 0.724 대비 6.1배 향상!)
- **CV Sharpe**: 0.559 ± 0.362 (5-fold)
- **Features**: Top 30 (원본 20 + interaction 10)
- **Model**: XGBoost (n_estimators=150, max_depth=7, lr=0.025)
- **K parameter**: 250

---

## 🎯 핵심 전략

### 문제 인식
- 기존 EXP-016 백업: lag/rolling features 사용 → InferenceServer 불가
- Kaggle Code Competition: row-by-row 예측 필요
- **1 row에서 계산 가능한 features만 사용**

### 해결책
1. **원본 features 선택** (94개 → Top 20)
2. **Interaction features 생성** (120개 → Top 30 선택)
3. **XGBoost 최적화** + K parameter tuning

---

## 📋 실험 진행 (완료)

### Phase 1: 원본 Features 분석 ✅
- RandomForest importance + Correlation 분석
- Top 20 원본 features 선택
- 카테고리: M(4), V(3), P(5), S(3), I(1), E(2), 기타(2)

### Phase 2: Feature Engineering ✅
- Interaction features 생성 (곱셈, 나눗셈, 다항식)
- 120개 features 생성 → XGBoost로 Top 30 선택
- MSE 개선: 0.000137 → 0.000132

### Phase 3: Sharpe Evaluation ✅
- K parameter 최적화 (50~300 테스트)
- Best K=250, Sharpe=0.559
- 5-fold CV 평가 완료

### Phase 4: InferenceServer 구현 ✅
- submissions/submission.py 작성
- Kaggle 제출 성공
- **Public Score: 4.440**

---

## 🔑 성공 요인

1. **InferenceServer 호환** - 1 row 계산 가능 설계
2. **Interaction Features** - 원본 features 간 상호작용 포착
3. **Feature Selection** - 120개 중 Top 30만 사용
4. **K=250 최적화** - Position sizing 최적화
5. **완전 재설계** - 처음부터 다시 시작한 결정

---

## 📊 Top 30 Features

**Interaction Features (상위 10개):**
- P8*S2, M4*V7, P8/P7, V7*P7, M4/S2
- S2*S5, S5/P7, M4*P8, M4², V13²

**Base Features:**
- P5, M4, V13, V7, P8, S2, I2, E19, S5, P7
- M2, V9, M3, P12, P10, V10, E12, P11, M12, S8

---

## 📁 파일 구조

```
experiments/016/
├── README.md                          # 이 파일
├── REPORT.md                          # 간결한 회고
├── phase1_analyze_features.py         # Phase 1 코드
├── phase2_feature_engineering.py      # Phase 2 코드
├── phase3_sharpe_evaluation.py        # Phase 3 코드
└── results/
    ├── feature_ranking.csv
    ├── top_20_features.csv
    ├── top_30_with_interactions.csv
    ├── final_cv_results.csv
    └── final_config.csv
```

---

## 🚀 제출 정보

- **Kaggle Version**: 15
- **Date**: 2025-10-21
- **Public Score**: 4.440
- **Status**: SubmissionStatus.COMPLETE
