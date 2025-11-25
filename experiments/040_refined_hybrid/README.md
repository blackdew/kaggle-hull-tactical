# EXP-040: Refined Hybrid Ensemble

## 1. 개요
- **목표**: EXP-039(Hybrid Ensemble)의 과적합 문제를 해결하기 위해 Feature Selection을 적용.
- **가설**:
    - EXP-039의 실패 원인은 "너무 많은 Feature(51개) + 복잡한 모델(Ensemble)"의 조합.
    - Feature를 핵심적인 35개로 줄이면(Refinement), Ensemble의 일반화 성능이 향상될 것.
- **기간**: 2025-11-25

## 2. 방법론
1.  **Feature Selection (Phase 1)**:
    - EXP-038 v3 (Single XGBoost) 모델을 사용하여 Feature Importance 산출.
    - 상위 35개 Feature 선정 (하위 30% 제거).
2.  **Ensemble (Phase 2)**:
    - 선별된 35개 Feature로 XGBoost, LightGBM, CatBoost 학습.
    - Simple Average Ensemble 적용.

## 3. 결과
- **CV Sharpe**: 0.6023
    - EXP-039 (Full Ensemble): 0.6588
    - EXP-038 v3 (Single Hybrid): 0.7025
    - EXP-022 (Baseline Ensemble): 0.6368
- **Public Score**: **4.527** ⚠️
    - EXP-039 (3.736) 대비 **+21.2% 회복** (과적합 완화 확인).
    - 하지만 EXP-038 v3 (4.798) 및 EXP-022 (5.860)에는 미치지 못함.

## 4. 분석
- **성과**: Feature Selection(Refinement)이 과적합을 줄이는 데 효과적임을 입증했습니다.
- **한계**:
    - Hybrid Feature Set은 Ensemble 모델(LGB, Cat)과는 시너지가 약하거나, 여전히 Feature 수가 많을 수 있습니다.
    - 현재까지는 **"Hybrid Features + Single XGBoost" (EXP-038 v3)** 조합이 가장 효율적입니다.

## 5. 결론
- **절반의 성공 (Partial Success)**.
- 과적합 문제는 해결했으나, SOTA 경신에는 실패.
- 향후 방향: EXP-038 v3 (4.798)를 베이스로 하되, EXP-022의 Ensemble 전략을 아주 보수적으로(예: 가중치 조절) 적용하거나, Feature Engineering을 다시 원점(EXP-016)에서 검토할 필요가 있음.
