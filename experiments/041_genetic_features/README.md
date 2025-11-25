# EXP-041: Genetic Feature Generation

## 1. 개요
- **목표**: 인간의 직관에 의존하는 Feature Engineering의 한계를 극복하기 위해 **Symbolic Regression (Genetic Programming)** 도입.
- **가설**:
    - 시장 데이터에는 인간이 발견하기 힘든 복잡한 비선형 관계가 존재함.
    - 유전 알고리즘이 이를 수식 형태로 찾아낼 수 있음.
- **기간**: 2025-11-25

## 2. 방법론
1.  **Genetic Discovery (Phase 1)**:
    - `gplearn.SymbolicTransformer` 사용.
    - Base Features (Top 20)를 재료로 사칙연산, log, sqrt 등의 함수 조합.
    - 20세대(Generations) 진화를 거쳐 상위 20개 Feature 발굴.
2.  **Evaluation (Phase 2)**:
    - 발굴된 Genetic Features + Base Features로 XGBoost 학습.
    - 5-fold CV 평가.

## 3. 결과
- **CV Sharpe**: 0.6672
    - EXP-038 v3 (Hybrid Single): 0.7025
    - EXP-040 (Refined Ensemble): 0.6023
    - EXP-022 (Baseline Ensemble): 0.6368
- **Public Score**: **3.251** ❌
    - 역대 최하위 기록.
    - EXP-039 (3.736)보다도 낮음.

## 4. 분석
- **실패 원인**: **과적합(Overfitting)**
    - CV(0.667)와 Public LB(3.251)의 괴리가 매우 큼.
    - Genetic Programming이 찾아낸 `add(sub(X16...))` 같은 복잡한 수식들이 Training Data의 노이즈에 과도하게 맞춰짐.
- **교훈**:
    - 금융 시계열 데이터에서는 **"복잡도(Complexity) = 독(Poison)"**.
    - EXP-040에서 Feature 수를 줄였을 때 점수가 올랐던 것과 정반대의 결과.

## 5. 결론
- **실패 (Failure)**.
- "공격적인 시도"는 실패로 돌아감.
- 다시 **"단순함(Simplicity)"**과 **"기본(Basics)"**으로 돌아가야 함.
- 향후 방향: EXP-038 v3 (4.798)를 베이스로 미세 튜닝하거나, EXP-022 (5.86) Baseline을 철저히 분석하여 "왜 그게 잘 됐는지"를 파고들어야 함.
