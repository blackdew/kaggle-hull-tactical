# Experiments 004 — Position Scaling & Prediction Amplification

## 목표
EXP-003 제출 결과 분석에서 발견된 **예측값 과도한 보수성 문제**를 해결하여 점수를 대폭 향상시킵니다.

**문제**: 현재 제출 점수 0.444 vs 상위권 17.333 (약 40배 차이)

**원인**:
- 예측 excess return: -0.0004 ~ +0.0005 (거의 0)
- 실제 훈련 데이터: -0.041 ~ +0.041 (**100배 더 큼**)
- k=50 적용 시 포지션: 0.98~1.02 (시장 중립, ±2% 변동만)
- 상위권은 [0, 2] 범위 적극 활용 (레버리지/현금 전환)

## 핵심 전략

### Phase 1: 빠른 검증 (우선 실행)
1. **H3a: 공격적 k 파라미터** - k={200, 500, 1000} 테스트
2. **H2a: 예측값 스케일 증폭** - 훈련 데이터 분포에 맞춰 조정
3. **H1a: 직접 포지션 예측** - excess return 대신 [0,2] 직접 예측

### Phase 2: 심화
4. **H5a: k 앙상블** - 여러 k 값 조합
5. **H6a: 변동성 기반 k 조정** - 낮은 변동성 시 k↑

### Phase 3: 고급 (시간 여유 시)
6. 모델 앙상블, 비선형 모델 튜닝 등

## 실행 방법

```bash
# Phase 1 (우선 실행)
uv run python experiments/004/run_experiments.py --phase 1

# Phase 2
uv run python experiments/004/run_experiments.py --phase 2

# 전체 실행
uv run python experiments/004/run_experiments.py --all

# 개별 실험
uv run python experiments/004/run_experiments.py --only H3a_large_k200 H3a_large_k500

# 베이스라인만
uv run python experiments/004/run_experiments.py
```

## 출력

- **폴드별 상세**: `experiments/004/results/*_folds.csv`
- **요약**: `experiments/004/results/summary.csv`
- **보고서**: `experiments/004/REPORT.md` (실험 완료 후 생성)

## 기대 효과

**현재 (EXP-003)**:
- 포지션 평균: 0.998
- 포지션 범위: 0.98~1.02
- 점수: 0.444

**목표 (EXP-004)**:
- 포지션 평균: 0.8~1.2 (더 다이나믹)
- 포지션 범위: 0.3~1.8 (0~2 활용)
- 점수: 5.0+ (10배 이상 향상 목표)

## 문서

- **가설 상세**: `HYPOTHESES.md` (전체 가설 체크리스트)
- **실험 코드**: `run_experiments.py`
- **EXP-002 참조**: `../002/REPORT.md` (기존 최고 성능 Sharpe 0.60)

## 다음 단계

1. Phase 1 실행 → 결과 분석
2. 최고 성능 설정 선택
3. Phase 2/3에서 추가 개선
4. 최종 후보를 EXP-005로 패키징 또는 EXP-003 업데이트
5. 캐글 재제출
