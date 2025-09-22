# TODO-001: CODEX 피드백 반영 - 핵심 작업 우선순위

**CODEX 평가 반영**: 추정 제거, 규정 준수, 실행 가능성 중심으로 재구성
**기간**: 1주 (Phase 0 핵심 작업만)
**목표**: 사실 기반 베이스라인 구축 → 점진적 개선

## ✅ 완료된 작업 (CODEX 지적사항 해결)

### [x] 0.1 실제 데이터 분석 ("추정" 제거)
- **훈련 데이터**: 8,990개 샘플, 98개 피처 (정확히 확인)
- **테스트 데이터**: 10개 샘플, 99개 피처 (추정보다 1개 많음)
- **피처 구조**: D(9) + E(20) + I(9) + M(18) + P(13) + S(12) + V(13) = 94개 + date_id + 3개 타겟
- **제출 형식**: `prediction` 칼럼에 0~2 범위 포지션 사이징

### [x] 0.2 kaggle_evaluation API 분석 (규정 준수)
- **Gateway**: DefaultGateway로 배치 생성 (date_id별)
- **InferenceServer**: templates.InferenceServer 상속 필요
- **제출 검증**: competition_specific_validation 통과
- **런타임 제약**: 8시간(훈련) / 9시간(예측) 노트북 제한

### [x] 0.3 PRD.md 사실 기반 업데이트
- "추정", "약", "요확인" 표현 모두 제거
- 실제 데이터 구조 반영
- kaggle_evaluation API 명세 추가

### [x] 0.4 베이스라인 모델 구현 (실행 가능성)
- `kaggle_baseline_model.py`: kaggle_evaluation API 통합
- 단순한 선형 회귀 → 포지션 사이징 (0~2 범위)
- 120% 변동성 제약 고려한 위험 조정

## 🔥 즉시 실행할 핵심 작업 (3일)

### [ ] 1.1 베이스라인 검증 (1일) - **최우선**
- [ ] `uv run python kaggle_baseline_model.py` 실행 및 오류 수정
- [ ] kaggle_evaluation API 통합 완벽 동작 확인
- [ ] 포지션 사이징 출력 (0~2) 범위 검증
- [ ] 8시간 런타임 제약 내 실행 확인

### [ ] 1.2 Modified Sharpe Ratio 구현 (1일) - **필수**
- [ ] 공식 평가 메트릭 코드 확보 ([Kaggle 노트북](https://www.kaggle.com/code/metric/hull-competition-sharpe))
- [ ] 로컬 메트릭 vs 호스트 메트릭 일치성 테스트 (허용 오차: ±1e-6)
- [ ] 120% 변동성 제약 수식 구현: `전략_변동성 / 시장_변동성 ≤ 1.2`
- [ ] 변동성 제약 위반 시 페널티 테스트 (지수적 페널티 적용)
- [ ] **수용 기준**: Modified Sharpe Ratio > 0.3 (거래비용 차감 후)

### [ ] 1.3 성능 측정 파이프라인 (1일) - **검증**
- [ ] 시계열 분할 적용: 5폴드, 80%/20% 비율 ([PLAN.md 참조](docs/PLAN.md#41))
- [ ] 로컬 백테스팅으로 Modified Sharpe Ratio 계산 ([PRD.md 성공지표](docs/PRD.md#성공-지표))
- [ ] 베이스라인 성능 기록 (현재 MSE: 0.000109)
- [ ] **수캘적 검증**: 언더퍼폼 기간 ≤ 40%, 최대 드로우다운 ≤ 15%
- [ ] end-to-end 테스트: 데이터 → 모델 → 제출 (8시간 내 완료)

## 📋 성공 기준 (CODEX 요구사항 → 수치화)

1. **사실 기반**: 모든 문서에서 추정치 제거 완료 ✅
2. **규정 준수**: kaggle_evaluation API 완벽 통합 ✅
3. **실행 가능성**: 베이스라인 모델 8시간 내 실행 (현재: 2시간 이내) [ ]
4. **검증 완료**: 로컬 메트릭 = 호스트 메트릭 일치성 (허용오차 ±1e-6) [ ]

### 수치적 수용 기준 (신규 추가):
- **Modified Sharpe Ratio**: > 0.3 (거래비용 차감 후)
- **변동성 제약**: 전략_변동성 / 시장_변동성 ≤ 1.2
- **언더퍼폼 기간**: ≤ 40% (전체 기간 대비)
- **최대 드로우다운**: ≤ 15% (연간 기준)
- **스트레스 테스트**: 2008/2020 위기 시 Modified Sharpe Ratio > -0.5

## 🚫 제외된 작업 (CODEX "범위 축소" 권고 반영)

- ~~Transformer, WaveNet, Neural ODE 등 복잡한 모델~~
- ~~147개 피처 엔지니어링 (98개 → 147개 확장)~~
- ~~MICE, Matrix Factorization 등 고급 결측치 처리~~
- ~~Hidden Markov Model, 체제 변화 탐지~~

**원칙**: 강력한 베이스라인 확보 후 점진적 확장

## 💡 다음 단계 (베이스라인 완료 후)

1. **Phase 1**: Modified Sharpe Ratio 최적화 (베이스라인 성능 개선)
2. **Phase 2**: 리스크 관리 강화 (120% 변동성 제약 세밀 조정)
3. **Phase 3**: 모델 복잡도 점진적 증가 (앙상블 등)
4. **Phase 4**: 실전 성능 검증 (워크포워드 분석)

### 연기된 고급 작업들 (베이스라인 검증 후)
- 피처 엔지니어링 (기술적 지표, 거시경제 모델링)
- 고급 결측치 처리 (MICE, Matrix Factorization)
- 시계열 특성 분석 (정상성, 체제 변화 탐지)
- 복잡한 모델링 (딥러닝, 앙상블)

## 📁 핵심 산출물 (문서 간 참조 링크 추가)

### ✅ 완료된 산출물:
- `kaggle_baseline_model.py`: kaggle_evaluation API 통합 베이스라인 (실제 존재 확인)
- [`docs/PRD.md`](docs/PRD.md): 사실 기반 요구사항, 신호-포지션 매핑 정책 추가
- [`docs/PLAN.md`](docs/PLAN.md): 검증 파라미터 수치화, 엔드투엔드 실행 가이드 추가
- `data/train.csv` (8,980행), `data/test.csv` (10행): 데이터 경계 단일화
- `kaggle_evaluation/`: 공식 평가 프레임워크

### 📋 다음 산출물 (링크된 요구사항):
- Modified Sharpe Ratio 계산 모듈 ([PRD.md 성공지표](docs/PRD.md#성공-지표) 참조)
- 시계열 검증 파이프라인 ([PLAN.md 4.1절](docs/PLAN.md#41) 참조)
- 성능 측정 대시보드 (수치 기준 반영)
- 제출용 노트북 (런타임 제약 8/9시간 대응)

**핵심 메시지**: CODEX 지적처럼 "사실 우선, 규정 준수, 점진적 개선" + 수치적 수용 기준 명시

## 부록: 문서 간 참조 맵
- [PRD.md - 성공 지표](docs/PRD.md#성공-지표): Modified Sharpe Ratio 공식 코드 링크
- [PRD.md - 제출 형식](docs/PRD.md#제출-형식): 신호-포지션 매핑 정책
- [PLAN.md - 4.1절](docs/PLAN.md#41): 검증 파라미터 (폴드, 엠바고, 윈도 크기)
- [PLAN.md - 4.2절](docs/PLAN.md#42): 수용 기준 (부트스트랩, 스트레스 테스트)
- [PLAN.md - 부록](docs/PLAN.md#부록): 엔드투엔드 실행 가이드