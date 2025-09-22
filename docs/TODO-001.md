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
- [ ] 공식 평가 메트릭 코드 확보 (Kaggle 노트북에서)
- [ ] 로컬 메트릭 vs 호스트 메트릭 일치성 테스트
- [ ] 120% 변동성 제약 수식 정확히 구현
- [ ] 변동성 제약 위반 시 페널티 테스트

### [ ] 1.3 성능 측정 파이프라인 (1일) - **검증**
- [ ] 시계열 분할 (Time Series Split) 적용
- [ ] 로컬 백테스팅으로 Modified Sharpe Ratio 계산
- [ ] 베이스라인 성능 기록 (개선 기준점)
- [ ] end-to-end 테스트: 데이터 → 모델 → 제출

## 📋 성공 기준 (CODEX 요구사항)

1. **사실 기반**: 모든 문서에서 추정치 제거 완료 ✅
2. **규정 준수**: kaggle_evaluation API 완벽 통합 ✅
3. **실행 가능성**: 베이스라인 모델 8시간 내 실행 [ ]
4. **검증 완료**: 로컬 메트릭 = 호스트 메트릭 일치성 [ ]

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

## 📁 핵심 산출물

### ✅ 완료된 산출물:
- `kaggle_baseline_model.py`: kaggle_evaluation API 통합 베이스라인
- `docs/PRD.md`: 사실 기반으로 업데이트된 요구사항 문서
- `data/train.csv`, `data/test.csv`: 실제 경진대회 데이터
- `kaggle_evaluation/`: 공식 평가 프레임워크

### 📋 다음 산출물:
- Modified Sharpe Ratio 계산 모듈
- 시계열 검증 파이프라인
- 성능 측정 대시보드
- 제출용 노트북

**핵심 메시지**: CODEX 지적처럼 "사실 우선, 규정 준수, 점진적 개선"