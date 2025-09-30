# Experiments 003 — Candidates Packaging

목표
- EXP-002 결과를 바탕으로 최종 후보 모델을 재현·서빙 관점에서 패키징.

후보
- 후보 A(성능): Lasso(α=1e-4) + Top-20(abs corr, train 기반) + k=50 매핑(0..2 클립)
- 후보 B(안정/단순): OLS + base features + k=18 매핑(0..2 클립)

사용 방법
- 후보 A 제출 생성: `python experiments/003/run_candidates.py --candidate A`
- 후보 B 제출 생성: `python experiments/003/run_candidates.py --candidate B`
- 출력: `experiments/003/submissions/candidate_*.csv`

운영/서빙(검증/로그/제약 포함)
- 후보 A 서빙: `python experiments/003/serve.py --candidate A` → `submissions/candidate_A_served.csv`
- 후보 B 서빙: `python experiments/003/serve.py --candidate B` → `submissions/candidate_B_served.csv`
- 로그: `experiments/003/logs/serve_*.json` (피처/스케일/검증 통계 기록)

유의사항
- vol-aware(제약 안정화)는 제출 단계에서 테스트 데이터의 시장 변동성이 불명이라 사전 스케일링 적용이 제한적입니다. k-매핑(클립)으로 보수적 포지션을 사용합니다.
- Top-20 선택은 훈련 구간 상관 기반으로 fold마다 약간 변동할 수 있습니다. 운영 시 고정 목록 또는 롤링 기준을 채택하세요.

검증 항목(자동)
- 필수 컬럼 존재 여부, NaN/Inf 비율, 예측치 [0,2] 범위 클립, train 기반 vol-aware 스케일 산출 및 적용

롤링 기준 고정 vs 동적
- 운영 기본(고정): Top‑N=20, α=1e‑4 고정. 재현성·안정성 유리(serve.py 기본값).
- 연구(동적): 롤링 윈도마다 Top‑N/α 재선정. 과적합·운영상 복잡도 증가 가능, 보고 목적에 한해 사용 권장.
- 변경 방법: `--top_n`, `--alpha` 인자 조정. 롤링 설정은 EXP‑002/rolling_eval.py 참조.

근거
- EXP-002/HYPOTHESES.md, REPORT.md 참조(Sharpe/vol/안정성 결과).
