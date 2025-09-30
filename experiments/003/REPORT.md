# Experiments 003 — Candidates Report

## 개요
- 대상: EXP‑002 결론을 바탕으로 최종 후보 A/B 패키징, 롤링 안정성, 민감도 분석 실행 결과.

## 후보 모델
- 후보 A(성능): Lasso(α=1e‑4) + Top‑20(abs corr) + k=50 매핑
- 후보 B(안정/단순): OLS + base features + k=18 매핑

## 제출 생성
- A: `python experiments/003/run_candidates.py --candidate A` → submissions/candidate_A.csv
- B: `python experiments/003/run_candidates.py --candidate B` → submissions/candidate_B.csv

## 롤링 안정성 요약(참고: EXP‑002/H10)
- 결합 Lasso+Top‑20@cap: Sharpe [0.09, 0.08, 1.06, 0.80], vol≈1.00~1.14
- OLS k=20@cap: [0.09, 0.06, 1.06, 0.82], vol≈1.01~1.20
- 해석: 결합안이 Sharpe·안정성 모두 양호, OLS는 더 보수적 vol.

## 민감도(Lasso, CV 5‑fold)
- Grid: Top‑N={15,20,25}, α={5e‑4,1e‑4,1e‑3}, k=50
- 최고: Top‑20, α=1e‑4 → Sharpe≈0.5733, vol≈1.051
- 그 외:
  - Top‑15, α=1e‑4 → ≈0.5720, vol≈1.036
  - Top‑25, α=1e‑4 → ≈0.5665, vol≈1.068
- 파일: results/sensitivity_lasso.csv

## 결론(003)
- 추천 배포 후보
  - 1안: Lasso(α=1e‑4)+Top‑20+k=50 (필요 시 vol‑aware 조합)
  - 2안: OLS k=18 (보수적 제약 안정)

## 운영/서빙 파이프라인 요약
- 파일: experiments/003/serve.py
- 기능: 컬럼/값 검증, NaN/Inf 점검, [0,2] 클리핑, train 기반 vol‑aware 스케일 산출/적용, 로그 JSON 기록
- 기본 운영 기준(고정): Top‑N=20, α=1e‑4 고정(serve.py 기본값). 재현성·안정성 유리
- 동적 롤링은 보고/연구 목적에 한해 권장(운영 복잡도↑)

## CI 스모크
- 파일: experiments/003/ci_smoke.sh
- 검사: served 제출 2종(A/B) 생성→예측 범위 확인→로그 필수 키 존재 확인
- 실행 결과: All checks passed.

## 제출 준비
- 후보 A 생성: `python experiments/003/serve.py --candidate A`
- 후보 B 생성: `python experiments/003/serve.py --candidate B`
- 제출: `bash scripts/submit.sh -f experiments/003/submissions/candidate_A_served.csv -m "003 A served"`
