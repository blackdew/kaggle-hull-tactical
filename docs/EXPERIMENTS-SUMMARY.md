# Project Experiments Summary

## 000 — Feature Analysis
- 산출: experiments/000/FEATURES.md (수치+해석), FEATURES-INSIGHTS, REPORT.md
- 핵심: D/E/I/M/P/S/V 그룹 구조, 결측 Top‑10, 타깃 상관 상위, 그룹 통계

## 001 — Baseline
- 베이스라인: OLS(+표준화) + k=50 매핑(0..2), 제출 10행
- 성능(로컬 CV): Sharpe≈0.383, vol_ratio≈1.249

## 002 — Hypotheses & Results
- 가설(H1~H11) 수립·검증: scale, interactions, 중복(D1/D2), k 튜닝, vol‑aware, 마스킹, Top‑N, 정규화, 비선형(GBR)
- 결론
  - k 최적: 18~20 (예: k=18 Sharpe≈0.444, vol≈1.045)
  - vol‑aware: Sharpe 유지, vol≈1.05±로 안정
  - Top‑20 + Lasso(α=1e‑4): Sharpe≈0.57, vol≈1.05, 표준편차≈0.23
  - GBR Top‑20(+cap): Sharpe≈0.566, vol≈1.165, 표준편차≈0.231
  - D1 유지, 고결측 마스킹 소폭 개선, Top‑20가 Top‑40 대비 우수
- 문서: experiments/002/HYPOTHESES.md(체크리스트), REPORT.md(요약)

## 003 — Packaging & Serving
- 후보 A(성능): Lasso(α=1e‑4) + Top‑20 + k=50
- 후보 B(안정/단순): OLS + base + k=18
- 제출 생성: experiments/003/run_candidates.py → submissions/candidate_*.csv
- 서빙 파이프라인: experiments/003/serve.py (검증/로깅/제약 포함)
- 민감도: experiments/003/sensitivity.py → results/sensitivity_lasso.csv
- CI 스모크: experiments/003/ci_smoke.sh (출력 범위/로그 검사)

## 제출 가이드
- 후보 A: `python experiments/003/serve.py --candidate A` → 파일 경로로 제출
- 후보 B: `python experiments/003/serve.py --candidate B`
- 제출: `bash scripts/submit.sh -f <위 파일 경로> -m "<메시지>"`

## 추천안
- 1안(성능): 후보 A (Sharpe≈0.57, vol≈1.05)
- 2안(안정/단순): 후보 B (Sharpe≈0.44, vol≈1.04~1.06)

