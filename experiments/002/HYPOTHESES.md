# Experiments 002 — Hypotheses & Plan

본 문서는 EXP-000 피처 분석 결과를 바탕으로 EXP-002에서 검증할 가설과 실험 설계를 정리합니다. 각 가설은 동일한 검증 체계(TimeSeriesSplit 5폴드, Sharpe 근사, vol_ratio)로 전/후 비교합니다.

공통 설정
- 데이터: `data/train.csv`
- 검증: TimeSeriesSplit(n_splits=5)
- 메트릭: Sharpe 근사(mean(excess)/std(strategy)×√252), MSE(excess), vol_ratio(≤1.2 권장)
- 베이스라인: EXP-001 OLS+표준화, 포지션 `1 + excess×k`(k=50)

가설 목록(실험 ID)
- H1: 스케일 보정(M4, V13) — winsorize/log/표준화로 MSR 개선
- H2: 상호작용(S5×V13, S2×V7) — 약신호 조합으로 MSR 개선
- H3: 중복 제거(D1/D2) — D1만 vs 둘 다 vs 제거, 성능 유지 또는 개선
- H4: 포지션 스케일 k 튜닝 — k∈{20,35,50,70,90} 최적화로 MSR 개선
- H5: vol‑aware 스케일 — 사전 스케일링으로 vol_ratio≤1.2 보장 시 안정화
- H6: 결측 마스킹 — 고결측(E7,V10,M1,M13,M14,V9) is_nan_* 추가로 MSR 개선

출력
- 결과 요약: `experiments/002/results/summary.csv`(실험별 평균/중앙값)
- 폴드별: `experiments/002/results/<EXPID>_folds.csv`
- 보고서: `experiments/002/REPORT.md`

실행
- 전체: `python experiments/002/run_experiments.py --all`
- 선택: `python experiments/002/run_experiments.py --only H1 H4`
