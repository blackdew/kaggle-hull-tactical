# Experiments 002 — Hypotheses & Plan

본 문서는 EXP-000 피처 분석 결과를 바탕으로 EXP-002에서 검증할 가설과 실험 설계를 정리합니다. 각 가설은 동일한 검증 체계(TimeSeriesSplit 5폴드, Sharpe 근사, vol_ratio)로 전/후 비교합니다.

공통 설정
- 데이터: `data/train.csv`
- 검증: TimeSeriesSplit(n_splits=5)
- 메트릭: Sharpe 근사(mean(excess)/std(strategy)×√252), MSE(excess), vol_ratio(≤1.2 권장)
- 베이스라인: EXP-001 OLS+표준화, 포지션 `1 + excess×k`(k=50)

가설 체크리스트(실험 ID)
- [x] H1 스케일 보정(M4,V13): winsorize/log/표준화로 MSR 개선
  - 결과: ≈ 유지 (sharpe_mean 0.3780 vs BASE 0.3831), vol_ratio 약간 악화(1.259)
  - 링크: results/H1_scale_folds.csv
- [x] H2 상호작용(S5×V13, S2×V7): 약신호 조합으로 MSR 개선
  - 결과: ≈ 유지 (0.3804), vol_ratio 유사(1.246)
  - 링크: results/H2_interact_folds.csv
- [ ] H3 중복 제거(D1/D2): D1만 vs 둘 다 vs 제거, 성능 유지 또는 개선
  - 계획: H3_d1_only, H3_none 실행 후 BASE 대비 ΔSharpe·vol_ratio 비교
- [x] H4 포지션 스케일 k 튜닝: k∈{20,35,50,70,90} 최적화로 MSR 개선
  - 결과(부분): k=35에서 개선(0.4081, +0.025p), vol_ratio 1.158로 안정
  - 링크: results/H4_k35_folds.csv
- [x] H5 vol‑aware 스케일: 사전 스케일링으로 vol_ratio≤1.2 보장 시 안정화
  - 결과: Sharpe 유지(0.3831), vol_ratio 개선(1.134)
  - 링크: results/H5_volaware_folds.csv
- [ ] H6 결측 마스킹: 고결측(E7,V10,M1,M13,M14,V9) is_nan_* 추가로 MSR 개선
  - 계획: BASE vs H6_missing_mask 전/후 ΔSharpe, vol_ratio 비교

출력
- 결과 요약: `experiments/002/results/summary.csv`(실험별 평균/중앙값)
- 폴드별: `experiments/002/results/<EXPID>_folds.csv`
- 보고서: `experiments/002/REPORT.md`

실행
- 전체: `python experiments/002/run_experiments.py --all`
- 선택: `python experiments/002/run_experiments.py --only H1 H4`

참고
- 요약: results/summary.csv, 상세 보고: REPORT.md
