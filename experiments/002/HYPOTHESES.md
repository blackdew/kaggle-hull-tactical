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
- [x] H3 중복 제거(D1/D2): D1만 vs 둘 다 vs 제거, 성능 유지 또는 개선
  - 결과: H3_d1_only≈BASE(0.3831), H3_none 하락(0.3703) → D1만 유지 권장
  - 링크: results/H3_d1_only_folds.csv, results/H3_none_folds.csv
- [x] H4 포지션 스케일 k 튜닝: k∈{20,35,50,70,90} 최적화로 MSR 개선
  - 결과(부분): k=35에서 개선(0.4081, +0.025p), vol_ratio 1.158로 안정
  - 링크: results/H4_k35_folds.csv
- [x] H5 vol‑aware 스케일: 사전 스케일링으로 vol_ratio≤1.2 보장 시 안정화
  - 결과: Sharpe 유지(0.3831), vol_ratio 개선(1.134)
  - 링크: results/H5_volaware_folds.csv
- [x] H6 결측 마스킹: 고결측(E7,V10,M1,M13,M14,V9) is_nan_* 추가로 MSR 개선
  - 결과: 소폭 개선(0.3865, +0.0034), vol_ratio 1.198로 안정화 경향
  - 링크: results/H6_missing_mask_folds.csv

신규 가설(추가)
- [x] H7 k‑튜닝+vol‑aware 결합: k=35와 vol_cap=1.2 동시 적용 시 안정적 개선
  - 근거: H4(k=35) 개선+H5(vol-aware) 안정화 → 결합 기대
  - 결과: 0.4081(=H4_k35 수준 유지), vol_ratio 1.106로 추가 안정
  - 링크: results/H7_k35_volaware_folds.csv

추가 가설(다음 단계)
- [x] H4b k 세분화(20~30): k∈{20,22,24,25,30}에서 최적 k 탐색 → BEST(k)
  - 결과: k=18(0.4436, vol 1.045) > k=20(0.4362, 1.058) > k=21(0.4318, 1.064) > k=22(0.4277, 1.070) > k=23(0.4241, 1.077) > k=24(0.4216, 1.083) > k=25(0.4198, 1.090) > k=30(0.4155, 1.123)
  - 링크: results/H4_k18_folds.csv, H4_k20_folds.csv, H4_k21_folds.csv, H4_k22_folds.csv, H4_k23_folds.csv, H4_k24_folds.csv, H4_k25_folds.csv, H4_k30_folds.csv
- [x] H7b k+vol‑aware 결합: {k∈{20,25,30}}×vol_cap=1.2에서 안정적 향상
  - 결과: k=20@cap 0.4362(=k20), vol 1.047; k=25@cap 0.4198(=k25), vol 1.067; k=30@cap 0.4155(=k30), vol 1.088
  - 링크: results/H7_k20_volaware_folds.csv, H7_k25_volaware_folds.csv, H7_k30_volaware_folds.csv
- [x] R_lasso 안정성: α∈{1e‑4, 1e‑3, 1e‑2}에서 일관 개선 여부와 vol_ratio 확인
  - 결과: α=1e‑4 0.6040(vol 1.097), α=1e‑3 0.5589(vol ≈1.000), α=1e‑2 0.5589(vol ≈1.000)
  - 링크: results/R_lasso_lo_folds.csv, R_lasso_folds.csv, R_lasso_hi_folds.csv
- [x] H6+정규화 결합: 결측 마스킹 + Ridge/Lasso가 단독 대비 추가 개선
  - 결과: H6mask_ridge 0.4039(↑), H6mask_lasso 0.5589(↑, vol ≈1.000)
  - 링크: results/H6mask_ridge_folds.csv, H6mask_lasso_folds.csv
- [ ] (보류) LightGBM 소규모: Top‑N 피처, 얕은 트리로 비선형 확인(환경 의존)
  - 설치 필요 시 후순위로 진행, 대안으로 트리기반 sklearn 모델 검토

신규 가설(추가)
- [x] H8 Top‑N(상관 절댓값) 특성 선택: N=20/40
  - 결과: Top‑20 0.5848(vol 1.080) > Top‑40 0.4972(vol 1.275)
  - 링크: results/H8_top20_folds.csv, H8_top40_folds.csv

안정성 가설
- [x] H9 모델 안정성(샤프 표준편차): Lasso vs k=20@cap 비교
  - 결과(상위): R_lasso_lo 0.6040(std 0.209); H8_top20 0.5848(std 0.235); H7_k20@cap 0.4362(std 0.350)
  - 롤링 안정성(H10): H7_lasso_lo_top20@cap 블록별 Sharpe [0.09, 0.08, 1.06, 0.80], vol≈1.00~1.14; H7_k20@cap [0.09, 0.06, 1.06, 0.82], vol≈1.01~1.20
  - 해석: 결합 모델이 Sharpe/vol 모두 양호하고, 롤링에서도 유사 안정성. k=20@cap은 vol이 더 보수적.
  - 링크: results/*_folds.csv, results/rolling_*.csv

추가 실행(H10 확장)
- R_lasso_lo 롤링: Sharpe [0.13, 0.14, 1.05, 0.82], vol≈1.03~1.17
- H8_top20 롤링: Sharpe [0.10, 0.07, 1.06, 0.82], vol≈0.98~1.28
  - 결론: Lasso/Top‑20/결합 모두 3번째 블록에서 강한 성과, 전·후반 약세 구간 존재(예상 가능한 체 regime).

보너스(정규화)
- [x] Ridge: 0.4008(↑), vol_ratio 1.246
  - 링크: results/R_ridge_folds.csv
- [x] Lasso: 0.5589(↑↑), vol_ratio ≈1.000
  - 링크: results/R_lasso_folds.csv

출력
- 결과 요약: `experiments/002/results/summary.csv`(실험별 평균/중앙값)
- 폴드별: `experiments/002/results/<EXPID>_folds.csv`
- 보고서: `experiments/002/REPORT.md`

실행
- 전체: `python experiments/002/run_experiments.py --all`
- 선택: `python experiments/002/run_experiments.py --only H1 H4`

참고
- 요약: results/summary.csv, 상세 보고: REPORT.md
