# Experiments 002 — Results

목표: EXP-000 분석 기반 가설(H1~H6)을 EXP-001 베이스라인 대비 검증.

공통 설정
- 데이터: data/train.csv, 검증: TimeSeriesSplit 5폴드
- 메트릭: Sharpe 근사(↑), vol_ratio=전략/시장(≤1.2 권장), MSE(excess)(↓)
- 베이스라인: OLS+표준화, 포지션 1 + excess×k, k=50

실험 요약(1차 실행)
- BASE: sharpe_mean=0.3831, vol_ratio_mean=1.2493
- H1_scale(M4,V13 스케일 보정): sharpe_mean=0.3780(≈), vol_ratio 1.2589(↑)
- H2_interact(S5×V13,S2×V7): sharpe_mean=0.3804(≈), vol_ratio 1.2460(≈)
- H4_k35(k=35): sharpe_mean=0.4081(↑), vol_ratio 1.1577(↓)
- H5_volaware(vol_cap=1.2): sharpe_mean=0.3831(≈), vol_ratio 1.1336(↓)

해석
- H4(k 튜닝): k=35에서 Sharpe 평균이 +0.025p 개선, vol_ratio가 1.16으로 제약에 근접하게 안정화 → 유효.
- H5(vol‑aware): 평균 Sharpe 유지하면서 vol_ratio를 1.13까지 낮춤 → 안정성 관점에서 긍정적.
- H1/H2: 현 설정에서는 유의 개선 없음. 스케일/상호작용 선택을 재탐색 필요(대상 피처/변환 강도/비율).

다음 단계(우선순위)
1) k 최적화(세분화: k=25~45 범위) + vol‑aware 동시 적용 → BEST(k)@cap
2) H3(D1/D2 중복) 비교 실험 추가(H3_d1_only/H3_none)
3) H6(결측 마스킹) + H1(스케일 보정) 조합 검증
4) 정규화 모델(Ridge/Lasso) 비교 및 Top‑N 피처 선별 후 LightGBM 스몰 튜닝

산출물
- 폴드별: experiments/002/results/*_folds.csv
- 요약: experiments/002/results/summary.csv
