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

추가 실행(2차)
- H3_d1_only: 0.3831(≈), H3_none: 0.3703(↓) → D1만 유지 권장
- H6_missing_mask: 0.3865(+), vol_ratio 1.198(↓)
- H4_k20: 0.4362(↑), vol 1.058(↓↓); H4_k25: 0.4198(↑), vol 1.090(↓↓); H4_k30: 0.4155(↑), vol 1.123(↓)
- H4_k40: 0.3988(↑), vol 1.191(↓); H4_k45: 0.3900(≈), vol 1.222(↓)
- H7_k25_volaware: 0.4198(=k25), vol 1.067(↓); H7_k30_volaware: 0.4155(=k30), vol 1.088(↓); H7_k35_volaware: 0.4081(=k35), vol 1.106(↓)
- Ridge: 0.4008(↑), Lasso: 0.5589~0.6040(↑↑), vol≈1.00~1.10

우선순위 결론(실무 관점)
- 1안(성능 극대화): Lasso(α=1e‑4) + Top‑20 + vol‑aware → Sharpe≈0.57, vol≈1.05, 표준편차 낮음(≈0.23)
- 2안(안정성/단순성): OLS k=20 + vol‑aware → Sharpe≈0.44, vol≈1.05, 구현 단순·리스크 낮음
- 롤링(H10): 양쪽 모두 중후반 블록에서 강세, 전반 약세 → 체 regime 대응 위해 동적 가중/리밸런싱 검토

해석
- H4(k 튜닝): k=35에서 Sharpe 평균이 +0.025p 개선, vol_ratio가 1.16으로 제약에 근접하게 안정화 → 유효.
- H5(vol‑aware): 평균 Sharpe 유지하면서 vol_ratio를 1.13까지 낮춤 → 안정성 관점에서 긍정적.
- H1/H2: 현 설정에서는 유의 개선 없음. 스케일/상호작용 선택을 재탐색 필요(대상 피처/변환 강도/비율).

다음 단계(우선순위)
1) k 최적화: k=20가 현재 최선 → k=18~24 추가 탐색 + vol‑aware 결합 검증
2) D군: D1만 유지(중복 제거)로 고정, 파이프라인 반영
3) H6 마스킹 + 정규화(Lasso/Ridge) 조합 검증(과적합/안정성: 폴드 분산/롤링 분석)
4) LightGBM 소규모 실험(Top‑N, 제한적 트리 깊이)로 비선형 검토(환경 허용 시)

산출물
- 폴드별: experiments/002/results/*_folds.csv
- 요약: experiments/002/results/summary.csv
