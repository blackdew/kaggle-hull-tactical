# Feature Analysis Report (EXP-000)

## 데이터 개요
- 학습 데이터: 8,990행 × 98피처 (`data/train.csv`)
- 타깃: `market_forward_excess_returns`(주) · 보조: `forward_returns`, `risk_free_rate`
- 피처 그룹: D/E/I/M/P/S/V (날짜/이벤트/지수/거시/가격/센티먼트/변동성)

## 결측 현황 요약
- 전체 결측률: 평균 0.156, 중앙값 0.112
- 그룹별(평균 결측률): M 0.255, S 0.202, V 0.198, E 0.153, P 0.127, I 0.112, D 0.000
- 결측 Top 10
  - E7 0.775, V10 0.673, S3 0.638, M1 0.617, M13 0.616
  - M14 0.616, M6 0.561, V9 0.505, S12 0.393, M5 0.365
- 조치 제안: 상위 결측 피처는 보간전략(중앙값/시계열 보간) 또는 제외 검토, 그룹 M/S/V 우선 점검

## 타깃 상관(피어슨, 절댓값 상위 10)
- 대상: `market_forward_excess_returns`
  - M4 -0.066, V13 0.062, M1 0.046, S5 0.040, S2 -0.038
  - D1 0.034, D2 0.034, M2 0.033, V10 0.033, E7 -0.032
- 참고: `forward_returns`도 유사 패턴(M4/V13/M1 등) 확인
- 해석 주의: 결측·스케일·공선성 영향 가능. 상관은 신호 강도 힌트로만 활용

## 산출물 경로
- 요약 CSV: `experiments/000/summary/`
  - `feature_missing.csv`, `feature_target_corr.csv`, `group_stats.csv`
- 플롯: `experiments/000/plots/`
  - `missing_top30.png`, `corr_market_forward_excess_returns_top30.png`

## 인사이트 & 다음 단계(002 후보)
- 데이터 품질: M/S/V 그룹 결측이 높아 단순 중앙값 대치 외 대안(MICE/시계열 보간) 평가 필요
- 피처 선별: M4, V13, M1, S5, S2 등 상위 상관 피처군 우선 관찰(분포/안정성/누수 여부)
- 스케일·제약: 120% 변동성 캡을 염두에 포지션 스케일 사전 캘리브레이션 도입 검토
- 검증 강화: Walk-Forward + Embargo로 일관성 확인, 그룹별(예: V만, M만) Ablation 비교
