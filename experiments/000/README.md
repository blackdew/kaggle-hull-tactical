# Experiments 000 — Feature Analysis

## 목적
주어진 `data/train.csv`의 피처 품질과 타깃 상관구조를 빠르게 파악합니다. 결측률, 그룹별(D/E/I/M/P/S/V) 통계, 타깃(`forward_returns`, `market_forward_excess_returns`) 상관을 요약/시각화합니다.

## 실행
- 기본 실행: `python experiments/000/feature_analysis.py`
- 출력물:
  - `experiments/000/summary/feature_missing.csv`
  - `experiments/000/summary/feature_target_corr.csv`
  - `experiments/000/summary/group_stats.csv`
  - `experiments/000/plots/missing_top30.png`
  - `experiments/000/plots/corr_market_excess_top30.png`

## 해석 가이드
- 결측률 상위 피처를 우선적으로 제외/보간 전략 수립
- 그룹별 결측·상관 분포로 피처군 중요도 가늠
- 타깃 상관 상위 피처는 초기 베이스라인 후보, 다중공선성은 별도 점검
