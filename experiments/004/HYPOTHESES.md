# Experiments 004 — Position Scaling & Prediction Amplification Hypotheses

본 문서는 EXP-002 결과 분석을 바탕으로 **예측값이 과도하게 보수적인 문제**를 해결하기 위한 새로운 가설을 정의합니다.

## 문제 진단 (EXP-003 제출 분석)

**현상**: 제출 점수 0.444 vs 상위권 17.333 (약 40배 차이)

**원인**:
- 현재 모델의 excess return 예측값: -0.0004 ~ +0.0005 (평균 ≈0)
- 훈련 데이터 실제 excess return 범위: -0.041 ~ +0.041 (**100배 더 큼**)
- k=50 적용 시 포지션: 0.98 ~ 1.02 (거의 시장 중립, ±2% 변동만)
- 상위권은 [0, 2] 범위를 적극 활용 (레버리지/현금 전환)

**EXP-002 핵심 결과 요약**:
- k 튜닝: k=18~20에서 Sharpe 최고 (0.44), k↓ 시 성능↑
- Lasso Top-20: Sharpe 0.57~0.60 (가장 높음), vol ≈1.00~1.10
- GBR Top-20: Sharpe 0.566, vol 1.165
- vol-aware: Sharpe 유지하며 vol_ratio 안정화 (≈1.05)

## EXP-004 가설 체크리스트

### 공통 설정
- 데이터: `data/train.csv`
- 검증: TimeSeriesSplit(n_splits=5)
- 메트릭: Sharpe 근사, vol_ratio(≤1.2), MSE(excess)
- 베이스라인: EXP-002 최고 성능 (Lasso Top-20, k=50)

### 핵심 전략: 포지션 다이내믹 확대

#### H1: 직접 포지션 예측 (Classification/Regression)
- [ ] **H1a_pos_direct**: excess return 대신 포지션 [0,2]를 직접 예측
  - 타깃 변환: `target_pos = clip(1.0 + excess × k_ref, 0, 2)` (k_ref=100)
  - 모델: Lasso/Ridge/GBR, Top-20 features
  - 기대: 포지션 범위 확대, 0~2 전체 활용
  - 링크: results/H1a_pos_direct_folds.csv

- [ ] **H1b_pos_class**: 포지션 3-class 분류 (0~0.7: cash, 0.7~1.3: neutral, 1.3~2.0: leverage)
  - 타깃 변환: 3-class binning → 확률 기반 포지션 산출
  - 모델: LogisticRegression/RandomForest, Top-20
  - 기대: 명확한 regime 구분, 극단적 포지션 활용
  - 링크: results/H1b_pos_class_folds.csv

#### H2: 예측값 스케일 증폭
- [ ] **H2a_pred_scaling**: 예측 excess return을 훈련 데이터 분포에 맞춰 스케일 조정
  - 변환: `y_pred_scaled = y_pred × (train_std / pred_std)`
  - 적용 후: `pos = clip(1.0 + y_pred_scaled × k, 0, 2)`
  - k 값: {50, 100, 200}
  - 기대: 예측값 변동성 증가, 포지션 다양화
  - 링크: results/H2a_pred_scaling_k50_folds.csv, ...

- [ ] **H2b_quantile_mapping**: 예측값을 훈련 데이터 분위수로 매핑
  - 변환: pred_quantile → train_quantile 매핑
  - 포지션: `pos = clip(1.0 + mapped_value × k, 0, 2)`
  - 기대: 분포 형태 보존, 극단값 활용
  - 링크: results/H2b_quantile_mapping_folds.csv

#### H3: 공격적 k 파라미터
- [ ] **H3a_large_k**: k 값 대폭 증가 ({200, 500, 1000, 2000})
  - 현재: k=50 → ±0.02 변동
  - 제안: k=1000 → 평균 0.967, 범위 0.596~1.464
  - 기대: 포지션 범위 확대, [0,2] 전체 활용
  - 링크: results/H3a_large_k200_folds.csv, H3a_large_k500_folds.csv, ...

- [ ] **H3b_adaptive_k**: 예측 신뢰도 기반 동적 k 조정
  - 신뢰도 높음(|excess_pred| 큼) → k↑ (공격적)
  - 신뢰도 낮음(|excess_pred| 작음) → k↓ (보수적)
  - 기대: 확신 있을 때만 극단 포지션
  - 링크: results/H3b_adaptive_k_folds.csv

#### H4: 타깃 엔지니어링
- [ ] **H4a_future_position**: 미래 최적 포지션을 타깃으로 역산
  - 타깃: `optimal_pos = argmax_p(reward)` where `reward = rf×(1-p) + fwd×p`
  - 단순화: `optimal_pos = 2 if fwd > rf else 0` (binary aggressive)
  - 모델: 분류 → 확률 기반 soft position
  - 기대: 실제 수익 최대화 포지션 학습
  - 링크: results/H4a_future_position_folds.csv

- [ ] **H4b_sharpe_target**: Sharpe 최대화 포지션을 타깃으로 설정
  - 타깃: rolling window 내 Sharpe 최대화하는 포지션 계산
  - 모델: Regression, Top-20
  - 기대: 메트릭에 직접 정렬된 학습
  - 링크: results/H4b_sharpe_target_folds.csv

#### H5: 앙상블 & 다양성
- [ ] **H5a_ensemble**: 여러 k 값 앙상블 ({50, 200, 500, 1000} 평균)
  - 각 k로 포지션 산출 → 가중 평균
  - 기대: 안정성 + 다양성 균형
  - 링크: results/H5a_ensemble_folds.csv

- [ ] **H5b_model_blend**: Lasso + GBR + Ridge 앙상블
  - 모델별 예측 → 가중 평균 포지션
  - 가중: Sharpe 기반 동적 조정
  - 기대: 비선형/선형 조합 효과
  - 링크: results/H5b_model_blend_folds.csv

#### H6: Volatility-Aware Amplification
- [ ] **H6a_vol_boost**: 낮은 변동성 구간에서 k 증폭
  - 변동성 낮음 → k↑ (여유 있을 때 공격)
  - 변동성 높음 → k↓ (안전 우선)
  - 기대: vol_ratio 제약 내에서 수익 극대화
  - 링크: results/H6a_vol_boost_folds.csv

- [ ] **H6b_inverse_vol_weight**: 포지션 크기를 변동성에 반비례 조정
  - `pos = clip(1.0 + excess_pred × k × (target_vol / current_vol), 0, 2)`
  - 기대: Kelly Criterion 유사 효과
  - 링크: results/H6b_inverse_vol_weight_folds.csv

### 보조 가설

#### H7: 피처 엔지니어링 심화
- [ ] **H7a_rolling_features**: 주요 피처 (M4, V13, S5, S2) 롤링 통계 추가
  - 5/10/20일 이동평균, 표준편차, 모멘텀
  - 기대: 트렌드/변동성 신호 강화
  - 링크: results/H7a_rolling_features_folds.csv

- [ ] **H7b_regime_features**: 시장 regime 감지 피처 추가
  - 변동성 상태 (high/low), 트렌드 방향
  - 모델: regime별 별도 k 또는 모델 선택
  - 기대: 시장 상황 맞춤 전략
  - 링크: results/H7b_regime_features_folds.csv

#### H8: 비선형 모델 확장
- [ ] **H8a_neural_net**: 간단한 MLP (2-3 hidden layers)
  - Top-20 features, ReLU activation
  - 출력: 직접 포지션 또는 증폭된 excess
  - 기대: 복잡한 비선형 패턴 학습
  - 링크: results/H8a_neural_net_folds.csv

- [ ] **H8b_gbr_tuning**: GBR 하이퍼파라미터 튜닝
  - depth, learning_rate, n_estimators 그리드 서치
  - 타깃: 직접 포지션 또는 증폭 excess
  - 기대: EXP-002 GBR 성능 개선
  - 링크: results/H8b_gbr_tuning_folds.csv

## 우선순위 실행 계획

### Phase 1 (핵심, 빠른 검증)
1. **H3a_large_k**: k={200, 500, 1000} 테스트 (가장 간단, 즉시 적용 가능)
2. **H2a_pred_scaling**: 예측값 스케일 조정 (구현 간단, 효과 기대)
3. **H1a_pos_direct**: 직접 포지션 예측 (근본적 접근)

### Phase 2 (심화)
4. **H5a_ensemble**: k 앙상블 (Phase 1 결과 조합)
5. **H6a_vol_boost**: 변동성 기반 k 조정
6. **H7a_rolling_features**: 롤링 피처 추가

### Phase 3 (고급, 시간 여유 시)
7. **H8b_gbr_tuning**: GBR 튜닝
8. **H5b_model_blend**: 모델 앙상블
9. **H4a_future_position**: 최적 포지션 타깃

## 출력
- 결과 요약: `experiments/004/results/summary.csv`
- 폴드별: `experiments/004/results/<EXPID>_folds.csv`
- 보고서: `experiments/004/REPORT.md`

## 실행
```bash
# 전체 실행
python experiments/004/run_experiments.py --all

# Phase별 실행
python experiments/004/run_experiments.py --phase 1
python experiments/004/run_experiments.py --phase 2

# 개별 실험
python experiments/004/run_experiments.py --only H3a_large_k200 H3a_large_k500
```

## 참고
- EXP-000: 피처 분석 (FEATURES.md)
- EXP-002: 기존 가설 검증 (HYPOTHESES.md, REPORT.md)
- EXP-003: 후보 패키징 (README.md)
