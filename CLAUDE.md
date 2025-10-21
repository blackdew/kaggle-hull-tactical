# CLAUDE.md

이 파일은 이 저장소에서 작업하는 Claude Code (claude.ai/code)를 위한 가이드입니다.

---

## 프로젝트 개요

Human-AI 협업으로 진행한 Kaggle Code Competition (Hull Tactical Market Prediction). 최종 성과: **Public Score 4.440** (6.1배 향상).

**핵심 제약**: InferenceServer는 **row-by-row 예측** 방식 - 단일 행에서 계산 가능한 features만 허용 (lag/rolling features 사용 불가).

---

## 개발 명령어

### 환경 설정
```bash
# 의존성 설치
uv sync
# 또는
pip install -r requirements.txt

# 가상환경 활성화
source .venv/bin/activate
```

### 데이터 다운로드
```bash
# Kaggle API 토큰 필요 (~/.kaggle/kaggle.json)
kaggle competitions download -c hull-tactical-market-prediction
unzip hull-tactical-market-prediction.zip -d data/
```

### 실험 실행
```bash
# EXP-016 (최고 성능)
cd experiments/016
python phase1_analyze_features.py   # Feature 선택
python phase2_feature_engineering.py # Interaction features 생성
python phase3_sharpe_evaluation.py   # Sharpe ratio 및 K 파라미터 평가

# 로컬 제출 테스트
cd ../../
python submissions/submission.py
```

---

## 아키텍처

### 실험 구조

```
experiments/
├── 000-007/          # 초기 실험 (baseline, feature eng, k-tuning)
├── 016/              # 최고 성능 (Public Score 4.440)
│   ├── phase1_*.py   # Top 20 원본 features 선택
│   ├── phase2_*.py   # 120 interaction features → Top 30 선택
│   ├── phase3_*.py   # K parameter 최적화 (K=250)
│   └── results/      # CSV 출력
└── CONCLUSION.md     # 전체 실험 요약
```

### 제출 흐름

1. **Feature 생성** (`submissions/submission.py`):
   - `create_features()`: 단일 행에서 interaction features 생성
   - 1 row 입력으로 작동해야 함 (InferenceServer 제약)

2. **예측**:
   - XGBoost 모델 (150 estimators, depth=7, lr=0.025)
   - StandardScaler로 feature 정규화
   - Position = clip(1.0 + excess_return × K, 0.0, 2.0) (K=250)

3. **InferenceServer**:
   - `train_if_needed()`: 첫 예측 시 lazy training
   - `predict(test_batch)`: Row-by-row 예측
   - pandas 및 polars DataFrame 모두 처리

### 핵심 설계 패턴

**Interaction Features** (핵심 혁신):
- 곱셈: `P8*S2`, `M4*V7` (비선형 관계)
- 나눗셈: `P8/P7`, `M4/S2` (상대적 변화)
- 다항식: `M4²`, `V13²` (비선형 패턴)
- **제약**: 모두 단일 행에서 계산 가능 (과거 데이터 불필요)

**평가**:
- TimeSeriesSplit 5-fold CV
- Sharpe ratio: `(mean(excess) / std(strategy)) × √252`
- Position clipping: [0.0, 2.0]

---

## 핵심 교훈

1. **InferenceServer 제약이 최우선**
   - Version 10-13 실패: lag/rolling features 사용
   - 해결책: 원본 + interaction features만
   - 항상 1-row 계산 가능 여부 확인

2. **Interaction Features > Feature 수량**
   - 754 features (EXP-007): CV Sharpe 0.749, Public 제출 안함
   - 30 features (EXP-016): CV Sharpe 0.559, **Public 4.440**
   - Quality > Quantity

3. **XGBoost 압도적 우위**
   - 모든 딥러닝 시도 (LSTM, Transformer) < 0.6 Sharpe
   - XGBoost가 일관되게 최고 성능
   - 작은 데이터셋 + 약한 신호 = 전통적 ML 승리

4. **필요시 완전 재설계**
   - 초기 EXP-016 (CV 1.001) InferenceServer 비호환으로 폐기
   - 처음부터 재설계 → 6.1배 향상
   - Sunk cost fallacy 회피

---

## 코드베이스 작업 방법

### 새 실험 추가

1. `experiments/0XX/` 디렉토리 생성
2. 가능하면 3단계 구조 따르기:
   - Phase 1: Feature 선택/분석
   - Phase 2: Feature engineering
   - Phase 3: 모델 평가
3. 결과를 `results/` 하위 디렉토리에 저장
4. 실험 README.md에 문서화

### 제출 테스트

```python
# 로컬 테스트 (submission.parquet 생성)
python submissions/submission.py

# 출력 검증
import pandas as pd
sub = pd.read_parquet('submission.parquet')
print(sub.shape, sub['prediction'].describe())
```

### 문서화

- 실험 결과 → `experiments/0XX/README.md`
- 일일 회고 → `docs/retrospectives/YYYY-MM-DD.md`
- 전체 결론 → `experiments/CONCLUSION.md`

---

## Feature 카테고리

Features는 유형별로 접두사 사용:
- **M**: Market 지표 (e.g., M4, M2)
- **V**: Volatility 측정 (e.g., V13, V7)
- **P**: Price 관련 (e.g., P8, P5)
- **S**: Sentiment (e.g., S2, S5)
- **I**: Interest rates (e.g., I2)
- **E**: Economic 지표 (e.g., E19, E12)

Top 20 base features: M4, V13, V7, P8, S2, I2, E19, S5, P5, P7, M2, V9, M3, P12, P10, V10, E12, P11, M12, S8

---

## Kaggle 제출

`submissions/submission.py`를 Kaggle Notebook에 업로드하고 실행. 스크립트는:
1. 환경 자동 감지 (notebook vs competition rerun)
2. 첫 예측 시 모델 학습
3. Notebook 모드에서 `submission.parquet` 생성
4. Competition 모드에서 InferenceServer로 예측 제공

**중요**: 로컬에서 먼저 테스트 - Kaggle에서 제출 실패 디버깅은 어려움.
