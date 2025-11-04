# CLAUDE.md

이 파일은 이 저장소에서 작업하는 Claude Code를 위한 가이드입니다.

---

## 프로젝트 목표

**Kaggle Code Competition: Hull Tactical Market Prediction**
- 목표: Public Score 20점 이상 달성
- 평가 지표: Sharpe Ratio

---

## 핵심 제약사항

**InferenceServer row-by-row 예측 방식**
- 단일 행(1-row)에서 계산 가능한 features만 사용 가능
- lag features, rolling window features 사용 불가
- 모든 feature는 현재 행 데이터만으로 계산되어야 함

---

## 주요 도구 및 기술 스택

**머신러닝**
- XGBoost: 주 모델 (딥러닝보다 일관되게 우수한 성능)
- StandardScaler: Feature 정규화
- TimeSeriesSplit: 5-fold CV

**Feature Engineering**
- Interaction features: 곱셈, 나눗셈, 다항식 (2-way까지만)
- Base features: M(Market), V(Volatility), P(Price), S(Sentiment), I(Interest), E(Economic)

**평가 및 검증**
- CV Sharpe Ratio: 교차검증 성능
- Public Score: Kaggle 리더보드 점수
- CV-to-Public Ratio: Overfitting 탐지 (정상: 7-8x, 위험: 1-2x)

**제출 확인**
```bash
kaggle competitions submissions hull-tactical-market-prediction
```

---

## 실험 및 개선 프로세스

**실험 디렉토리 구조**
```
experiments/XXX/
├── phase1_*.py    # Feature 선택/분석
├── phase2_*.py    # Feature engineering
├── phase3_*.py    # 모델 평가
├── results/       # 실험 결과
└── README.md      # 실험 문서
```

**필수 작업 흐름**
1. `experiments/` 디렉토리에서 과거 실패/성공 사례 분석
2. 실패 원인 파악 및 개선 방향 도출
3. 새로운 가설 설계 및 실험 수행
4. CV 및 Public Score로 검증
5. 결과 문서화 (README.md)

**Overfitting 탐지**
- High CV + Low Public Score = Overfitting
- CV-to-Public Ratio 1-2x = 위험 신호
- 복잡도 증가 시 반드시 Public Score 확인 필요

**검증된 원칙**
- Simple > Complex (2-way interactions까지만 유효)
- Feature Quality > Quantity
- 과거 실험 결과를 참고하여 반복 방지

---

## 개발 명령어

**환경 설정**
```bash
uv sync  # 또는 pip install -r requirements.txt
source .venv/bin/activate
```

**데이터 다운로드**
```bash
kaggle competitions download -c hull-tactical-market-prediction
unzip hull-tactical-market-prediction.zip -d data/
```

**로컬 테스트**
```bash
python submissions/submission.py
```

---

## Position Sizing 공식

**기본 공식**
```python
position = clip(1.0 + excess_return_pred * K, 0.0, 2.0)
```

**구성 요소**
- `excess_return_pred`: 모델이 예측한 초과 수익률
- `K`: 민감도 파라미터 (과거 실험에서 250이 최적)
- `clip(0.0, 2.0)`: 포지션 크기 제한 (0배~2배)

---

## InferenceServer 구현

**submission.py 구조**
```python
class MyServer(InferenceServer):
    def train_if_needed(self):
        # 첫 예측 시 lazy training
        # train.csv 로드 및 모델 학습

    def create_features(self, df):
        # 1-row에서 계산 가능한 features 생성
        # Interaction features 포함

    def predict(self, test_batch):
        # Row-by-row 예측
        # Position 계산 및 반환
```

**제출 프로세스**
1. 로컬에서 테스트: `python submissions/submission.py`
2. `submission.parquet` 생성 확인
3. Kaggle Notebook에 업로드 및 실행
4. Public Score 확인: `kaggle competitions submissions ...`

---

## Feature 카테고리

**접두사 체계**
- M: Market 지표
- V: Volatility 측정
- P: Price 관련
- S: Sentiment
- I: Interest rates
- E: Economic 지표

**Interaction Feature 예시**
- 곱셈: `M4*V7`, `P8*S2`
- 나눗셈: `P8/P7`, `M4/S2`
- 다항식: `M4²`, `V13³`
