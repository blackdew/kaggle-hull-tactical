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

**템플릿 파일 사용**
- 새로운 제출 파일을 만들 때: `submissions/template_inference_server.py` 복사
- 템플릿에는 모든 필수 구조가 포함되어 있음
- TODO 주석 부분만 채워서 사용

**핵심 구조**
```python
class MyServer(InferenceServer):
    def __init__(self):
        # 필수: super().__init__(predict) 호출
        def predict(batch):
            return MyServer.predict(self, batch)
        super().__init__(predict)

    def _get_gateway_for_test(self, ...):
        # 필수: Gateway 반환
        return DefaultGateway(data_paths)

    def train_if_needed(self):
        # Lazy training 구현

    def create_features(self, df):
        # Polars → Pandas 변환 (필수)
        # Missing value 처리 (필수)
        # Feature engineering

    def predict(self, test_batch):
        # Batch unpacking (필수)
        # Polars → Pandas 변환 (필수)
        # return float(position[0])  # Scalar 반환 (필수)
```

**제출 파일 생성 시 자주 발생하는 에러 및 해결법**

1. **TypeError: Can't instantiate abstract class with abstract method _get_gateway_for_test**
   - 원인: `_get_gateway_for_test` 메서드 미구현
   - 해결: `def _get_gateway_for_test(self, ...): return DefaultGateway(data_paths)` 추가

2. **'DataFrame' object has no attribute 'fillna'**
   - 원인: Polars DataFrame을 Pandas로 변환 안함
   - 해결: `create_features()` 시작 부분에 `if pl is not None and isinstance(df, pl.DataFrame): df = df.to_pandas()` 추가

3. **Invalid prediction data type, received: numpy.ndarray**
   - 원인: predict()가 numpy array 반환
   - 해결: `return float(position[0])` - **반드시 scalar float 반환**

4. **Missing features in DataFrame**
   - 원인: test data에 일부 features 없을 수 있음
   - 해결: `for feat in features: if feat not in df.columns: df[feat] = 0.0`

5. **submission.parquet not created**
   - 원인: `if __name__ == '__main__'` 블록에서 Gateway 미실행
   - 해결: 위 템플릿의 main 블록 전체 복사

**제출 프로세스**
1. 로컬에서 테스트: `python submissions/submission.py` (선택, kaggle_evaluation 필요)
2. Kaggle Notebook에 코드 업로드
3. 노트북 실행 → `submission.parquet` 생성 확인
4. "Submit to Competition" 또는 수동 제출
5. Public Score 확인: `kaggle competitions submissions hull-tactical-market-prediction`

**체크리스트**
- [ ] `_get_gateway_for_test()` 메서드 구현
- [ ] `__init__`에서 `super().__init__(predict)` 호출
- [ ] `predict()` scalar float 반환 (`return float(position[0])`)
- [ ] Polars → Pandas 변환 (create_features, predict 모두)
- [ ] Missing features 처리 (0.0으로 채우기)
- [ ] `if __name__ == '__main__'` 블록 구현
- [ ] train.csv 경로 처리 (Kaggle/로컬 모두 대응)

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
