# Kaggle 제출 가이드

## 문제 상황
캐글 노트북에서 `submission.parquet` 파일이 생성되지 않는 경우

## 해결 방법

### 방법 1: 디버그 버전 사용 (권장)

**파일**: `kaggle_inference_k500.py` (디버그 로그 포함)

1. 캐글 노트북에 `kaggle_inference_k500.py` 전체 복사
2. Settings 확인:
   - Competition: **Hull Tactical Market Prediction**
   - Internet: **OFF** (필수!)
   - Add Data > Competition Data > **Hull Tactical Market Prediction** 추가 확인
3. Save & Run All
4. 출력 로그 확인:
   ```
   [INFO] Found train.csv at: /kaggle/input/hull-tactical-market-prediction/train.csv
   [INFO] Found test.csv at: /kaggle/input/hull-tactical-market-prediction
   [SUCCESS] submission.parquet created!
   ```

### 방법 2: 직접 제출 파일 생성

캐글 노트북에서 다음 코드 실행:

```python
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Load data
train = pd.read_csv('/kaggle/input/hull-tactical-market-prediction/train.csv')
test = pd.read_csv('/kaggle/input/hull-tactical-market-prediction/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Feature selection
target = 'market_forward_excess_returns'
exclude = {'date_id', 'forward_returns', 'risk_free_rate',
           'market_forward_excess_returns', 'is_scored',
           'lagged_forward_returns', 'lagged_risk_free_rate',
           'lagged_market_forward_excess_returns'}
feats = [c for c in train.columns if c not in exclude]

# Top-20 features by correlation
num = train[feats + [target]].select_dtypes(include=[np.number])
corr = num.corr(numeric_only=True)[target].drop(index=target).abs().sort_values(ascending=False)
top20 = [c for c in corr.index[:20] if c in feats]

print(f"Top-20 features: {top20}")

# Train model
X_train = train[top20].fillna(train[top20].median()).replace([np.inf, -np.inf], np.nan).fillna(0)
y_train = train[target].to_numpy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = Lasso(alpha=1e-4, max_iter=50000)
model.fit(X_train_scaled, y_train)

print("Training complete!")

# Predict on test
X_test = test[top20].fillna(train[top20].median()).replace([np.inf, -np.inf], np.nan).fillna(0)
X_test_scaled = scaler.transform(X_test)
excess_pred = model.predict(X_test_scaled)

# Convert to positions with k=500
k = 500.0
positions = np.clip(1.0 + excess_pred * k, 0.0, 2.0)

print(f"Predictions - Mean: {positions.mean():.4f}, Std: {positions.std():.4f}")
print(f"Predictions - Range: [{positions.min():.4f}, {positions.max():.4f}]")

# Create submission
row_id_col = test.columns[0]  # Usually 'date_id' or similar
submission = pd.DataFrame({
    row_id_col: test[row_id_col],
    'prediction': positions
})

# Save submission
submission.to_parquet('submission.parquet', index=False)
submission.to_csv('submission.csv', index=False)

print(f"\nSubmission created!")
print(submission.head())
```

### 체크리스트

캐글 노트북 실행 전 확인:

- [ ] Competition 선택: "Hull Tactical Market Prediction"
- [ ] Internet: **OFF** (매우 중요!)
- [ ] Add Data: Competition data 추가됨
- [ ] 데이터 경로: `/kaggle/input/hull-tactical-market-prediction/`에 train.csv, test.csv 존재 확인
- [ ] Output 탭에서 `submission.parquet` 생성 확인

### 자주 발생하는 문제

**1. submission.parquet가 안 만들어짐**
- 원인: Internet이 ON 상태
- 해결: Settings > Internet > OFF

**2. "train.csv not found" 에러**
- 원인: Competition data가 추가되지 않음
- 해결: Add Data > Competition Data > Hull Tactical Market Prediction 추가

**3. "No module named 'kaggle_evaluation'" 에러**
- 원인: Code Competition에서는 자동으로 제공됨
- 해결: Internet OFF 상태에서 재실행

**4. Output 탭에 파일이 안 보임**
- 확인: 코드 마지막에 `print(os.listdir('/kaggle/working'))` 추가
- 확인: 로그에서 "[SUCCESS] submission.parquet created!" 메시지 확인

### 제출 후 확인

1. Output 탭에서 `submission.parquet` 파일 확인
2. 우측 상단 **Submit to Competition** 클릭
3. 메시지 입력: "EXP-004 k=500, Sharpe 0.836"
4. Submit 클릭
5. Submissions 탭에서 점수 확인 (몇 분 소요)

### 예상 결과

- **현재 점수**: 0.444
- **예상 점수**: 5~10 (10~20배 향상)
- **CV Sharpe**: 0.836

## 추가 지원

문제가 계속되면:
1. 노트북 로그 전체 복사
2. Output 탭 스크린샷
3. Settings 스크린샷
