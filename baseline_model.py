import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
print("데이터 로딩...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# 피처 선택 (결측치가 적은 피처들 우선)
feature_cols = []
for col in train_df.columns:
    if col not in ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']:
        missing_pct = train_df[col].isnull().sum() / len(train_df)
        if missing_pct < 0.5:  # 결측치 50% 미만인 피처만 사용
            feature_cols.append(col)

print(f"Selected {len(feature_cols)} features with <50% missing values")

# 타겟 변수
target_col = 'market_forward_excess_returns'

# 데이터 전처리
def preprocess_data(df, feature_cols, is_train=True):
    # 피처 선택
    features = df[feature_cols].copy()

    # 결측치 처리 (중앙값으로 채우기)
    features = features.fillna(features.median())

    # 무한대 값 처리
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)

    return features

# 훈련 데이터 전처리
X_train = preprocess_data(train_df, feature_cols, is_train=True)
y_train = train_df[target_col].values

print(f"Training features shape: {X_train.shape}")

# 시계열 교차 검증
tscv = TimeSeriesSplit(n_splits=5)
scores = []

print("\n시계열 교차 검증 진행...")
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # 스케일링
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    # 모델 훈련
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_tr_scaled, y_tr)

    # 예측
    y_pred = model.predict(X_val_scaled)
    mse = mean_squared_error(y_val, y_pred)
    scores.append(mse)

    print(f"Fold {fold+1} MSE: {mse:.6f}")

print(f"\nCV MSE: {np.mean(scores):.6f} (+/- {np.std(scores)*2:.6f})")

# 전체 데이터로 최종 모델 훈련
print("\n최종 모델 훈련...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

final_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
final_model.fit(X_train_scaled, y_train)

# 피처 중요도
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n상위 10개 중요 피처:")
print(feature_importance.head(10))

# 간단한 전략 백테스팅
print("\n전략 백테스팅...")
train_predictions = final_model.predict(X_train_scaled)

# 간단한 시그널 변환: 예측값 기반 포지션 결정
def predictions_to_positions(predictions, threshold=0.001):
    """예측값을 포지션으로 변환"""
    positions = np.where(predictions > threshold, 1.0, 0.0)  # 단순 이진 전략
    return positions

positions = predictions_to_positions(train_predictions)

# 전략 수익률 계산
strategy_returns = (
    train_df['risk_free_rate'].values * (1 - positions) +
    train_df['forward_returns'].values * positions
)

# 성과 지표 계산
excess_returns = strategy_returns - train_df['risk_free_rate'].values
mean_excess_return = np.mean(excess_returns)
volatility = np.std(strategy_returns)
sharpe_ratio = mean_excess_return / volatility * np.sqrt(252) if volatility > 0 else 0

market_excess_returns = train_df['market_forward_excess_returns'].values
market_volatility = np.std(train_df['forward_returns'].values)

print(f"전략 평균 초과수익률: {mean_excess_return:.6f}")
print(f"전략 변동성: {volatility:.6f}")
print(f"샤프 비율: {sharpe_ratio:.4f}")
print(f"시장 변동성: {market_volatility:.6f}")
print(f"변동성 비율: {volatility/market_volatility:.4f}")

print("\n베이스라인 모델 완료!")