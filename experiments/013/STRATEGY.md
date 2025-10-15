# EXP-013: Chart Pattern Recognition - 추세 기반 매매

## 진짜 문제 정의

### 기존 접근 (완전히 틀림!)
```python
# 날짜별 독립 예측
for date in dates:
    pred = model.predict(features[date])
    position = 1 + pred * k
```

### 올바른 접근 (Chart Reading!)
```python
# 전체 시계열을 보고 패턴 인식
chart = prices[-60:]  # 최근 60일 차트

# 패턴 분석
if is_bottom_pattern(chart):  # 최저점 감지
    position = 2.0  # 매수 (All-in)
elif is_top_pattern(chart):   # 최고점 감지
    position = 0.0  # 매도 (All-out)
elif is_uptrend(chart):        # 상승 추세
    position = 1.5  # 매수 홀딩
elif is_downtrend(chart):      # 하락 추세
    position = 0.5  # 매도 홀딩
else:
    position = 1.0  # 중립
```

## 핵심 인사이트

### 1. 기술적 분석 지표들

**추세 지표:**
- Moving Average (MA): 5일, 20일, 60일
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)
- Trend line breakout

**모멘텀 지표:**
- RSI (Relative Strength Index): 과매수/과매도
- Stochastic Oscillator
- Rate of Change (ROC)

**변동성 지표:**
- Bollinger Bands: 상단/하단 돌파
- ATR (Average True Range)

**패턴 인식:**
- Double Bottom (W형) → 매수 신호
- Double Top (M형) → 매도 신호
- Head and Shoulders → 추세 전환
- Cup and Handle → 상승 시작

### 2. 매수/매도 타이밍

**최저점 (매수 타이밍):**
```python
def is_bottom(prices, idx):
    # 1. 최근 급락 후
    recent_drop = prices[idx-5:idx].pct_change().sum() < -0.05

    # 2. RSI 과매도
    rsi = calculate_rsi(prices[:idx])
    oversold = rsi < 30

    # 3. Bollinger Band 하단 돌파
    bb_lower = bollinger_lower(prices[:idx])
    below_bb = prices[idx] < bb_lower

    # 4. MACD golden cross
    macd_signal = macd_cross(prices[:idx])

    return recent_drop and (oversold or below_bb) and macd_signal
```

**최고점 (매도 타이밍):**
```python
def is_top(prices, idx):
    # 1. 최근 급등 후
    recent_surge = prices[idx-5:idx].pct_change().sum() > 0.05

    # 2. RSI 과매수
    rsi = calculate_rsi(prices[:idx])
    overbought = rsi > 70

    # 3. Bollinger Band 상단 돌파
    bb_upper = bollinger_upper(prices[:idx])
    above_bb = prices[idx] > bb_upper

    # 4. MACD death cross
    macd_signal = macd_cross_down(prices[:idx])

    return recent_surge and (overbought or above_bb) and macd_signal
```

### 3. CNN for Chart Pattern Recognition

**핵심 아이디어:** 차트 이미지처럼 시계열 패턴 학습

```python
# Input: [batch, seq_len=60, features=94]
# Output: action (0=sell, 1=hold, 2=buy)

class ChartPatternCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 1D CNN for time series
        self.conv1 = nn.Conv1d(94, 128, kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, stride=1)

        # Pattern recognition
        self.fc = nn.Sequential(
            nn.Linear(32 * remaining_len, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)  # 0=sell, 1=hold, 2=buy
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        action = self.fc(x)
        return action  # [batch, 3]
```

### 4. Reinforcement Learning with Chart States

```python
class TradingEnv:
    def __init__(self, prices, features):
        self.prices = prices
        self.features = features
        self.position = 1.0  # Start neutral
        self.idx = 60  # Start after 60 days for history

    def get_state(self):
        # State = chart of last 60 days
        return {
            'chart': self.prices[self.idx-60:self.idx],
            'features': self.features[self.idx-60:self.idx],
            'position': self.position,
            'trend': calculate_trend(self.prices[:self.idx])
        }

    def step(self, action):
        # action: 0=sell (pos=0), 1=hold (pos=1), 2=buy (pos=2)
        self.position = action

        # Calculate reward
        rf = self.risk_free[self.idx]
        fwd = self.forward_returns[self.idx]

        strategy_return = rf * (1 - self.position) + fwd * self.position
        reward = strategy_return - rf  # Excess return

        self.idx += 1
        done = (self.idx >= len(self.prices))

        return self.get_state(), reward, done
```

## 구현 계획

### Phase 1: Technical Indicators (1시간) ⭐⭐⭐⭐⭐

```python
def create_technical_features(df):
    prices = df['forward_returns'].cumsum()  # Reconstruct price chart

    # Moving Averages
    df['MA_5'] = prices.rolling(5).mean()
    df['MA_20'] = prices.rolling(20).mean()
    df['MA_60'] = prices.rolling(60).mean()

    # RSI
    df['RSI'] = calculate_rsi(prices, period=14)

    # MACD
    df['MACD'], df['MACD_signal'] = calculate_macd(prices)

    # Bollinger Bands
    df['BB_upper'], df['BB_lower'] = bollinger_bands(prices, period=20)

    # Trend indicators
    df['is_uptrend'] = (df['MA_5'] > df['MA_20']).astype(int)
    df['is_downtrend'] = (df['MA_5'] < df['MA_20']).astype(int)

    # Pattern signals
    df['oversold'] = (df['RSI'] < 30).astype(int)  # Buy signal
    df['overbought'] = (df['RSI'] > 70).astype(int)  # Sell signal

    return df
```

### Phase 2: Rule-Based Strategy (30분) ⭐⭐⭐⭐⭐

```python
def decide_position(row):
    """Rule-based trading decision."""

    # Strong buy signals
    if row['oversold'] and row['MA_5'] > row['MA_20']:
        return 2.0  # All-in buy

    # Strong sell signals
    if row['overbought'] and row['MA_5'] < row['MA_20']:
        return 0.0  # All-out sell

    # Uptrend
    if row['is_uptrend'] and row['RSI'] < 60:
        return 1.5  # Buy and hold

    # Downtrend
    if row['is_downtrend'] and row['RSI'] > 40:
        return 0.5  # Sell and hold

    # Neutral
    return 1.0
```

### Phase 3: CNN Chart Pattern Model (1.5시간) ⭐⭐⭐⭐

- 60일 window로 학습
- Conv1D로 패턴 추출
- Classification: sell/hold/buy

### Phase 4: Ensemble (30분) ⭐⭐⭐

- Rule-based + CNN + XGBoost
- Majority voting or weighted average

## 예상 결과

### Rule-Based만으로
- Sharpe: 2.0~3.5 (기술적 지표가 효과 있다면)
- 이유: 명확한 매수/매도 시그널

### CNN 추가
- Sharpe: 3.5~5.0 (패턴 학습이 잘 되면)
- 이유: 복잡한 패턴 인식

### 완벽한 실행
- Sharpe: 5.0~6.0
- Utility: 15~20+ ✅ **목표 달성!**

## 왜 이게 맞는가

1. **시장은 패턴을 반복함**
   - 과매수 → 조정
   - 과매도 → 반등
   - 추세는 지속됨

2. **타이밍이 전부**
   - 최저점 매수 → 큰 수익
   - 최고점 매도 → 손실 방지
   - 중간은 홀드

3. **Sharpe 6.0 달성 방법**
   - 확실한 신호만 거래 → 낮은 변동성
   - All-in/All-out → 높은 수익
   - 나머지는 neutral → 안정적

## 시작!

Phase 1 (Technical Indicators) 먼저 구현!
