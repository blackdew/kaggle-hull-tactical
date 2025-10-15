# EXP-014: Multi-variate Time Series Analysis

## 진짜 돌파구!

### 기존 실수
```python
# ❌ 잘못: 종가 하나만 사용
df['price'] = df['forward_returns'].cumsum()
sequence = price[-60:]  # 1차원 시계열
```

### 올바른 접근
```python
# ✅ 맞음: 94개 feature 전체 사용
sequence = features[-60:, :]  # [60, 94]
# 60일간의 94개 feature 변화 패턴을 모두 보고 예측!
```

## 왜 94개 feature가 중요한가

### Feature Groups의 의미

**D1~D9 (Discrete):**
- 카테고리컬 정보 (0, 1, -1)
- 시장 상태, 이벤트, 조건
- D1이 0→1로 바뀔 때 특정 패턴

**E1~E20 (Economic):**
- 경제 지표
- date 1784부터 등장 (중요한 시점!)
- E features의 상승/하락이 시장 추세

**I1~I9 (Industry):**
- 산업별 지표
- 섹터 로테이션 신호

**M1~M18 (Market):**
- 시장 지표
- 변동성, 거래량, 심리
- M features가 급변하면 큰 움직임

**P1~P13 (Price):**
- 가격 관련 features
- 여기가 실제 "차트"!
- P features의 패턴이 핵심

**S1~S12 (Sentiment):**
- 시장 심리 지표
- 과매수/과매도 신호
- 추세 전환 예측

**V1~V13 (Volatility):**
- 변동성 지표
- 리스크 관리
- 높으면 조심, 낮으면 공격적

## Multi-variate Pattern Recognition

### 예시 1: 매수 신호
```
시점 t-5 ~ t:
- D4: 0 → 1 (상태 변화)
- E1~E5: 상승 추세 (경제 개선)
- P1~P5: 하락 후 반등 (가격 바닥)
- S1~S3: 극도로 낮음 (과매도)
- V1~V3: 하락 중 (변동성 감소)
- M1~M5: 거래량 증가 (매집)

→ 이 패턴 조합 = 강한 매수 신호!
→ Position = 2.0 (All-in)
```

### 예시 2: 매도 신호
```
시점 t-5 ~ t:
- D7: 1 → 0 (상태 변화)
- E10~E15: 둔화 (경제 둔화)
- P8~P13: 급등 후 횡보 (가격 천장)
- S8~S12: 극도로 높음 (과매수)
- V8~V13: 급증 (변동성 폭발)
- M10~M18: 거래량 감소 (매물 부족)

→ 이 패턴 조합 = 강한 매도 신호!
→ Position = 0.0 (All-out)
```

## Architecture

### LSTM Multi-variate Model

```python
class MultivariateLSTM(nn.Module):
    def __init__(self, input_dim=94, hidden_dim=256, num_layers=3):
        super().__init__()

        # LSTM for sequential pattern
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.3
        )

        # Attention over time steps
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)

        # Position prediction
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len=60, features=94]

        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch, 60, hidden_dim]

        # Attention (which time steps matter most?)
        attn_out, _ = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        attn_out = attn_out.transpose(0, 1)  # [batch, 60, hidden_dim]

        # Use last time step
        final_state = attn_out[:, -1, :]  # [batch, hidden_dim]

        # Predict excess return
        pred = self.predictor(final_state)

        return pred.squeeze(-1)
```

### Transformer Multi-variate Model

```python
class MultivarateTransformer(nn.Module):
    def __init__(self, input_dim=94, d_model=256, nhead=8, num_layers=4):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=60)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len=60, features=94]

        x = self.input_proj(x)  # [batch, 60, d_model]
        x = self.pos_encoding(x)

        x = self.transformer(x)  # [batch, 60, d_model]

        # Use last timestep
        x = x[:, -1, :]

        return self.predictor(x).squeeze(-1)
```

## 구현 계획

### Phase 1: LSTM Multi-variate (1.5시간) ⭐⭐⭐⭐⭐

- Input: [batch, 60, 94]
- LSTM + Attention
- Train with MSE loss
- k parameter tuning

### Phase 2: Transformer Multi-variate (1시간) ⭐⭐⭐⭐⭐

- Input: [batch, 60, 94]
- Multi-head attention
- Larger capacity (d_model=256)
- More layers (4-6)

### Phase 3: Ensemble (30분) ⭐⭐⭐⭐

- LSTM + Transformer + XGBoost (from EXP-007)
- Weighted average
- Confidence-based position sizing

## 예상 결과

### LSTM alone
- Sharpe: 1.5~3.0 (시계열 패턴 학습)

### Transformer alone
- Sharpe: 2.0~4.0 (더 강력한 패턴 인식)

### Ensemble
- Sharpe: 3.0~5.0
- Utility: 10~20+ ✅ **17점 달성 가능!**

## 왜 이게 작동할까

1. **94개 feature = 94개 차트를 동시에 봄**
   - 종가 하나가 아니라 모든 지표
   - P features는 가격, S는 심리, V는 변동성

2. **시계열 패턴 = 추세 인식**
   - 60일간의 변화 패턴
   - 상승/하락/횡보 인식
   - 전환점 포착

3. **Multi-variate = Feature 간 상호작용**
   - D가 바뀔 때 P가 어떻게 반응?
   - E가 상승할 때 S는?
   - V가 높을 때 M은?

4. **Attention = 중요한 시점 집중**
   - 급변하는 날에 높은 가중치
   - 안정적인 날은 무시
   - 패턴의 핵심 포착

## This is it!

**94개 feature 전체를 시계열로 보는 것**이 17점의 비밀입니다!

시작합니다!
