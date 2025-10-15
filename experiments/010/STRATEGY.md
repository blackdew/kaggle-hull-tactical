# EXP-010: Temporal Attention Transformer

## 핵심 아이디어

### 기존 접근의 한계
- **Single-day prediction**: 각 날짜를 독립적으로 예측
- Manual lag features (lag1, lag5, lag10...) → 정보 손실
- 시계열의 **temporal dependency 활용 부족**

### 새로운 접근: Attention-based Temporal Modeling

```
Input Sequence (과거 N일)
[day t-N, day t-N+1, ..., day t-2, day t-1, day t]
  ↓
[Positional Encoding]
  ↓
[Multi-Head Self-Attention] ← 과거 날짜 간 관계 학습
  ↓
[Feed-Forward Network]
  ↓
[Output Head] → Predict excess_return for day t
```

**Attention의 장점:**
1. **Dynamic weighting**: 중요한 과거 날짜에 자동으로 높은 가중치
2. **Long-range dependency**: lag60도 직접 참조 가능
3. **Context-aware**: 시장 상황에 따라 다른 패턴 학습

## Architecture

### 1. Input Preparation
```python
# Sequence length = 20일
# 각 날짜: 97 features

Input shape: [batch_size, seq_len=20, feature_dim=97]
```

### 2. Model Structure
```python
class TemporalAttentionTransformer(nn.Module):
    def __init__(
        self,
        feature_dim=97,
        d_model=128,      # Transformer hidden dimension
        nhead=4,          # Number of attention heads
        num_layers=2,     # Number of Transformer blocks
        seq_len=20        # Lookback window
    ):
        # Embedding
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.3,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predict excess_return
        )

    def forward(self, x):
        # x: [batch, seq_len, feature_dim]

        # Project to d_model
        x = self.input_proj(x)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer expects [seq_len, batch, d_model]
        x = x.transpose(0, 1)

        # Apply Transformer
        x = self.transformer(x)  # [seq_len, batch, d_model]

        # Use last position for prediction
        x = x[-1, :, :]  # [batch, d_model]

        # Predict
        output = self.predictor(x)  # [batch, 1]

        return output
```

## 데이터 준비

### Sequence Construction
```python
# 각 샘플: 과거 20일 + 현재 날짜
# Target: 현재 날짜의 excess_return

for date_id in range(20, len(train)):
    # Input: day[date_id-20:date_id+1]의 features
    sequence = train.iloc[date_id-20:date_id+1][features].values
    # Target: day[date_id]의 excess_return
    target = train.iloc[date_id]['market_forward_excess_returns']
```

### Time Series Split
- Train/Val split을 **날짜 기준**으로
- Validation에서 미래 정보 절대 사용 안 함

## 예상 효과

### Why This Should Work Better

1. **Attention learns what matters**
   - 급등/급락 있었던 날짜에 높은 attention
   - 안정적인 날짜는 낮은 attention
   - Manual lag features보다 효율적

2. **Capture market dynamics**
   - Volatility regime changes
   - Momentum patterns
   - Mean reversion signals

3. **Flexible lookback**
   - Fixed lag (5, 10, 20) 대신
   - Dynamic weighting by attention

### 예상 Sharpe

**Conservative**: 0.9~1.2 (EXP-007 대비 +20~60%)
- Attention이 약간의 temporal pattern 발견

**Optimistic**: 1.5~2.0 (EXP-007 대비 +100~167%)
- Strong temporal dependency 존재 시

**Realistic**: 1.0~1.3
- **최소 목표: 1.0+ (EXP-007 0.749 대비 +33%)**

## Hyperparameters

### Grid Search
```python
seq_len: [10, 20, 30]
d_model: [64, 128, 256]
nhead: [2, 4, 8]
num_layers: [1, 2, 3]
```

### Best Guess (시작점)
- seq_len=20 (약 1개월)
- d_model=128
- nhead=4
- num_layers=2
- dropout=0.3
- lr=0.0001 (Adam)
- batch_size=64

## 구현 계획

### Phase 1: Basic Transformer (1시간)
- Sequence construction
- Positional encoding
- Single-head attention
- CV evaluation

### Phase 2: Multi-head + Optimization (1시간)
- Multi-head attention (nhead=4)
- Hyperparameter tuning
- k parameter optimization

### Phase 3: Ensemble (30분)
- Transformer + XGBoost ensemble
- Weighted by CV Sharpe

## Risk Mitigation

### Overfitting Prevention
1. **Dropout**: 0.3 in attention & FFN
2. **Early stopping**: patience=15
3. **L2 regularization**: weight_decay=0.01
4. **Small model**: d_model=128 (not 512)

### Data Leakage Prevention
- Strict time series split
- Sequence에서 미래 날짜 제외
- Cross-sectional features 사용 안 함

## Success Criteria

### Minimum (MUST)
- ✅ CV Sharpe 1.0+ (EXP-007 대비 +33%)

### Target (SHOULD)
- ✅ CV Sharpe 1.3+ (EXP-007 대비 +73%)

### Stretch (COULD)
- ✅ CV Sharpe 1.5+ (EXP-007 대비 +100%)

만약 Sharpe < 0.75면:
- Sequence length 조정
- Architecture simplification
- EXP-007로 회귀

## 시작!
Transformer 구현 시작.
