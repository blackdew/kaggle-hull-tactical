# EXP-011: Direct Utility Maximization Strategy

## 문제 재정의

### 기존 접근의 오류

**현재 방식:**
1. Excess return 예측 (MSE loss)
2. 예측값으로 position 계산
3. 희망: 좋은 예측 = 좋은 수익

**문제:**
- ❌ MSE가 낮다고 utility가 높은 게 아님
- ❌ k 파라미터가 임의적 (600이 최선이라는 보장 없음)
- ❌ **예측 정확도 ≠ 수익**

### 실제 문제

**목표 함수:**
```python
utility = min(max(sharpe, 0), 6) × Σ(profits)
```

**제약:**
- position ∈ [0, 2]
- Sharpe를 6에 가깝게 유지
- Profit을 최대화

**이것은 제약 조건이 있는 최적화 문제입니다!**

---

## 새로운 접근

### Approach 1: Reinforcement Learning (RL)

**핵심 아이디어:** Position 선택을 sequential decision making으로

```python
State: [features, current_position, portfolio_value, sharpe_so_far]
Action: position ∈ [0, 2]
Reward: utility_t (즉각적 수익 + Sharpe penalty/bonus)

Agent: Policy Network
  Input: state
  Output: position (continuous action)
```

**알고리즘:** PPO (Proximal Policy Optimization) 또는 DDPG (Deep Deterministic Policy Gradient)

**장점:**
- ✅ 직접 utility 최대화
- ✅ Sharpe와 profit을 동시에 고려
- ✅ Dynamic position sizing
- ✅ 장기 전략 학습 가능

**단점:**
- ⚠️ 구현 복잡
- ⚠️ 학습 시간 오래 걸림
- ⚠️ Hyperparameter 민감

---

### Approach 2: Differentiable Sharpe Optimization

**핵심 아이디어:** Sharpe를 미분 가능한 loss로 만들어 end-to-end 학습

```python
# Neural network: features → positions
model = PositionNetwork(input_dim=94, output_dim=1)

# Forward
positions = model(features)  # [batch, 1]
positions = sigmoid(positions) * 2  # [0, 2]

# Calculate strategy returns
strategy_returns = rf * (1 - positions) + fwd * positions
excess_returns = strategy_returns - rf

# Differentiable Sharpe
sharpe = mean(excess_returns) / (std(excess_returns) + eps) * sqrt(252)

# Loss: Negative utility
utility = min(sharpe, 6) * sum(excess_returns)
loss = -utility

# Backprop
loss.backward()
optimizer.step()
```

**장점:**
- ✅ 직접 utility 최대화
- ✅ End-to-end 학습
- ✅ 구현 상대적으로 간단

**단점:**
- ⚠️ Batch statistics 불안정
- ⚠️ Min/max 연산 미분 불가능 (approximation 필요)

---

### Approach 3: Kelly Criterion + Risk Management

**핵심 아이디어:** 켈리 기준으로 optimal position sizing

```python
# Kelly criterion
f* = (p * b - q) / b

where:
  p = P(win) = P(excess_return > 0)
  q = 1 - p
  b = average_win / average_loss

# Position sizing
if pred_return > 0:
    position = 1 + kelly_fraction * confidence
else:
    position = 1 - kelly_fraction * confidence

# Risk management
position = adjust_for_sharpe_constraint(position, current_sharpe, target_sharpe=6)
```

**장점:**
- ✅ 이론적 근거 (Kelly criterion은 장기 수익 최대화)
- ✅ Risk management 내장
- ✅ 구현 간단

**단점:**
- ⚠️ Win/loss 확률 추정 필요
- ⚠️ Sharpe 제약 조건 handling

---

### Approach 4: Multi-Objective Optimization

**핵심 아이디어:** Sharpe와 Profit을 별도 목표로 Pareto optimization

```python
# Objective 1: Maximize Sharpe
sharpe = mean(excess) / std(strategy_returns) * sqrt(252)

# Objective 2: Maximize Profit
profit = sum(excess)

# Multi-objective loss
loss = -alpha * sharpe - beta * profit

# Constraint: Sharpe should be close to 6
if sharpe > 6:
    loss += penalty * (sharpe - 6)^2
```

**장점:**
- ✅ 명시적으로 두 목표 균형
- ✅ Constraint handling 명확
- ✅ α, β로 조절 가능

---

## 추천: Approach 2 (Differentiable Sharpe)

**이유:**
1. **구현 가능** - PyTorch로 직접 구현 가능
2. **빠른 실험** - RL보다 간단
3. **직접 최적화** - Utility를 loss로 사용

**구현 계획:**

### Phase 1: Position Network (2시간)
```python
class PositionNetwork(nn.Module):
    def __init__(self, input_dim=94, hidden_dims=[128, 64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output [0, 1]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Output: [0, 2]
        return self.network(x) * 2
```

### Phase 2: Differentiable Sharpe Loss (1시간)
```python
def utility_loss(positions, rf, fwd, lambda_sharpe=1.0, target_sharpe=6.0):
    """
    Args:
        positions: [batch] tensor, values in [0, 2]
        rf: [batch] risk-free rate
        fwd: [batch] forward returns
    """
    # Strategy returns
    strat = rf * (1 - positions) + fwd * positions
    excess = strat - rf

    # Sharpe (differentiable)
    mean_excess = torch.mean(excess)
    std_excess = torch.std(excess) + 1e-6
    sharpe = (mean_excess / std_excess) * math.sqrt(252)

    # Clip Sharpe to [0, 6] (soft clipping)
    sharpe_clipped = torch.clamp(sharpe, 0, target_sharpe)

    # Profit
    profit = torch.sum(excess)

    # Utility
    utility = sharpe_clipped * profit

    # Loss = negative utility
    loss = -utility

    # Penalty if Sharpe too high (encourage staying near 6)
    if sharpe > target_sharpe:
        loss += lambda_sharpe * (sharpe - target_sharpe) ** 2

    return loss, sharpe, profit
```

### Phase 3: Training (1시간)
- TimeSeriesSplit (5-fold)
- Batch size = full fold (전체 fold를 한 번에)
- Epochs = 100
- Learning rate = 0.001

---

## 예상 결과

### Conservative
- Sharpe: 1.5~2.5 (현재 0.749 대비 2~3배)
- Utility: 직접 최적화하므로 더 높을 것

### Optimistic
- Sharpe: 3.0~4.5
- Utility: 5~10 (목표 달성 가능성!)

### Realistic
- Sharpe: 1.0~2.0
- **최소 목표: Sharpe 1.0+ (현재 대비 +33%)**

---

## 리스크

### 1. Batch Statistics 불안정
- Batch 내에서 mean/std 계산 → variance 큼
- **대응:** Full batch (전체 fold 사용)

### 2. Overfitting
- Directly optimize validation utility
- **대응:** Dropout, Early stopping, Validation split

### 3. Local Optima
- Non-convex optimization
- **대응:** Multiple random seeds, Ensemble

---

## Fallback Plan

만약 Approach 2 실패하면:
1. **Approach 3 (Kelly Criterion)** - 가장 간단
2. **Approach 4 (Multi-objective)** - Pareto front 탐색
3. **Approach 1 (RL)** - 최후의 수단

---

## 시작!

Approach 2 (Differentiable Sharpe Optimization) 구현 시작.

**This is the right way to approach this problem!**
