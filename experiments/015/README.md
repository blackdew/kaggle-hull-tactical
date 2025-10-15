# EXP-015: Transformer with Residual Connections

## 목표

LSTM(EXP-014)보다 나은 성능을 위해 Transformer + Residual Connections 적용

## 구현

### Architecture

- **Pre-LayerNorm Transformer**: 깊은 네트워크에 적합
- **Residual Connections**: Attention 및 FFN 후 명시적 residual
- **GELU Activation**: ReLU 대신 사용
- **AdamW Optimizer**: weight_decay=0.01
- **Gradient Clipping**: max_norm=1.0
- **Multi-variate Input**: 94개 feature × sequence length

### Models

1. **transformer_tiny.py** (최종 버전)
   - d_model=64, 2 heads, 2 layers, seq=30
   - Parameters: 75,265
   - Avg Sharpe: 0.257

2. **transformer_medium.py**
   - d_model=96, 3 heads, 2 layers, seq=35
   - Parameters: 200,641
   - Avg Sharpe: 0.299

## 결과

| Model | Sharpe | vs Baseline | vs LSTM |
|-------|--------|-------------|---------|
| **Baseline (EXP-007)** | 0.749 | - | - |
| **LSTM (EXP-014)** | 0.471 | -37.1% | - |
| **Transformer Tiny** | 0.257 | -65.7% | -45.4% |
| **Transformer Medium** | 0.299 | -60.0% | -36.5% |

**결론: 실패 ❌**

## 실패 원인

1. **데이터 부족**
   - Fold 1: 2,220 samples로 75K+ parameters 학습
   - Transformer는 LSTM보다 훨씬 많은 데이터 필요

2. **Fold별 성능 격차** (Transformer Tiny)
   - Fold 1: -0.029 ❌
   - Fold 2: 0.229
   - Fold 3: 0.572

3. **짧은 Sequence**
   - 30-40일은 Attention의 장점 발휘하기 어려움
   - LSTM의 recurrent inductive bias가 더 적합

## 교훈

1. **More parameters ≠ Better**: 200K < 75K
2. **Architecture matters**: Transformer가 항상 최고는 아님
3. **Data efficiency**: XGBoost가 여전히 가장 강력 (0.749)

## 다음 방향

- 94개 feature를 모두 시계열로 보는 접근 자체를 재고
- Feature group별 다른 전략 필요
- XGBoost 기반 앙상블 강화

상세 결과: [RESULT.md](RESULT.md)
