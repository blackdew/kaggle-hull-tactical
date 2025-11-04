# EXP-025: Deep Learning with Residual Blocks & U-Net Architecture

**목표**: ResNet + U-Net 구조로 CV Sharpe 0.8+ 달성

**전략**: Tabular data를 2D로 재구성하여 U-Net 적용

---

## 핵심 아이디어

### 1. U-Net for Tabular Data
**원래 U-Net**: 이미지 segmentation
- Encoder: Downsampling (feature extraction)
- Decoder: Upsampling (reconstruction)
- Skip connections: Low-level + High-level features

**우리 적용**:
- Input: 305 features → Reshape to 2D
- Encoder: Feature hierarchy 학습
- Decoder: Prediction으로 압축
- Skip connections: Multi-scale features

### 2. Residual Blocks
**ResNet 핵심**:
```python
output = F(x) + x  # Skip connection
```

**장점**:
- Deep network 학습 가능
- Gradient vanishing 방지
- Feature reuse

---

## Architecture Design

### Option A: 1D U-Net with Category Grouping
```python
Input: (batch, 305)
→ Reshape: (batch, 305, 1)  # 1D signal

Encoder:
  Conv1D(305→256) + ResBlock  ↓
  Conv1D(256→128) + ResBlock  ↓
  Conv1D(128→64)  + ResBlock  ↓ (bottleneck)

Decoder:
  ConvTranspose1D(64→128) + Skip + ResBlock  ↑
  ConvTranspose1D(128→256) + Skip + ResBlock  ↑
  ConvTranspose1D(256→305) + Skip + ResBlock  ↑

Output: (batch, 1)  # Scalar prediction
```

### Option B: 2D U-Net with Feature Matrix
```python
Input: (batch, 305)
→ Reshape: (batch, 1, 20, 16)  # Pseudo 2D (category × features per category)

Encoder:
  Conv2D(1→32) + ResBlock   ↓
  Conv2D(32→64) + ResBlock  ↓
  Conv2D(64→128) + ResBlock ↓ (bottleneck)

Decoder:
  ConvTranspose2D(128→64) + Skip + ResBlock  ↑
  ConvTranspose2D(64→32) + Skip + ResBlock   ↑
  ConvTranspose2D(32→1) + Skip + ResBlock    ↑

GlobalPooling → FC(128→1)
```

### Option C: TabNet-inspired with Attention
```python
Input: (batch, 305)

Feature Transformer:
  FC(305→512) + BN + ReLU
  ResBlock(512) × 3
  Attention Mask (select important features)

Decision Steps (sequential):
  Step 1: Attention → FC → Decision
  Step 2: Attention → FC → Decision (with residual)
  ...
  Step N: Aggregate all decisions

Output: (batch, 1)
```

---

## Implementation Plan

### Phase 1: 1D ResNet Baseline
**가장 단순한 deep model**
- Input: (batch, 305)
- Multiple ResBlocks
- FC output

**목표**: Baseline deep learning performance

### Phase 2: 1D U-Net
**Encoder-Decoder with Skip Connections**
- 1D Convolutions
- Skip connections
- Feature hierarchy

**목표**: U-Net 효과 검증

### Phase 3: Feature Grouping 2D U-Net
**Category structure 활용**
- M/V/P/S/I/E를 spatial dimension으로
- 2D convolutions
- Category interaction 학습

**목표**: Feature structure 활용

### Phase 4: Best Model + Attention
**Attention mechanism 추가**
- Self-attention
- Feature importance
- Interpretability

**목표**: 최종 성능 극대화

---

## Expected Results

### Optimistic (20% 확률)
- CV Sharpe: 0.9~1.2
- Public Score: 8~12
- **평가**: DL breakthrough!

### Realistic (50% 확률)
- CV Sharpe: 0.7~0.9
- Public Score: 6~8
- **평가**: XGBoost 대비 개선

### Pessimistic (30% 확률)
- CV Sharpe: 0.5~0.7
- Public Score: 4~6
- **평가**: 실패 (EXP-014, 015 재현)

---

## 이전 DL 실패 분석

### EXP-014: LSTM (Sharpe 0.471)
**문제**:
- Temporal dimension 활용 못함 (InferenceServer 제약)
- 짧은 sequence
- Overfitting

### EXP-015: Transformer (Sharpe 0.257~0.299)
**문제**:
- 데이터 부족 (8990 samples)
- Complexity 과다
- Attention 효과 미미

### EXP-025 개선 방향
**차이점**:
1. ✅ **305 features 활용** (vs 94 in EXP-014/015)
2. ✅ **Spatial structure** (category grouping)
3. ✅ **Residual connections** (deeper network 가능)
4. ✅ **U-Net multi-scale** (다양한 feature level)
5. ✅ **Strong regularization** (dropout, weight decay)

---

## 리스크 & 완화

### Risk 1: Overfitting
**완화책**:
- Dropout: 0.3~0.5
- BatchNorm
- Weight decay
- Early stopping
- 5-fold CV

### Risk 2: InferenceServer 호환
**완화책**:
- 모든 연산 1-row 가능 확인
- PyTorch → ONNX export 테스트
- 또는 feature 추출 후 lightweight model

### Risk 3: 학습 불안정
**완화책**:
- Learning rate scheduling
- Gradient clipping
- Batch normalization

---

## 일정

**Day 1 (오늘)**
- [x] EXP-025 계획
- [ ] Phase 1: 1D ResNet baseline (1시간)
- [ ] Phase 2: 1D U-Net (1시간)

**Day 2**
- [ ] Phase 3: 2D U-Net with categories (2시간)
- [ ] Phase 4: Attention (1시간)
- [ ] Best model 선택 및 제출

**Total**: ~5시간

---

**Status**: 계획 완료
**Next**: Phase 1 - 1D ResNet baseline
