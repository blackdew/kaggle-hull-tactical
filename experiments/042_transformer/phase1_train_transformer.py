#!/usr/bin/env python3
"""
EXP-042 Phase 1: Train Time Series Transformer
Goal: Train a Transformer model on sequence data (past 30 days) to predict returns.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import os
import random

# Set seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

print("="*80)
print("EXP-042 Phase 1: Train Time Series Transformer")
print("="*80)

# Hyperparameters
SEQ_LEN = 30
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DROPOUT = 0.1

# 1. Load Data
print("\n[1] Loading data...")
train = pd.read_csv("data/train.csv")
y_excess = train['market_forward_excess_returns'].values
fwd_returns = train['forward_returns'].values
risk_free = train['risk_free_rate'].values

# 2. Prepare Features
print("\n[2] Preparing Features...")
# Use Top 20 Base Features
top_20 = [
    'M4', 'V13', 'V7', 'P8', 'S2', 'I2', 'E19', 'S5', 'P5', 'P7',
    'M2', 'V9', 'M3', 'P12', 'P10', 'V10', 'E12', 'P11', 'M12', 'S8'
]
X = train[top_20].fillna(train[top_20].median()).replace([np.inf, -np.inf], np.nan).fillna(0).values

print(f"   Features shape: {X.shape}")

# 3. Dataset Class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=30):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        # Sequence: [idx, idx+seq_len)
        # Target: idx+seq_len
        x_seq = self.X[idx : idx + self.seq_len]
        y_target = self.y[idx + self.seq_len]
        return x_seq, y_target

# 4. Model Definition
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        # src: [batch, seq_len, input_dim]
        x = self.input_proj(src) # [batch, seq_len, d_model]
        x = x + self.pos_encoder # Add positional encoding
        x = self.transformer_encoder(x) # [batch, seq_len, d_model]
        # Take the last time step
        x = x[:, -1, :] # [batch, d_model]
        output = self.decoder(x) # [batch, 1]
        return output.squeeze()

# 5. Train and Evaluate (CV)
print("\n[5] Evaluating Strategy (5-fold CV)...")

def calculate_sharpe(positions, fwd_returns, risk_free):
    positions = np.clip(positions, 0.0, 2.0)
    strategy_returns = risk_free * (1.0 - positions) + fwd_returns * positions
    excess_returns = strategy_returns - risk_free
    if np.std(strategy_returns) > 0:
        sharpe = (np.mean(excess_returns) / np.std(strategy_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0
    return sharpe

sharpes = []
tscv = TimeSeriesSplit(n_splits=5)
K = 250

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Using device: {device}")

for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
    print(f"\n   Fold {fold_idx}/5...")
    
    # Split Data
    # Note: For sequence data, we need to be careful with indices.
    # We simply split the raw data, then create Datasets.
    # To avoid look-ahead bias in validation, we should strictly separate.
    
    X_tr_raw, X_va_raw = X[tr_idx], X[va_idx]
    y_tr_raw, y_va_raw = y_excess[tr_idx], y_excess[va_idx]
    
    # Scale
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_raw)
    X_va_scaled = scaler.transform(X_va_raw)
    
    # Create Datasets
    train_dataset = TimeSeriesDataset(X_tr_scaled, y_tr_raw, SEQ_LEN)
    val_dataset = TimeSeriesDataset(X_va_scaled, y_va_raw, SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    model = TransformerModel(input_dim=len(top_20), d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Train Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # print(f"      Epoch {epoch+1}/{EPOCHS} Loss: {total_loss/len(train_loader):.6f}")

    # Predict
    model.eval()
    preds = []
    with torch.no_grad():
        for batch_X, _ in val_loader:
            batch_X = batch_X.to(device)
            output = model(batch_X)
            preds.extend(output.cpu().numpy())
    
    preds = np.array(preds)
    
    # Align predictions with validation targets
    # The dataset returns targets starting from index SEQ_LEN.
    # So predictions correspond to y_va_raw[SEQ_LEN:]
    # We need to align fwd_returns and risk_free similarly.
    
    valid_len = len(preds)
    
    # Strategy
    # Simple strategy: Position = 1 + Pred * K
    # (No quantile logic here for simplicity, just directional magnitude)
    
    positions = 1.0 + preds * K
    
    # Evaluate
    # Get corresponding fwd_returns and risk_free
    # va_idx starts at some index. 
    # val_dataset[0] corresponds to X_va_raw[0:SEQ_LEN] -> Target y_va_raw[SEQ_LEN]
    # So we need fwd_returns[va_idx[SEQ_LEN:]]
    
    va_fwd = fwd_returns[va_idx][SEQ_LEN:]
    va_rf = risk_free[va_idx][SEQ_LEN:]
    
    # Truncate positions if needed (should match)
    positions = positions[:len(va_fwd)]
    
    sharpe = calculate_sharpe(positions, va_fwd, va_rf)
    sharpes.append(sharpe)
    
    print(f"      Sharpe: {sharpe:.4f}")

# 6. Results
avg_sharpe = np.mean(sharpes)
std_sharpe = np.std(sharpes)

print("\n" + "="*80)
print("RESULTS:")
print("="*80)
print(f"EXP-042 (Transformer) CV Sharpe: {avg_sharpe:.4f} ± {std_sharpe:.4f}")
print(f"EXP-038 v3 (Hybrid Single) CV Sharpe: 0.7025")

improvement = ((avg_sharpe - 0.7025) / 0.7025) * 100
print(f"Improvement vs EXP-038 v3: {improvement:+.2f}%")

# Save results
results_df = pd.DataFrame({
    'fold': range(1, 6),
    'sharpe': sharpes
})
results_df.to_csv("experiments/042_transformer/results/cv_results.csv", index=False)
print("\nResults saved to experiments/042_transformer/results/")
