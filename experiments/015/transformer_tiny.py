#!/usr/bin/env python3
"""EXP-015 Tiny: Best config from test - 3-fold CV"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block with residual connections."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Attention with residual
        normalized = self.norm1(x)
        attn_out, _ = self.attention(normalized, normalized, normalized)
        x = x + self.dropout1(attn_out)  # ✅ Residual

        # FFN with residual
        normalized = self.norm2(x)
        ffn_out = self.ffn(normalized)
        x = x + ffn_out  # ✅ Residual
        return x


class TinyTransformer(nn.Module):
    def __init__(self, input_dim: int = 94, d_model: int = 64, nhead: int = 2, num_layers: int = 2, seq_len: int = 30):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_model * 2, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)  # ✅ Residuals inside each block
        x = self.final_norm(x)
        x = x[:, -1, :]
        return self.predictor(x).squeeze(-1)


class MultivarSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 30):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx:idx+self.seq_len]), torch.FloatTensor([self.y[idx+self.seq_len-1]])


def load_train(path: str = "data/train.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def select_all_features(df: pd.DataFrame) -> list[str]:
    exclude = {'date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns', 'is_scored'}
    return [c for c in df.columns if c not in exclude]


def preprocess(df: pd.DataFrame, features: list[str], scaler=None) -> Tuple[np.ndarray, StandardScaler]:
    X = df[features].copy().fillna(0).replace([np.inf, -np.inf], 0)
    if scaler is None:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = scaler.transform(X)
    return Xs, scaler


def train_model(model, train_loader, val_loader, epochs=30, lr=0.001, patience=8, device="cpu"):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.squeeze().to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.squeeze().to(device)
                val_loss += criterion(model(X_batch), y_batch).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stop at epoch {epoch+1}")
                break

    return model


def eval_fold(y_pred: np.ndarray, valid_df: pd.DataFrame, seq_len: int, k: float) -> Dict[str, float]:
    valid_df_eval = valid_df.iloc[seq_len-1:]
    rf = valid_df_eval["risk_free_rate"].to_numpy()
    fwd = valid_df_eval["forward_returns"].to_numpy()

    min_len = min(len(y_pred), len(rf))
    y_pred, rf, fwd = y_pred[:min_len], rf[:min_len], fwd[:min_len]

    positions = np.clip(1.0 + y_pred * k, 0.0, 2.0)
    strat = rf * (1.0 - positions) + fwd * positions
    excess = strat - rf

    strat_vol = float(np.std(strat))
    sharpe = (np.mean(excess) / strat_vol) * math.sqrt(252) if strat_vol > 0 else 0.0

    return {"sharpe": sharpe, "utility": min(max(sharpe, 0), 6.0) * float(np.sum(excess))}


if __name__ == "__main__":
    if not HAS_TORCH:
        print("[ERROR] PyTorch required")
        exit(1)

    print("Loading data...")
    train = load_train()
    print(f"Train: {train.shape}")

    all_features = select_all_features(train)
    print(f"Features: {len(all_features)}\n")

    print("Transformer + Residual (Tiny): d_model=64, heads=2, layers=2, seq=30")
    print("="*80)

    target = "market_forward_excess_returns"
    tscv = TimeSeriesSplit(n_splits=3)
    seq_len = 30
    k = 1000.0
    results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(train), 1):
        print(f"\n[Fold {fold_idx}/3]")
        tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]

        X_tr, scaler = preprocess(tr_df, all_features)
        y_tr = tr_df[target].to_numpy()
        X_va, _ = preprocess(va_df, all_features, scaler=scaler)
        y_va = va_df[target].to_numpy()

        train_loader = DataLoader(MultivarSequenceDataset(X_tr, y_tr, seq_len), batch_size=256, shuffle=True)
        val_loader = DataLoader(MultivarSequenceDataset(X_va, y_va, seq_len), batch_size=256, shuffle=False)

        print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

        model = TinyTransformer(input_dim=len(all_features), d_model=64, nhead=2, num_layers=2, seq_len=seq_len)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        model = train_model(model, train_loader, val_loader, epochs=30, lr=0.001, patience=8, device="cpu")

        model.eval()
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                all_preds.append(model(X_batch).cpu().numpy())

        y_pred = np.concatenate(all_preds)
        metrics = eval_fold(y_pred, va_df, seq_len, k)
        metrics.update({"fold": fold_idx, "k": k})
        results.append(metrics)

        print(f"  Sharpe: {metrics['sharpe']:.3f}, Utility: {metrics['utility']:.3f}")

    df_results = pd.DataFrame(results)
    avg_sharpe = df_results['sharpe'].mean()
    avg_utility = df_results['utility'].mean()

    print(f"\n{'='*80}")
    print("[RESULT]")
    print(f"  Avg Sharpe:  {avg_sharpe:.3f}")
    print(f"  Avg Utility: {avg_utility:.3f}")
    print(f"  Baseline:    0.749 (EXP-007)")
    print(f"  LSTM:        0.471 (EXP-014)")
    improvement = (avg_sharpe / 0.749 - 1) * 100
    print(f"  vs Baseline: {improvement:+.1f}%")
    improvement_vs_lstm = (avg_sharpe / 0.471 - 1) * 100
    print(f"  vs LSTM:     {improvement_vs_lstm:+.1f}%")

    if avg_sharpe > 3.0:
        print("\n🎉 BREAKTHROUGH! Sharpe > 3.0!")
    elif avg_sharpe > 1.0:
        print("\n✅ SUCCESS! Sharpe > 1.0!")
    elif avg_sharpe > 0.749:
        print("\n📈 IMPROVEMENT over baseline!")
    elif avg_sharpe > 0.471:
        print("\n⬆️ Better than LSTM!")

    df_results.to_csv(RESULTS / "transformer_tiny.csv", index=False)
    print("\n[DONE]")
