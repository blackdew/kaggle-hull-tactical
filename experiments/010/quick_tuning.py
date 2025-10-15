#!/usr/bin/env python3
"""Quick tuning - Test most promising configurations"""
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


class SimpleTemporalTransformer(nn.Module):
    def __init__(self, feature_dim: int, d_model: int, nhead: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.predictor(x)


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx:idx+self.seq_len]), torch.FloatTensor([self.y[idx+self.seq_len-1]])


def load_train(path: str = "data/train.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def select_base_features(df: pd.DataFrame) -> list[str]:
    exclude = {
        "date_id", "forward_returns", "risk_free_rate",
        "market_forward_excess_returns", "is_scored",
        "lagged_forward_returns", "lagged_risk_free_rate",
        "lagged_market_forward_excess_returns"
    }
    return [c for c in df.columns if c not in exclude]


def preprocess(df: pd.DataFrame, features: list[str], scaler=None) -> Tuple[np.ndarray, StandardScaler]:
    X = df[features].copy()
    X = X.fillna(X.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
    if scaler is None:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = scaler.transform(X)
    return Xs, scaler


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, patience=8, device="cpu"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_loss += criterion(model(X_batch.to(device)), y_batch.to(device)).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model


def eval_fold(y_pred: np.ndarray, valid_df: pd.DataFrame, seq_len: int, k: float) -> float:
    valid_df_eval = valid_df.iloc[seq_len-1:]
    rf = valid_df_eval["risk_free_rate"].to_numpy()
    fwd = valid_df_eval["forward_returns"].to_numpy()

    min_len = min(len(y_pred), len(rf))
    y_pred, rf, fwd = y_pred[:min_len], rf[:min_len], fwd[:min_len]

    positions = np.clip(1.0 + y_pred * k, 0.0, 2.0)
    strat = rf * (1.0 - positions) + fwd * positions
    excess = strat - rf

    strat_vol = float(np.std(strat))
    sharpe = (np.mean(excess) / strat_vol) * np.sqrt(252) if strat_vol > 0 else 0.0
    return sharpe


def test_config(train, seq_len, d_model, nhead, num_layers, k, base_features):
    """Test single configuration with 3-fold CV."""
    device = "cpu"
    tscv = TimeSeriesSplit(n_splits=3)
    sharpes = []

    for tr_idx, va_idx in tscv.split(train):
        tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]

        X_tr, scaler = preprocess(tr_df, base_features)
        y_tr = tr_df["market_forward_excess_returns"].to_numpy()
        X_va, _ = preprocess(va_df, base_features, scaler=scaler)
        y_va = va_df["market_forward_excess_returns"].to_numpy()

        train_loader = DataLoader(SequenceDataset(X_tr, y_tr, seq_len), batch_size=128, shuffle=True)
        val_loader = DataLoader(SequenceDataset(X_va, y_va, seq_len), batch_size=128, shuffle=False)

        model = SimpleTemporalTransformer(len(base_features), d_model, nhead, num_layers, 0.2)
        model = train_model(model, train_loader, val_loader, epochs=50, lr=0.001, patience=8, device=device)

        model.eval()
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                all_preds.append(model(X_batch.to(device)).cpu().numpy())

        y_pred = np.concatenate(all_preds).flatten()
        sharpe = eval_fold(y_pred, va_df, seq_len, k)
        sharpes.append(sharpe)

    return np.mean(sharpes), np.std(sharpes)


if __name__ == "__main__":
    if not HAS_TORCH:
        print("[ERROR] PyTorch required")
        exit(1)

    print("Loading data...")
    train = load_train()
    base_features = select_base_features(train)
    print(f"Train: {train.shape}, Features: {len(base_features)}\n")

    # Test promising configurations
    configs = [
        # (seq_len, d_model, nhead, num_layers, k, description)
        (20, 64, 2, 1, 600, "Baseline (from fast test)"),
        (30, 64, 2, 1, 600, "Longer context"),
        (20, 128, 4, 2, 600, "Larger model"),
        (30, 128, 4, 2, 600, "Longer + Larger"),
        (20, 64, 2, 1, 800, "Higher k"),
        (25, 128, 4, 2, 600, "Balanced"),
    ]

    results = []

    print("="*80)
    print("QUICK TUNING - Testing promising configurations")
    print("="*80)

    for i, (seq_len, d_model, nhead, num_layers, k, desc) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {desc}")
        print(f"  Config: seq={seq_len}, d={d_model}, heads={nhead}, layers={num_layers}, k={k}")

        sharpe_mean, sharpe_std = test_config(train, seq_len, d_model, nhead, num_layers, k, base_features)

        results.append({
            'seq_len': seq_len,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'k': k,
            'description': desc,
            'sharpe': sharpe_mean,
            'sharpe_std': sharpe_std
        })

        print(f"  Result: Sharpe {sharpe_mean:.3f} ± {sharpe_std:.3f}")

    # Results
    df_results = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    df_results.to_csv(RESULTS / "quick_tuning.csv", index=False)

    print("\n" + "="*80)
    print("RESULTS (sorted by Sharpe)")
    print("="*80 + "\n")
    print(df_results[['description', 'sharpe', 'sharpe_std']].to_string(index=False))

    # Best
    best = df_results.iloc[0]
    print("\n" + "="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    print(f"{best['description']}")
    print(f"seq_len={int(best['seq_len'])}, d_model={int(best['d_model'])}, nhead={int(best['nhead'])}, layers={int(best['num_layers'])}, k={int(best['k'])}")
    print(f"Sharpe: {best['sharpe']:.3f} ± {best['sharpe_std']:.3f}")
    print(f"Baseline (EXP-007): 0.749")
    improvement = (best['sharpe'] / 0.749 - 1) * 100
    print(f"Improvement: {improvement:+.1f}%")

    if best['sharpe'] > 0.749:
        print("\n🎉 NEW BEST! Transformer beats EXP-007!")
    else:
        print(f"\n📊 Close to baseline (gap: {0.749 - best['sharpe']:.3f})")

    print("\n[DONE]")
