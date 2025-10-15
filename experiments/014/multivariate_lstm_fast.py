#!/usr/bin/env python3
"""EXP-014 Fast: Multi-variate LSTM - Faster version for quick testing"""
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


class SimpleLSTM(nn.Module):
    """Simpler LSTM for faster training."""

    def __init__(self, input_dim=94, hidden_dim=128, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_state = lstm_out[:, -1, :]
        return self.predictor(last_state).squeeze(-1)


class MultivarSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 40):
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
    exclude = {'date_id', 'forward_returns', 'risk_free_rate',
               'market_forward_excess_returns', 'is_scored'}
    return [c for c in df.columns if c not in exclude]


def preprocess(df: pd.DataFrame, features: list[str], scaler=None) -> Tuple[np.ndarray, StandardScaler]:
    X = df[features].copy().fillna(0).replace([np.inf, -np.inf], 0)
    if scaler is None:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = scaler.transform(X)
    return Xs, scaler


def train_model_fast(model, train_loader, val_loader, epochs=50, lr=0.001, patience=10, device="cpu"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.squeeze().to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_loss += criterion(model(X_batch.to(device)), y_batch.squeeze().to(device)).item()
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


def run_fast_experiment(train: pd.DataFrame, seq_len: int = 40, k: float = 1000.0) -> pd.DataFrame:
    if not HAS_TORCH:
        return pd.DataFrame()

    print(f"\nMulti-variate LSTM (FAST): seq={seq_len}, k={k}")
    print("="*80)

    all_features = select_all_features(train)
    print(f"[INFO] Using {len(all_features)} features")

    device = "cpu"
    target = "market_forward_excess_returns"
    tscv = TimeSeriesSplit(n_splits=3)  # 3-fold for speed
    results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(train), 1):
        print(f"\n[Fold {fold_idx}/3]")
        tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]

        X_tr, scaler = preprocess(tr_df, all_features)
        y_tr = tr_df[target].to_numpy()
        X_va, _ = preprocess(va_df, all_features, scaler=scaler)
        y_va = va_df[target].to_numpy()

        train_loader = DataLoader(MultivarSequenceDataset(X_tr, y_tr, seq_len), batch_size=128, shuffle=True)
        val_loader = DataLoader(MultivarSequenceDataset(X_va, y_va, seq_len), batch_size=128, shuffle=False)

        model = SimpleLSTM(input_dim=len(all_features), hidden_dim=128, num_layers=2)
        model = train_model_fast(model, train_loader, val_loader, epochs=50, lr=0.001, patience=10, device=device)

        model.eval()
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                all_preds.append(model(X_batch.to(device)).cpu().numpy())

        y_pred = np.concatenate(all_preds)
        metrics = eval_fold(y_pred, va_df, seq_len, k)
        metrics.update({"fold": fold_idx, "k": k})
        results.append(metrics)

        print(f"  Sharpe: {metrics['sharpe']:.3f}, Utility: {metrics['utility']:.3f}")

    df_results = pd.DataFrame(results)
    avg_sharpe = df_results['sharpe'].mean()

    print(f"\n[RESULT] Avg Sharpe: {avg_sharpe:.3f}, Baseline: 0.749")
    improvement = (avg_sharpe / 0.749 - 1) * 100
    print(f"Improvement: {improvement:+.1f}%")

    if avg_sharpe > 2.0:
        print("\n🎉 BREAKTHROUGH! Sharpe > 2.0!")
    elif avg_sharpe > 1.0:
        print("\n✅ SUCCESS! Sharpe > 1.0!")
    elif avg_sharpe > 0.749:
        print("\n📈 IMPROVEMENT!")

    return df_results


if __name__ == "__main__":
    if not HAS_TORCH:
        print("[ERROR] PyTorch required")
        exit(1)

    print("Loading data...")
    train = load_train()
    print(f"Train: {train.shape}\n")

    # Test different k values
    for k in [800, 1000, 1500, 2000]:
        df_results = run_fast_experiment(train, seq_len=40, k=k)
        df_results.to_csv(RESULTS / f"lstm_fast_k{int(k)}.csv", index=False)

    print("\n[DONE]")
