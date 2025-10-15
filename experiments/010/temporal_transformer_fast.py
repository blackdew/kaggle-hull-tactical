#!/usr/bin/env python3
"""Fast Temporal Transformer - Simplified version for quick testing"""
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
    """Simplified Transformer for faster training."""

    def __init__(self, feature_dim: int = 97, d_model: int = 64, nhead: int = 2, seq_len: int = 10):
        super().__init__()

        self.input_proj = nn.Linear(feature_dim, d_model)

        # Single transformer layer (lighter)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.2,
            activation='relu',
            batch_first=True
        )

        # Simple predictor
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, feature_dim]
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        x = self.transformer_layer(x)  # [batch, seq_len, d_model]
        x = x[:, -1, :]  # Use last timestep
        output = self.predictor(x)
        return output


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 10):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx+self.seq_len]
        y_target = self.y[idx+self.seq_len-1]
        return torch.FloatTensor(x_seq), torch.FloatTensor([y_target])


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


def train_model_fast(
    model: SimpleTemporalTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,  # Reduced
    device: str = "cpu",
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Higher LR
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 5:  # Early stop
            print(f"  Early stop at epoch {epoch + 1}")
            break

    return model


def eval_fold(y_pred: np.ndarray, valid_df: pd.DataFrame, seq_len: int, k: float = 600.0) -> Dict[str, float]:
    valid_df_eval = valid_df.iloc[seq_len-1:]

    rf = valid_df_eval["risk_free_rate"].to_numpy()
    fwd = valid_df_eval["forward_returns"].to_numpy()
    excess_true = valid_df_eval["market_forward_excess_returns"].to_numpy()

    min_len = min(len(y_pred), len(rf))
    y_pred = y_pred[:min_len]
    rf = rf[:min_len]
    fwd = fwd[:min_len]
    excess_true = excess_true[:min_len]

    positions = np.clip(1.0 + y_pred * k, 0.0, 2.0)
    strat = rf * (1.0 - positions) + fwd * positions
    excess = strat - rf

    strat_vol = float(np.std(strat))
    sharpe = (np.mean(excess) / strat_vol) * np.sqrt(252) if strat_vol > 0 else 0.0
    mse = float(np.mean((excess_true - y_pred) ** 2))

    return {"sharpe": sharpe, "mse": mse}


def run_experiment_fast(train: pd.DataFrame, seq_len: int = 10) -> pd.DataFrame:
    if not HAS_TORCH:
        return pd.DataFrame()

    print(f"\nTransformer (Fast): seq_len={seq_len}")
    print("="*60)

    target = "market_forward_excess_returns"
    base_features = select_base_features(train)
    print(f"Features: {len(base_features)}, Seq length: {seq_len}")

    device = "cpu"  # Force CPU for consistency
    tscv = TimeSeriesSplit(n_splits=3)  # Reduced to 3 folds
    results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(train), 1):
        print(f"\n[Fold {fold_idx}/3]")
        tr_df = train.iloc[tr_idx]
        va_df = train.iloc[va_idx]

        X_tr, scaler = preprocess(tr_df, base_features)
        y_tr = tr_df[target].to_numpy()

        X_va, _ = preprocess(va_df, base_features, scaler=scaler)
        y_va = va_df[target].to_numpy()

        train_dataset = SequenceDataset(X_tr, y_tr, seq_len=seq_len)
        val_dataset = SequenceDataset(X_va, y_va, seq_len=seq_len)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        model = SimpleTemporalTransformer(
            feature_dim=len(base_features),
            d_model=64,
            nhead=2,
            seq_len=seq_len
        )

        model = train_model_fast(model, train_loader, val_loader, epochs=30, device=device)

        # Predict
        model.eval()
        all_preds = []

        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                output = model(X_batch)
                all_preds.append(output.cpu().numpy())

        y_pred = np.concatenate(all_preds).flatten()

        metrics = eval_fold(y_pred, va_df, seq_len, k=600)
        metrics.update({"fold": fold_idx, "seq_len": seq_len})
        results.append(metrics)

        print(f"  Sharpe: {metrics['sharpe']:.3f}, MSE: {metrics['mse']:.6f}")

    df_results = pd.DataFrame(results)
    avg_sharpe = df_results['sharpe'].mean()

    print(f"\n[RESULT] Avg Sharpe: {avg_sharpe:.3f}")
    print(f"[BASELINE] EXP-007: 0.749")
    improvement = (avg_sharpe / 0.749 - 1) * 100
    print(f"[IMPROVEMENT] {improvement:+.1f}%")

    return df_results


if __name__ == "__main__":
    if not HAS_TORCH:
        print("[ERROR] PyTorch required")
        exit(1)

    print("Loading data...")
    train = load_train()
    print(f"Train shape: {train.shape}\n")

    # Test seq_len=10 (faster)
    df_1 = run_experiment_fast(train, seq_len=10)
    df_1.to_csv(RESULTS / "fast_seq10.csv", index=False)

    # Test seq_len=20
    df_2 = run_experiment_fast(train, seq_len=20)
    df_2.to_csv(RESULTS / "fast_seq20.csv", index=False)

    if not df_1.empty and not df_2.empty:
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"seq_len=10: {df_1['sharpe'].mean():.3f}")
        print(f"seq_len=20: {df_2['sharpe'].mean():.3f}")
        print(f"Best: {max(df_1['sharpe'].mean(), df_2['sharpe'].mean()):.3f}")
        print(f"EXP-007: 0.749")

    print("\n[DONE]")
