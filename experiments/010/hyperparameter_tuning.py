#!/usr/bin/env python3
"""Hyperparameter tuning for Temporal Transformer

Grid search over:
- seq_len: [15, 20, 30]
- d_model: [64, 128]
- nhead: [2, 4]
- num_layers: [1, 2]
- k: [400, 600, 800]
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple
from itertools import product

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
    """Transformer for temporal sequences."""

    def __init__(self, feature_dim: int, d_model: int, nhead: int, num_layers: int, dropout: float = 0.2):
        super().__init__()

        self.input_proj = nn.Linear(feature_dim, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # Last timestep
        return self.predictor(x)


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
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


def train_model(
    model: SimpleTemporalTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    patience: int = 8,
    device: str = "cpu",
):
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
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return model


def eval_fold(y_pred: np.ndarray, valid_df: pd.DataFrame, seq_len: int, k: float) -> Dict[str, float]:
    valid_df_eval = valid_df.iloc[seq_len-1:]

    rf = valid_df_eval["risk_free_rate"].to_numpy()
    fwd = valid_df_eval["forward_returns"].to_numpy()
    excess_true = valid_df_eval["market_forward_excess_returns"].to_numpy()

    min_len = min(len(y_pred), len(rf))
    y_pred, rf, fwd, excess_true = y_pred[:min_len], rf[:min_len], fwd[:min_len], excess_true[:min_len]

    positions = np.clip(1.0 + y_pred * k, 0.0, 2.0)
    strat = rf * (1.0 - positions) + fwd * positions
    excess = strat - rf

    strat_vol = float(np.std(strat))
    sharpe = (np.mean(excess) / strat_vol) * np.sqrt(252) if strat_vol > 0 else 0.0
    mse = float(np.mean((excess_true - y_pred) ** 2))

    return {"sharpe": sharpe, "mse": mse}


def run_config(
    train: pd.DataFrame,
    seq_len: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    k: float,
    base_features: list[str],
) -> Dict[str, float]:
    """Run single configuration."""

    device = "cpu"
    tscv = TimeSeriesSplit(n_splits=3)  # 3 folds for speed
    results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(train), 1):
        tr_df = train.iloc[tr_idx]
        va_df = train.iloc[va_idx]

        X_tr, scaler = preprocess(tr_df, base_features)
        y_tr = tr_df["market_forward_excess_returns"].to_numpy()

        X_va, _ = preprocess(va_df, base_features, scaler=scaler)
        y_va = va_df["market_forward_excess_returns"].to_numpy()

        train_dataset = SequenceDataset(X_tr, y_tr, seq_len=seq_len)
        val_dataset = SequenceDataset(X_va, y_va, seq_len=seq_len)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        model = SimpleTemporalTransformer(
            feature_dim=len(base_features),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=0.2
        )

        model = train_model(model, train_loader, val_loader, epochs=50, lr=0.001, patience=8, device=device)

        # Predict
        model.eval()
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                output = model(X_batch)
                all_preds.append(output.cpu().numpy())

        y_pred = np.concatenate(all_preds).flatten()
        metrics = eval_fold(y_pred, va_df, seq_len, k)
        results.append(metrics['sharpe'])

    avg_sharpe = np.mean(results)
    return {
        'seq_len': seq_len,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'k': k,
        'sharpe': avg_sharpe,
        'sharpe_std': np.std(results)
    }


def grid_search(train: pd.DataFrame, base_features: list[str]) -> pd.DataFrame:
    """Run grid search over hyperparameters."""

    # Grid
    seq_lens = [15, 20, 30]
    d_models = [64, 128]
    nheads = [2, 4]
    num_layers_list = [1, 2]
    ks = [400, 600, 800]

    results = []
    total = len(seq_lens) * len(d_models) * len(nheads) * len(num_layers_list) * len(ks)

    print(f"\n{'='*80}")
    print(f"GRID SEARCH: {total} configurations")
    print(f"{'='*80}\n")

    count = 0
    for seq_len, d_model, nhead, num_layers, k in product(seq_lens, d_models, nheads, num_layers_list, ks):
        # Skip invalid configs
        if d_model % nhead != 0:
            continue

        count += 1
        print(f"[{count}/{total}] seq={seq_len}, d={d_model}, heads={nhead}, layers={num_layers}, k={k}")

        result = run_config(train, seq_len, d_model, nhead, num_layers, k, base_features)
        results.append(result)

        print(f"  → Sharpe: {result['sharpe']:.3f} ± {result['sharpe_std']:.3f}")

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('sharpe', ascending=False)

    return df_results


if __name__ == "__main__":
    if not HAS_TORCH:
        print("[ERROR] PyTorch required")
        exit(1)

    print("Loading data...")
    train = load_train()
    base_features = select_base_features(train)
    print(f"Train shape: {train.shape}, Features: {len(base_features)}\n")

    # Run grid search
    df_results = grid_search(train, base_features)

    # Save results
    df_results.to_csv(RESULTS / "grid_search.csv", index=False)
    print(f"\n{'='*80}")
    print("GRID SEARCH RESULTS")
    print(f"{'='*80}\n")
    print(df_results.head(10).to_string(index=False))

    # Best config
    best = df_results.iloc[0]
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"seq_len:    {int(best['seq_len'])}")
    print(f"d_model:    {int(best['d_model'])}")
    print(f"nhead:      {int(best['nhead'])}")
    print(f"num_layers: {int(best['num_layers'])}")
    print(f"k:          {int(best['k'])}")
    print(f"Sharpe:     {best['sharpe']:.3f} ± {best['sharpe_std']:.3f}")
    print(f"Baseline:   0.749 (EXP-007)")
    improvement = (best['sharpe'] / 0.749 - 1) * 100
    print(f"Improvement: {improvement:+.1f}%")

    print("\n[DONE] Grid search completed!")
