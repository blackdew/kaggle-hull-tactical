#!/usr/bin/env python3
"""Temporal Attention Transformer for Time Series Prediction

Uses past N days with multi-head self-attention to predict excess returns.
"""
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
    print("[ERROR] PyTorch not available")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input.

        Args:
            x: [batch, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class TemporalAttentionTransformer(nn.Module):
    """Transformer encoder for temporal sequence prediction."""

    def __init__(
        self,
        feature_dim: int = 97,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        seq_len: int = 20,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            activation='relu',
            batch_first=True  # Important: use batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: [batch, seq_len, feature_dim]

        Returns:
            output: [batch, 1]
        """
        # Project to d_model
        x = self.input_proj(x)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = self.pos_encoding(x)  # [batch, seq_len, d_model]

        # Apply Transformer
        x = self.transformer(x)  # [batch, seq_len, d_model]

        # Use last position for prediction
        x = x[:, -1, :]  # [batch, d_model]

        # Predict
        output = self.predictor(x)  # [batch, 1]

        return output


class SequenceDataset(Dataset):
    """Dataset for temporal sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 20):
        """
        Args:
            X: [num_samples, feature_dim]
            y: [num_samples]
            seq_len: length of lookback window
        """
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        # Get sequence [idx:idx+seq_len]
        x_seq = self.X[idx:idx+self.seq_len]  # [seq_len, feature_dim]
        # Target is the last day's excess return
        y_target = self.y[idx+self.seq_len-1]

        return torch.FloatTensor(x_seq), torch.FloatTensor([y_target])


def load_train(path: str = "data/train.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def select_base_features(df: pd.DataFrame) -> list[str]:
    """Select original features only."""
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
    model: TemporalAttentionTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 0.0001,
    epochs: int = 100,
    patience: int = 15,
    device: str = "cpu",
) -> Tuple[TemporalAttentionTransformer, list]:
    """Train transformer with early stopping."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        # Training
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

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
        })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

    return model, history


def eval_fold(
    y_pred: np.ndarray,
    valid_df: pd.DataFrame,
    seq_len: int,
    k: float = 600.0,
) -> Dict[str, float]:
    """Evaluate predictions on validation fold."""
    # Skip first seq_len-1 samples (used for sequence construction)
    valid_df_eval = valid_df.iloc[seq_len-1:]

    rf = valid_df_eval["risk_free_rate"].to_numpy()
    fwd = valid_df_eval["forward_returns"].to_numpy()
    excess_true = valid_df_eval["market_forward_excess_returns"].to_numpy()

    # Ensure same length
    min_len = min(len(y_pred), len(rf))
    y_pred = y_pred[:min_len]
    rf = rf[:min_len]
    fwd = fwd[:min_len]
    excess_true = excess_true[:min_len]

    # Convert predictions to positions
    positions = np.clip(1.0 + y_pred * k, 0.0, 2.0)

    # Strategy returns
    strat = rf * (1.0 - positions) + fwd * positions
    excess = strat - rf

    strat_vol = float(np.std(strat))
    sharpe = (np.mean(excess) / strat_vol) * np.sqrt(252) if strat_vol > 0 else 0.0
    mse = float(np.mean((excess_true - y_pred) ** 2))

    return {
        "sharpe": sharpe,
        "mse": mse,
        "strat_vol": strat_vol,
        "pos_mean": float(np.mean(positions)),
        "pos_std": float(np.std(positions)),
    }


def run_experiment(
    train: pd.DataFrame,
    seq_len: int = 20,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    k: float = 600.0,
) -> pd.DataFrame:
    """Run transformer experiment with time series CV."""
    if not HAS_TORCH:
        return pd.DataFrame()

    print("\n" + "="*80)
    print(f"Transformer: seq_len={seq_len}, d_model={d_model}, nhead={nhead}, layers={num_layers}, k={k}")
    print("="*80)

    target = "market_forward_excess_returns"
    base_features = select_base_features(train)
    print(f"[INFO] Input features: {len(base_features)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(train), 1):
        print(f"\n[Fold {fold_idx}/5]")
        tr_df = train.iloc[tr_idx]
        va_df = train.iloc[va_idx]

        # Preprocess
        X_tr, scaler = preprocess(tr_df, base_features)
        y_tr = tr_df[target].to_numpy()

        X_va, _ = preprocess(va_df, base_features, scaler=scaler)
        y_va = va_df[target].to_numpy()

        # Create sequence datasets
        train_dataset = SequenceDataset(X_tr, y_tr, seq_len=seq_len)
        val_dataset = SequenceDataset(X_va, y_va, seq_len=seq_len)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        print(f"  Train sequences: {len(train_dataset)}, Val sequences: {len(val_dataset)}")

        # Train model
        model = TemporalAttentionTransformer(
            feature_dim=len(base_features),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            seq_len=seq_len,
            dropout=0.3
        )

        model, history = train_model(
            model,
            train_loader,
            val_loader,
            lr=0.0001,
            epochs=100,
            patience=15,
            device=device
        )

        # Predict on validation
        model.eval()
        all_preds = []

        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                output = model(X_batch)
                all_preds.append(output.cpu().numpy())

        y_pred = np.concatenate(all_preds).flatten()

        # Evaluate
        metrics = eval_fold(y_pred, va_df, seq_len, k=k)
        metrics.update({
            "fold": fold_idx,
            "seq_len": seq_len,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "k": k,
        })
        results.append(metrics)

        print(f"  Fold {fold_idx} Result: Sharpe {metrics['sharpe']:.3f}, MSE {metrics['mse']:.6f}")

    df_results = pd.DataFrame(results)
    avg_sharpe = df_results['sharpe'].mean()
    avg_mse = df_results['mse'].mean()

    print(f"\n[RESULT] Transformer:")
    print(f"  Sharpe:   {avg_sharpe:.3f}")
    print(f"  MSE:      {avg_mse:.6f}")
    print(f"  Baseline: 0.749 (EXP-007)")
    improvement = (avg_sharpe / 0.749 - 1) * 100
    print(f"  Improvement: {improvement:+.1f}%")

    return df_results


if __name__ == "__main__":
    if not HAS_TORCH:
        print("[ERROR] Please install PyTorch")
        exit(1)

    print("[INFO] Loading training data...")
    train = load_train()
    print(f"[INFO] Train shape: {train.shape}")

    # Experiment 1: seq_len=20 (baseline)
    df_1 = run_experiment(train, seq_len=20, d_model=128, nhead=4, num_layers=2, k=600)
    df_1.to_csv(RESULTS / "exp1_seq20.csv", index=False)

    # Experiment 2: seq_len=30 (longer context)
    df_2 = run_experiment(train, seq_len=30, d_model=128, nhead=4, num_layers=2, k=600)
    df_2.to_csv(RESULTS / "exp2_seq30.csv", index=False)

    # Experiment 3: More attention heads
    df_3 = run_experiment(train, seq_len=20, d_model=128, nhead=8, num_layers=2, k=600)
    df_3.to_csv(RESULTS / "exp3_nhead8.csv", index=False)

    # Compare
    if not df_1.empty and not df_2.empty and not df_3.empty:
        sharpe_1 = df_1['sharpe'].mean()
        sharpe_2 = df_2['sharpe'].mean()
        sharpe_3 = df_3['sharpe'].mean()

        print("\n" + "="*80)
        print("TRANSFORMER COMPARISON")
        print("="*80)
        print(f"seq_len=20, nhead=4: {sharpe_1:.3f}")
        print(f"seq_len=30, nhead=4: {sharpe_2:.3f}")
        print(f"seq_len=20, nhead=8: {sharpe_3:.3f}")
        print(f"Best: {max(sharpe_1, sharpe_2, sharpe_3):.3f}")
        print(f"EXP-007 Baseline: 0.749")

    print("\n[DONE] Transformer experiments completed!")
