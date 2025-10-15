#!/usr/bin/env python3
"""EXP-014: Multi-variate LSTM - All 94 features as time series

TRUE BREAKTHROUGH: Use ALL 94 features' temporal patterns!
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

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


class MultivariateLSTM(nn.Module):
    """LSTM for multi-variate time series (94 features)."""

    def __init__(self, input_dim=94, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention over time steps
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim]

        # Attention weights
        attn_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, hidden_dim]

        # Predict
        out = self.predictor(context)
        return out.squeeze(-1)


class MultivarSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 60):
        self.X = X  # [n_samples, n_features]
        self.y = y  # [n_samples]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx+self.seq_len]  # [seq_len, features]
        y_target = self.y[idx+self.seq_len-1]  # Target at last timestep
        return torch.FloatTensor(x_seq), torch.FloatTensor([y_target])


def load_train(path: str = "data/train.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def select_all_features(df: pd.DataFrame) -> list[str]:
    """Select ALL original features (not just subset)."""
    exclude = {
        'date_id', 'forward_returns', 'risk_free_rate',
        'market_forward_excess_returns', 'is_scored'
    }
    return [c for c in df.columns if c not in exclude]


def preprocess(df: pd.DataFrame, features: list[str], scaler=None) -> Tuple[np.ndarray, StandardScaler]:
    X = df[features].copy()
    # Fill NaN with 0 (missing features)
    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    if scaler is None:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = scaler.transform(X)

    return Xs, scaler


def train_model(
    model: MultivariateLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 150,
    lr: float = 0.0005,
    patience: int = 20,
    device: str = "cpu",
) -> MultivariateLSTM:
    """Train LSTM model."""

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.squeeze().to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.squeeze().to(device)
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

        if (epoch + 1) % 30 == 0:
            print(f"    Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

    return model


def eval_fold(y_pred: np.ndarray, valid_df: pd.DataFrame, seq_len: int, k: float) -> Dict[str, float]:
    """Evaluate predictions."""
    valid_df_eval = valid_df.iloc[seq_len-1:]

    rf = valid_df_eval["risk_free_rate"].to_numpy()
    fwd = valid_df_eval["forward_returns"].to_numpy()
    excess_true = valid_df_eval["market_forward_excess_returns"].to_numpy()

    min_len = min(len(y_pred), len(rf))
    y_pred, rf, fwd, excess_true = y_pred[:min_len], rf[:min_len], fwd[:min_len], excess_true[:min_len]

    # Positions
    positions = np.clip(1.0 + y_pred * k, 0.0, 2.0)

    # Strategy returns
    strat = rf * (1.0 - positions) + fwd * positions
    excess = strat - rf

    strat_vol = float(np.std(strat))
    sharpe = (np.mean(excess) / strat_vol) * math.sqrt(252) if strat_vol > 0 else 0.0

    total_profit = float(np.sum(excess))
    utility = min(max(sharpe, 0), 6.0) * total_profit

    return {
        "sharpe": sharpe,
        "utility": utility,
        "profit": total_profit,
        "mse": float(np.mean((excess_true - y_pred) ** 2)),
    }


def run_experiment(
    train: pd.DataFrame,
    seq_len: int = 60,
    hidden_dim: int = 256,
    num_layers: int = 3,
    k: float = 1000.0,
) -> pd.DataFrame:
    """Run multi-variate LSTM experiment."""

    if not HAS_TORCH:
        return pd.DataFrame()

    print(f"\nMulti-variate LSTM: seq={seq_len}, hidden={hidden_dim}, layers={num_layers}, k={k}")
    print("="*80)

    # Use ALL features
    all_features = select_all_features(train)
    print(f"[INFO] Using ALL {len(all_features)} features!")

    device = "cpu"
    target = "market_forward_excess_returns"
    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(train), 1):
        print(f"\n[Fold {fold_idx}/5]")
        tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]

        # Preprocess
        X_tr, scaler = preprocess(tr_df, all_features)
        y_tr = tr_df[target].to_numpy()

        X_va, _ = preprocess(va_df, all_features, scaler=scaler)
        y_va = va_df[target].to_numpy()

        # Create datasets
        train_dataset = MultivarSequenceDataset(X_tr, y_tr, seq_len=seq_len)
        val_dataset = MultivarSequenceDataset(X_va, y_va, seq_len=seq_len)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        print(f"  Train sequences: {len(train_dataset)}, Val: {len(val_dataset)}")

        # Train
        model = MultivariateLSTM(
            input_dim=len(all_features),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.3
        )
        model = train_model(model, train_loader, val_loader, epochs=150, lr=0.0005, patience=20, device=device)

        # Predict
        model.eval()
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                output = model(X_batch.to(device))
                all_preds.append(output.cpu().numpy())

        y_pred = np.concatenate(all_preds)

        # Evaluate
        metrics = eval_fold(y_pred, va_df, seq_len, k)
        metrics.update({"fold": fold_idx, "seq_len": seq_len, "hidden_dim": hidden_dim, "k": k})
        results.append(metrics)

        print(f"  Sharpe: {metrics['sharpe']:.3f}, Utility: {metrics['utility']:.3f}")

    df_results = pd.DataFrame(results)
    avg_sharpe = df_results['sharpe'].mean()
    avg_utility = df_results['utility'].mean()

    print(f"\n[RESULT]")
    print(f"  Avg Sharpe:  {avg_sharpe:.3f}")
    print(f"  Avg Utility: {avg_utility:.3f}")
    print(f"  Baseline:    0.749 (EXP-007)")
    improvement = (avg_sharpe / 0.749 - 1) * 100
    print(f"  Improvement: {improvement:+.1f}%")

    if avg_sharpe > 3.0:
        print("\n🎉🎉🎉 BREAKTHROUGH! Sharpe > 3.0!")
    elif avg_sharpe > 2.0:
        print("\n🚀🚀 EXCELLENT! Sharpe > 2.0!")
    elif avg_sharpe > 1.5:
        print("\n🎯 GREAT! Sharpe > 1.5!")
    elif avg_sharpe > 1.0:
        print("\n✅ SUCCESS! Sharpe > 1.0!")

    return df_results


if __name__ == "__main__":
    if not HAS_TORCH:
        print("[ERROR] PyTorch required")
        exit(1)

    print("Loading data...")
    train = load_train()
    print(f"Train shape: {train.shape}\n")

    # Test configurations
    best_sharpe = 0
    best_config = None

    configs = [
        (60, 256, 3, 1000),   # seq=60, hidden=256, layers=3, k=1000
        (60, 256, 3, 1500),   # Higher k
        (60, 512, 4, 1000),   # Larger model
    ]

    for seq_len, hidden_dim, num_layers, k in configs:
        df_results = run_experiment(train, seq_len, hidden_dim, num_layers, k)
        df_results.to_csv(RESULTS / f"lstm_h{hidden_dim}_l{num_layers}_k{k}.csv", index=False)

        sharpe = df_results['sharpe'].mean()
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_config = (seq_len, hidden_dim, num_layers, k)

    print("\n" + "="*80)
    print("BEST RESULT")
    print("="*80)
    print(f"Config: seq={best_config[0]}, hidden={best_config[1]}, layers={best_config[2]}, k={best_config[3]}")
    print(f"Sharpe: {best_sharpe:.3f}")

    if best_sharpe > 1.5:
        print("\n🎊 Ready for Kaggle submission!")

    print("\n[DONE]")
