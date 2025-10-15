#!/usr/bin/env python3
"""Direct Utility Maximization via Differentiable Sharpe Optimization

Instead of predicting returns, directly learn positions that maximize utility.
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
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


class PositionNetwork(nn.Module):
    """Neural network that directly outputs positions [0, 2]."""

    def __init__(self, input_dim: int = 94, hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer: sigmoid → [0, 1] → scale to [0, 2]
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Output positions in [0, 2]."""
        out = self.network(x)
        # Sigmoid for [0, 1], then scale to [0, 2]
        positions = torch.sigmoid(out) * 2.0
        return positions.squeeze(-1)  # [batch]


def utility_loss(
    positions: torch.Tensor,
    rf: torch.Tensor,
    fwd: torch.Tensor,
    lambda_sharpe: float = 0.1,
    target_sharpe: float = 6.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate utility loss (negative utility).

    Args:
        positions: [batch] tensor, values in [0, 2]
        rf: [batch] risk-free rates
        fwd: [batch] forward returns
        lambda_sharpe: penalty weight for Sharpe > target
        target_sharpe: target Sharpe ratio

    Returns:
        loss: negative utility
        sharpe: calculated Sharpe ratio
        profit: total profit
    """
    # Strategy returns
    strat_returns = rf * (1.0 - positions) + fwd * positions
    excess_returns = strat_returns - rf

    # Sharpe ratio (differentiable)
    mean_excess = torch.mean(excess_returns)
    std_excess = torch.std(excess_returns, unbiased=False) + 1e-6  # Add epsilon for stability
    sharpe = (mean_excess / std_excess) * math.sqrt(252)

    # Clip Sharpe to [0, target_sharpe]
    sharpe_clipped = torch.clamp(sharpe, 0.0, target_sharpe)

    # Profit (sum of excess returns)
    profit = torch.sum(excess_returns)

    # Utility = sharpe_clipped × profit
    utility = sharpe_clipped * profit

    # Loss = negative utility
    loss = -utility

    # Penalty: discourage Sharpe > target (soft constraint)
    sharpe_penalty = torch.relu(sharpe - target_sharpe) ** 2
    loss = loss + lambda_sharpe * sharpe_penalty

    return loss, sharpe, profit


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


def train_position_network(
    model: PositionNetwork,
    X_train: np.ndarray,
    rf_train: np.ndarray,
    fwd_train: np.ndarray,
    X_val: np.ndarray,
    rf_val: np.ndarray,
    fwd_val: np.ndarray,
    epochs: int = 200,
    lr: float = 0.001,
    patience: int = 20,
    device: str = "cpu",
) -> PositionNetwork:
    """Train position network to maximize utility."""

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    rf_train_t = torch.FloatTensor(rf_train).to(device)
    fwd_train_t = torch.FloatTensor(fwd_train).to(device)

    X_val_t = torch.FloatTensor(X_val).to(device)
    rf_val_t = torch.FloatTensor(rf_val).to(device)
    fwd_val_t = torch.FloatTensor(fwd_val).to(device)

    best_val_utility = -float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        positions = model(X_train_t)
        loss, sharpe, profit = utility_loss(positions, rf_train_t, fwd_train_t)

        loss.backward()
        optimizer.step()

        train_utility = -loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            positions_val = model(X_val_t)
            val_loss, val_sharpe, val_profit = utility_loss(positions_val, rf_val_t, fwd_val_t)
            val_utility = -val_loss.item()

        # Early stopping
        if val_utility > best_val_utility:
            best_val_utility = val_utility
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}: Train Utility={train_utility:.3f}, Val Utility={val_utility:.3f}, "
                  f"Val Sharpe={val_sharpe:.3f}")

    return model


def evaluate_model(
    model: PositionNetwork,
    X: np.ndarray,
    rf: np.ndarray,
    fwd: np.ndarray,
    excess_true: np.ndarray,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate model on validation set."""

    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        rf_t = torch.FloatTensor(rf).to(device)
        fwd_t = torch.FloatTensor(fwd).to(device)

        positions = model(X_t).cpu().numpy()

    # Calculate metrics
    strat_returns = rf * (1.0 - positions) + fwd * positions
    excess_returns = strat_returns - rf

    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)
    sharpe = (mean_excess / std_excess) * math.sqrt(252) if std_excess > 0 else 0.0

    profit = np.sum(excess_returns)
    utility = min(max(sharpe, 0), 6.0) * profit

    return {
        "sharpe": float(sharpe),
        "profit": float(profit),
        "utility": float(utility),
        "pos_mean": float(np.mean(positions)),
        "pos_std": float(np.std(positions)),
        "pos_min": float(np.min(positions)),
        "pos_max": float(np.max(positions)),
    }


def run_experiment(
    train: pd.DataFrame,
    hidden_dims: list = None,
    lambda_sharpe: float = 0.1,
) -> pd.DataFrame:
    """Run direct utility optimization experiment."""

    if not HAS_TORCH:
        return pd.DataFrame()

    if hidden_dims is None:
        hidden_dims = [128, 64, 32]

    print(f"\nDirect Utility Optimization: hidden={hidden_dims}, λ={lambda_sharpe}")
    print("="*80)

    base_features = select_base_features(train)
    print(f"Features: {len(base_features)}")

    device = "cpu"
    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(train), 1):
        print(f"\n[Fold {fold_idx}/5]")
        tr_df = train.iloc[tr_idx]
        va_df = train.iloc[va_idx]

        # Preprocess
        X_tr, scaler = preprocess(tr_df, base_features)
        rf_tr = tr_df["risk_free_rate"].to_numpy()
        fwd_tr = tr_df["forward_returns"].to_numpy()

        X_va, _ = preprocess(va_df, base_features, scaler=scaler)
        rf_va = va_df["risk_free_rate"].to_numpy()
        fwd_va = va_df["forward_returns"].to_numpy()
        excess_va = va_df["market_forward_excess_returns"].to_numpy()

        # Train model
        model = PositionNetwork(input_dim=len(base_features), hidden_dims=hidden_dims, dropout=0.3)
        model = train_position_network(
            model, X_tr, rf_tr, fwd_tr, X_va, rf_va, fwd_va,
            epochs=200, lr=0.001, patience=20, device=device
        )

        # Evaluate
        metrics = evaluate_model(model, X_va, rf_va, fwd_va, excess_va, device=device)
        metrics.update({"fold": fold_idx, "lambda_sharpe": lambda_sharpe})
        results.append(metrics)

        print(f"  Result: Sharpe={metrics['sharpe']:.3f}, Profit={metrics['profit']:.4f}, "
              f"Utility={metrics['utility']:.3f}")

    df_results = pd.DataFrame(results)
    avg_sharpe = df_results['sharpe'].mean()
    avg_utility = df_results['utility'].mean()

    print(f"\n[RESULT]")
    print(f"  Avg Sharpe:  {avg_sharpe:.3f}")
    print(f"  Avg Utility: {avg_utility:.3f}")
    print(f"  Baseline (EXP-007): Sharpe 0.749")
    improvement = (avg_sharpe / 0.749 - 1) * 100
    print(f"  Improvement: {improvement:+.1f}%")

    return df_results


if __name__ == "__main__":
    if not HAS_TORCH:
        print("[ERROR] PyTorch required")
        exit(1)

    print("Loading data...")
    train = load_train()
    print(f"Train shape: {train.shape}\n")

    # Experiment 1: Default settings
    df_1 = run_experiment(train, hidden_dims=[128, 64, 32], lambda_sharpe=0.1)
    df_1.to_csv(RESULTS / "exp1_default.csv", index=False)

    # Experiment 2: Lower Sharpe penalty (focus more on profit)
    df_2 = run_experiment(train, hidden_dims=[128, 64, 32], lambda_sharpe=0.01)
    df_2.to_csv(RESULTS / "exp2_low_penalty.csv", index=False)

    # Experiment 3: Smaller network
    df_3 = run_experiment(train, hidden_dims=[64, 32], lambda_sharpe=0.1)
    df_3.to_csv(RESULTS / "exp3_small.csv", index=False)

    # Compare
    if not df_1.empty and not df_2.empty and not df_3.empty:
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        print(f"Default (λ=0.1, [128,64,32]):     Sharpe {df_1['sharpe'].mean():.3f}, Utility {df_1['utility'].mean():.3f}")
        print(f"Low penalty (λ=0.01, [128,64,32]): Sharpe {df_2['sharpe'].mean():.3f}, Utility {df_2['utility'].mean():.3f}")
        print(f"Small net (λ=0.1, [64,32]):       Sharpe {df_3['sharpe'].mean():.3f}, Utility {df_3['utility'].mean():.3f}")
        print(f"\nEXP-007 Baseline: Sharpe 0.749")

        best_sharpe = max(df_1['sharpe'].mean(), df_2['sharpe'].mean(), df_3['sharpe'].mean())
        print(f"Best Sharpe: {best_sharpe:.3f}")

        if best_sharpe > 0.749:
            print("\n🎉 NEW BEST! Direct optimization beats EXP-007!")
        else:
            print(f"\n📊 Gap to baseline: {0.749 - best_sharpe:.3f}")

    print("\n[DONE]")
