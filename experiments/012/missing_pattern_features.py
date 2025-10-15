#!/usr/bin/env python3
"""EXP-012: Missing Pattern Recognition + Period-Aware Strategy

Key insights:
1. Missing patterns are signals
2. Early period (date < 1784) vs Late period (date >= 1784)
3. High-confidence only trading for Sharpe 6.0
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def create_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from missing patterns."""
    df_new = df.copy()

    # Get feature columns
    feature_cols = [c for c in df.columns if c not in {
        'date_id', 'forward_returns', 'risk_free_rate',
        'market_forward_excess_returns', 'is_scored'
    }]

    # 1. Total missing count and percentage
    df_new['n_missing'] = df[feature_cols].isnull().sum(axis=1)
    df_new['pct_missing'] = df_new['n_missing'] / len(feature_cols)

    # 2. Group-level missing indicators
    for group in ['D', 'E', 'I', 'M', 'P', 'S', 'V']:
        group_cols = [c for c in feature_cols if c.startswith(group)]
        if group_cols:
            df_new[f'{group}_n_missing'] = df[group_cols].isnull().sum(axis=1)
            df_new[f'{group}_pct_missing'] = df_new[f'{group}_n_missing'] / len(group_cols)
            df_new[f'{group}_available'] = (df_new[f'{group}_n_missing'] < len(group_cols)).astype(int)

    # 3. Period indicator
    df_new['period'] = (df['date_id'] >= 1784).astype(int)  # 0=early, 1=late

    # 4. Data richness score
    df_new['data_richness'] = 1 - df_new['pct_missing']

    return df_new


def select_features(df: pd.DataFrame) -> List[str]:
    """Select all features including missing pattern features."""
    exclude = {
        'date_id', 'forward_returns', 'risk_free_rate',
        'market_forward_excess_returns', 'is_scored'
    }
    return [c for c in df.columns if c not in exclude]


def preprocess(df: pd.DataFrame, features: List[str], scaler=None) -> Tuple[np.ndarray, StandardScaler]:
    X = df[features].copy()
    # Fill NaN with 0 (after creating missing indicators)
    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    if scaler is None:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = scaler.transform(X)

    return Xs, scaler


def eval_fold_with_confidence(
    y_pred: np.ndarray,
    y_std: np.ndarray,  # Uncertainty estimates
    valid_df: pd.DataFrame,
    confidence_threshold: float = 0.7,
    k_base: float = 1000.0,
) -> Dict[str, float]:
    """Evaluate with confidence-based position sizing."""

    rf = valid_df["risk_free_rate"].to_numpy()
    fwd = valid_df["forward_returns"].to_numpy()
    excess_true = valid_df["market_forward_excess_returns"].to_numpy()

    # Calculate confidence score (lower std = higher confidence)
    # Normalize std to [0, 1] range
    std_normalized = y_std / (np.abs(y_pred) + 1e-6)  # Coefficient of variation
    confidence = np.exp(-std_normalized)  # Higher std → lower confidence

    # Position sizing based on confidence
    positions = np.ones_like(y_pred)  # Start with neutral (1.0)

    for i in range(len(y_pred)):
        if confidence[i] > confidence_threshold:
            # High confidence → trade
            positions[i] = np.clip(1.0 + y_pred[i] * k_base, 0.0, 2.0)
        else:
            # Low confidence → stay neutral
            positions[i] = 1.0

    # Strategy returns
    strat = rf * (1.0 - positions) + fwd * positions
    excess = strat - rf

    strat_vol = float(np.std(strat))
    sharpe = (np.mean(excess) / strat_vol) * np.sqrt(252) if strat_vol > 0 else 0.0
    mse = float(np.mean((excess_true - y_pred) ** 2))

    # Count trades
    n_trades = np.sum(positions != 1.0)
    trade_pct = n_trades / len(positions)

    return {
        "sharpe": sharpe,
        "mse": mse,
        "strat_vol": strat_vol,
        "pos_mean": float(np.mean(positions)),
        "pos_std": float(np.std(positions)),
        "n_trades": int(n_trades),
        "trade_pct": float(trade_pct),
    }


def train_ensemble_with_uncertainty(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    n_models: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train ensemble and return mean + std predictions."""

    predictions = []

    for i in range(n_models):
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + i,
            tree_method='hist',
            verbosity=0
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        predictions.append(pred)

    predictions = np.array(predictions)  # [n_models, n_samples]
    pred_mean = np.mean(predictions, axis=0)
    pred_std = np.std(predictions, axis=0)

    return pred_mean, pred_std


def run_experiment(
    train: pd.DataFrame,
    confidence_threshold: float = 0.7,
    k_base: float = 1000.0,
) -> pd.DataFrame:
    """Run missing pattern + confidence-based experiment."""

    if not HAS_XGB:
        return pd.DataFrame()

    print(f"\nEXP-012: Missing Pattern + Confidence-Based")
    print(f"  confidence_threshold={confidence_threshold}, k_base={k_base}")
    print("="*80)

    # Create missing pattern features
    print("[INFO] Creating missing pattern features...")
    train_eng = create_missing_features(train)

    all_features = select_features(train_eng)
    print(f"[INFO] Total features: {len(all_features)}")

    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(train_eng), 1):
        print(f"\n[Fold {fold_idx}/5]")
        tr_df = train_eng.iloc[tr_idx]
        va_df = train_eng.iloc[va_idx]

        target = "market_forward_excess_returns"

        X_tr, scaler = preprocess(tr_df, all_features)
        y_tr = tr_df[target].to_numpy()

        X_va, _ = preprocess(va_df, all_features, scaler=scaler)

        # Train ensemble for uncertainty estimation
        print("  Training ensemble (5 models)...")
        pred_mean, pred_std = train_ensemble_with_uncertainty(X_tr, y_tr, X_va, n_models=5)

        # Evaluate with confidence-based position
        metrics = eval_fold_with_confidence(
            pred_mean, pred_std, va_df,
            confidence_threshold=confidence_threshold,
            k_base=k_base
        )
        metrics.update({
            "fold": fold_idx,
            "conf_threshold": confidence_threshold,
            "k_base": k_base,
        })
        results.append(metrics)

        print(f"  Sharpe: {metrics['sharpe']:.3f}, Trades: {metrics['n_trades']} ({metrics['trade_pct']:.1%})")

    df_results = pd.DataFrame(results)
    avg_sharpe = df_results['sharpe'].mean()
    avg_trades = df_results['trade_pct'].mean()

    print(f"\n[RESULT]")
    print(f"  Avg Sharpe:    {avg_sharpe:.3f}")
    print(f"  Avg Trade %:   {avg_trades:.1%}")
    print(f"  Baseline:      0.749 (EXP-007)")
    improvement = (avg_sharpe / 0.749 - 1) * 100
    print(f"  Improvement:   {improvement:+.1f}%")

    return df_results


def load_train(path: str = "data/train.csv") -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    if not HAS_XGB:
        print("[ERROR] XGBoost required")
        exit(1)

    print("Loading data...")
    train = load_train()
    print(f"Train shape: {train.shape}\n")

    # Experiment 1: Moderate confidence threshold
    df_1 = run_experiment(train, confidence_threshold=0.7, k_base=1000)
    df_1.to_csv(RESULTS / "exp1_conf07.csv", index=False)

    # Experiment 2: High confidence threshold (fewer trades, higher Sharpe)
    df_2 = run_experiment(train, confidence_threshold=0.8, k_base=1500)
    df_2.to_csv(RESULTS / "exp2_conf08.csv", index=False)

    # Experiment 3: Very high confidence (extreme selectivity)
    df_3 = run_experiment(train, confidence_threshold=0.9, k_base=2000)
    df_3.to_csv(RESULTS / "exp3_conf09.csv", index=False)

    # Compare
    if not df_1.empty and not df_2.empty and not df_3.empty:
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        print(f"Conf=0.7, k=1000: Sharpe {df_1['sharpe'].mean():.3f}, Trades {df_1['trade_pct'].mean():.1%}")
        print(f"Conf=0.8, k=1500: Sharpe {df_2['sharpe'].mean():.3f}, Trades {df_2['trade_pct'].mean():.1%}")
        print(f"Conf=0.9, k=2000: Sharpe {df_3['sharpe'].mean():.3f}, Trades {df_3['trade_pct'].mean():.1%}")
        print(f"\nEXP-007 Baseline: 0.749")

        best_sharpe = max(df_1['sharpe'].mean(), df_2['sharpe'].mean(), df_3['sharpe'].mean())
        print(f"Best Sharpe: {best_sharpe:.3f}")

        if best_sharpe > 1.5:
            print("\n🎉 MAJOR BREAKTHROUGH! Sharpe > 1.5!")
        elif best_sharpe > 1.0:
            print("\n✅ SUCCESS! Sharpe > 1.0!")
        elif best_sharpe > 0.749:
            print("\n📈 IMPROVEMENT over baseline!")
        else:
            print("\n📊 Need further tuning...")

    print("\n[DONE]")
