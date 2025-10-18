#!/usr/bin/env python3
"""EXP-016 Phase 3.3: Hyperparameter Tuning

Optimize XGBoost hyperparameters with Top 20 features.
Baseline: Sharpe 0.874
Goal: Sharpe 0.95~1.0+
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Import from feature_analysis
sys.path.insert(0, str(Path(__file__).parent))
from feature_analysis import load_exp007_features

try:
    from xgboost import XGBRegressor
except ImportError:
    print("[ERROR] XGBoost not available")
    sys.exit(1)

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("[ERROR] Optuna not available. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])
    import optuna
    from optuna.samplers import TPESampler


def eval_fold(y_pred: np.ndarray, y_true: np.ndarray, fwd_returns: np.ndarray,
              risk_free: np.ndarray, k: float = 600.0) -> dict:
    """Evaluate predictions on validation fold."""
    positions = np.clip(1.0 + y_pred * k, 0.0, 2.0)
    strat = risk_free * (1.0 - positions) + fwd_returns * positions
    excess = strat - risk_free

    mkt_vol = float(np.std(fwd_returns))
    strat_vol = float(np.std(strat))

    vol_ratio = strat_vol / mkt_vol if mkt_vol > 0 else np.nan
    sharpe = (np.mean(excess) / strat_vol) * np.sqrt(252) if strat_vol > 0 else 0.0
    mse = float(np.mean((y_true - y_pred) ** 2))

    return {
        "mse": mse,
        "vol_ratio": vol_ratio,
        "sharpe": sharpe,
    }


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series,
              train_df: pd.DataFrame, k: float = 600.0) -> float:
    """Optuna objective function for hyperparameter tuning."""

    # Hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0, step=0.1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0, step=0.1),
        'random_state': 42,
        'tree_method': 'hist',
        'verbosity': 0,
        'n_jobs': -1,
    }

    # 3-fold CV
    tscv = TimeSeriesSplit(n_splits=3)
    sharpes = []

    for tr_idx, va_idx in tscv.split(X):
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]

        # Preprocess
        X_tr_filled = X_tr.fillna(X_tr.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_filled)

        X_va_filled = X_va.fillna(X_va.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_va_scaled = scaler.transform(X_va_filled)

        # Train model
        model = XGBRegressor(**params)
        model.fit(X_tr_scaled, y_tr)
        y_pred = model.predict(X_va_scaled)

        # Evaluate
        va_df = train_df.iloc[va_idx]
        metrics = eval_fold(
            y_pred,
            y_va.values,
            va_df["forward_returns"].values,
            va_df["risk_free_rate"].values,
            k=k
        )
        sharpes.append(metrics['sharpe'])

    # Return average Sharpe (maximize)
    return np.mean(sharpes)


def train_final_model(X: pd.DataFrame, y: pd.Series, train_df: pd.DataFrame,
                     best_params: dict, k: float = 600.0) -> tuple:
    """Train final model with best parameters and evaluate."""
    print(f"\n{'='*80}")
    print("Training final model with best parameters...")
    print(f"{'='*80}")

    tscv = TimeSeriesSplit(n_splits=3)
    results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        print(f"  Fold {fold_idx}/3...", end=" ", flush=True)

        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]

        # Preprocess
        X_tr_filled = X_tr.fillna(X_tr.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_filled)

        X_va_filled = X_va.fillna(X_va.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_va_scaled = scaler.transform(X_va_filled)

        # Train model
        model = XGBRegressor(**best_params)
        model.fit(X_tr_scaled, y_tr)
        y_pred = model.predict(X_va_scaled)

        # Evaluate
        va_df = train_df.iloc[va_idx]
        metrics = eval_fold(
            y_pred,
            y_va.values,
            va_df["forward_returns"].values,
            va_df["risk_free_rate"].values,
            k=k
        )
        metrics.update({
            "fold": fold_idx,
            "n_features": X.shape[1]
        })
        results.append(metrics)

        print(f"Sharpe: {metrics['sharpe']:.3f}")

    df_results = pd.DataFrame(results)
    avg_sharpe = df_results['sharpe'].mean()
    std_sharpe = df_results['sharpe'].std()
    print(f"\n  Average Sharpe: {avg_sharpe:.3f} ± {std_sharpe:.3f}")

    return df_results, model


def main():
    print("="*80)
    print("EXP-016 Phase 3.3: Hyperparameter Tuning")
    print("="*80)
    print()
    print("Baseline: Top 20 features (Sharpe 0.874)")
    print("Method: Optuna TPE")
    print("Goal: Sharpe 0.95~1.0+")
    print()

    # Create output directory
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading data...")
    X_full, y, all_feature_names = load_exp007_features()
    train = pd.read_csv("data/train.csv")

    # Load Top 20 features
    with open(output_dir / 'top_50_common_features.txt', 'r') as f:
        top_50 = [line.strip() for line in f if line.strip()]

    top_20_features = top_50[:20]
    print(f"Top 20 features loaded: {len(top_20_features)}")
    print()

    # Baseline check
    print("="*80)
    print("Baseline Check (Current hyperparameters)")
    print("="*80)

    baseline_params = {
        'n_estimators': 300,
        'learning_rate': 0.01,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'random_state': 42,
        'tree_method': 'hist',
        'verbosity': 0,
        'n_jobs': -1,
    }

    X_top20 = X_full[top_20_features]
    baseline_results, _ = train_final_model(X_top20, y, train, baseline_params)
    baseline_sharpe = baseline_results['sharpe'].mean()
    print(f"\nBaseline Sharpe: {baseline_sharpe:.3f}")

    # Optuna optimization
    print("\n" + "="*80)
    print("Hyperparameter Optimization with Optuna")
    print("="*80)
    print("Trials: 200")
    print("Sampler: TPE (Tree-structured Parzen Estimator)")
    print()

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='xgboost_sharpe_optimization'
    )

    # Optimize
    print("Starting optimization...")
    study.optimize(
        lambda trial: objective(trial, X_top20, y, train),
        n_trials=200,
        show_progress_bar=True,
    )

    # Best parameters
    print("\n" + "="*80)
    print("Optimization Complete!")
    print("="*80)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best CV Sharpe: {study.best_value:.3f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Train final model with best params
    best_params = study.best_params.copy()
    best_params.update({
        'random_state': 42,
        'tree_method': 'hist',
        'verbosity': 0,
        'n_jobs': -1,
    })

    final_results, final_model = train_final_model(X_top20, y, train, best_params)
    final_sharpe = final_results['sharpe'].mean()

    # Results comparison
    print("\n" + "="*80)
    print("Results Comparison")
    print("="*80)

    diff = final_sharpe - baseline_sharpe
    pct = 100 * diff / baseline_sharpe if baseline_sharpe != 0 else 0

    print(f"\nBaseline (default params):  Sharpe {baseline_sharpe:.3f}")
    print(f"Optimized (best params):    Sharpe {final_sharpe:.3f}")
    print(f"Difference:                 {diff:+.3f} ({pct:+.1f}%)")
    print()

    # Save results
    all_results = pd.concat([
        baseline_results.assign(experiment='baseline'),
        final_results.assign(experiment='optimized')
    ], ignore_index=True)
    all_results.to_csv(output_dir / 'phase_3_3_results.csv', index=False)

    # Save best params
    best_params_df = pd.DataFrame([best_params])
    best_params_df.to_csv(output_dir / 'best_hyperparameters.csv', index=False)

    # Save optimization history
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / 'optuna_trials.csv', index=False)

    print("="*80)
    print("Phase 3.3 Complete!")
    print("="*80)
    print()

    # Decision
    if final_sharpe >= 1.0:
        print("🎉 **SUCCESS! Sharpe 1.0+ achieved!**")
        print("   Next: Final validation and Kaggle submission")
    elif final_sharpe >= 0.95:
        print("📊 **Very close!** (Sharpe 0.95+)")
        print(f"   Sharpe: {baseline_sharpe:.3f} → {final_sharpe:.3f}")
        print("   Next: Consider ensemble or accept current result")
    elif final_sharpe > baseline_sharpe:
        print("✅ **Improvement!** Hyperparameter tuning helped.")
        print(f"   Sharpe: {baseline_sharpe:.3f} → {final_sharpe:.3f}")
        print("   Next: Consider ensemble to reach 1.0")
    else:
        print("⚠️ **No improvement.** Default parameters were already good.")
        print("   Next: Consider ensemble or accept Sharpe 0.874")

    print()
    print(f"Results saved to: {output_dir}/")
    print("  - phase_3_3_results.csv")
    print("  - best_hyperparameters.csv")
    print("  - optuna_trials.csv")
    print("="*80)

    # Summary for next steps
    print()
    print("="*80)
    print("Overall Progress Summary")
    print("="*80)
    print(f"EXP-007 baseline:           Sharpe 0.749")
    print(f"Phase 1 (Top 20 features):  Sharpe 0.874 (+16.7%)")
    print(f"Phase 3 (Optimized):        Sharpe {final_sharpe:.3f} ({(final_sharpe/0.749-1)*100:+.1f}% vs EXP-007)")
    print()
    print(f"Target (Sharpe 1.0):        {100*(1.0-final_sharpe)/final_sharpe:.1f}% gap remaining")
    print("="*80)


if __name__ == '__main__':
    main()
