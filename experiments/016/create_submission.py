#!/usr/bin/env python3
"""EXP-016 Submission File Creation

Create submission file with best model:
- Features: Top 20
- Hyperparameters: Optimized from Phase 3.3
- Expected Sharpe: ~0.78 (5-fold CV estimate)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import from feature_analysis
sys.path.insert(0, str(Path(__file__).parent))
from feature_analysis import load_exp007_features

try:
    from xgboost import XGBRegressor
except ImportError:
    print("[ERROR] XGBoost not available")
    sys.exit(1)


def main():
    print("="*80)
    print("EXP-016 Submission File Creation")
    print("="*80)
    print()
    print("Model Configuration:")
    print("  - Features: Top 20")
    print("  - Hyperparameters: Optimized (from Phase 3.3)")
    print("  - Expected Performance: Sharpe ~0.78 (5-fold CV)")
    print()

    # Create output directory
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    # Load train data
    print("Loading training data...")
    X_full, y, all_feature_names = load_exp007_features()
    train = pd.read_csv("data/train.csv")

    # Load Top 20 features
    with open(output_dir / 'top_50_common_features.txt', 'r') as f:
        top_50 = [line.strip() for line in f if line.strip()]

    top_20_features = top_50[:20]
    print(f"Top 20 features loaded: {len(top_20_features)}")

    # Print features
    print("\nTop 20 Features:")
    for i, feat in enumerate(top_20_features, 1):
        print(f"  {i:2d}. {feat}")
    print()

    # Load best hyperparameters
    best_params_df = pd.read_csv(output_dir / 'best_hyperparameters.csv')
    best_params = best_params_df.iloc[0].to_dict()

    # Convert to proper types
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])
    best_params['random_state'] = int(best_params['random_state'])
    best_params['verbosity'] = int(best_params['verbosity'])
    best_params['n_jobs'] = int(best_params['n_jobs'])

    print("Best hyperparameters:")
    for key, value in best_params.items():
        if key not in ['random_state', 'tree_method', 'verbosity', 'n_jobs']:
            print(f"  {key}: {value}")
    print()

    # Prepare training data
    X_train = X_full[top_20_features]

    # Preprocess
    print("Preprocessing training data...")
    X_train_filled = X_train.fillna(X_train.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filled)

    # Train final model on full training data
    print("Training final model on full training data...")
    model = XGBRegressor(**best_params)
    model.fit(X_train_scaled, y)
    print("Training complete!")
    print()

    # Load test data
    print("Loading test data...")
    try:
        test = pd.read_csv("data/test.csv")
        print(f"Test data loaded: {len(test)} samples")
    except FileNotFoundError:
        print("[ERROR] Test data not found at data/test.csv")
        print("Please make sure test.csv exists")
        sys.exit(1)

    # Create features for test data
    print("Creating features for test data...")

    # Import feature engineering from EXP-007
    sys.path.insert(0, str(Path(__file__).parent.parent / '007'))
    from feature_engineering import create_all_features

    # Get base features (exclude date_id if exists)
    base_cols = [col for col in test.columns if col not in ['id', 'date', 'date_id']]
    print(f"Base features in test: {len(base_cols)}")

    # Save date_id if exists
    if 'date_id' in test.columns:
        test_ids = test['date_id'].copy()
    elif 'id' in test.columns:
        test_ids = test['id'].copy()
    else:
        test_ids = pd.Series(range(len(test)), name='id')
        print("WARNING: No id column found, using row numbers")

    # Create all features
    X_test_full = create_all_features(test, base_cols)
    print(f"Total features created: {X_test_full.shape[1]}")

    # Select Top 20 features
    X_test = X_test_full[top_20_features]
    print(f"Selected Top 20 features")
    print()

    # Preprocess test data
    print("Preprocessing test data...")
    X_test_filled = X_test.fillna(X_test.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_scaled = scaler.transform(X_test_filled)

    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test_scaled)
    print(f"Predictions made: {len(predictions)} samples")
    print()

    # Prediction statistics
    print("Prediction Statistics:")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std:  {predictions.std():.6f}")
    print(f"  Min:  {predictions.min():.6f}")
    print(f"  Max:  {predictions.max():.6f}")
    print()

    # Create submission file
    submission = pd.DataFrame({
        'date_id': test_ids,
        'prediction': predictions
    })

    # Save submission
    submission_path = Path('submission_exp016.csv')
    submission.to_csv(submission_path, index=False)

    print("="*80)
    print("Submission File Created!")
    print("="*80)
    print(f"File: {submission_path.absolute()}")
    print(f"Rows: {len(submission)}")
    print()

    # Show first few rows
    print("First 10 rows:")
    print(submission.head(10).to_string(index=False))
    print()

    print("="*80)
    print("Ready for Kaggle submission!")
    print("="*80)
    print()
    print("Model Details:")
    print(f"  - Top 20 features (from 754)")
    print(f"  - Optimized XGBoost hyperparameters")
    print(f"  - Expected Sharpe: ~0.78 (5-fold CV)")
    print(f"  - Training Sharpe: 0.874 (Phase 1)")
    print()
    print("Next Steps:")
    print("  1. Submit to Kaggle: submission_exp016.csv")
    print("  2. Check Public LB score")
    print("  3. Compare with CV estimate (0.78)")
    print("="*80)


if __name__ == '__main__':
    main()
