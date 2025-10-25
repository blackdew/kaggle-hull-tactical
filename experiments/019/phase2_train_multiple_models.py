#!/usr/bin/env python3
"""
EXP-019 Phase 2: Train Multiple Models for Ensemble

Goal: Train 10+ models with different configurations
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
import pickle

print("="*80)
print("EXP-019 Phase 2: Train Multiple Models")
print("="*80)
print()

# Load data
print("[1] Loading data...")
train = pd.read_csv("data/train.csv")
y = train['market_forward_excess_returns'].copy()
fwd_returns = train['forward_returns'].copy()
risk_free = train['risk_free_rate'].copy()

# Base features
top_20_base = [
    'M4', 'V13', 'V7', 'P8', 'S2', 'I2', 'E19', 'S5', 'P5', 'P7',
    'M2', 'V9', 'M3', 'P12', 'P10', 'V10', 'E12', 'P11', 'M12', 'S8'
]

print(f"Train shape: {train.shape}")
print()

# Helper function to create all features
def create_all_features(df):
    """Create all 284 features (same as phase1)"""
    from itertools import combinations
    from scipy.stats import skew, kurtosis

    # Fill base
    X_base = df[top_20_base].fillna(0).replace([np.inf, -np.inf], 0)

    features = {}
    eps = 1e-8

    top_10 = top_20_base[:10]
    top_5 = top_20_base[:5]
    top_8 = top_20_base[:8]
    top_6 = top_20_base[:6]

    # 2-way interactions
    for i, feat1 in enumerate(top_10):
        for feat2 in top_10[i+1:]:
            features[f'{feat1}*{feat2}'] = X_base[feat1] * X_base[feat2]
            features[f'{feat1}/{feat2}'] = X_base[feat1] / (X_base[feat2].abs() + eps)

    for feat in top_5:
        features[f'{feat}²'] = X_base[feat] ** 2
        features[f'{feat}³'] = X_base[feat] ** 3
        features[f'{feat}⁴'] = X_base[feat] ** 4

    # 3-way interactions
    for feat1, feat2, feat3 in combinations(top_8, 3):
        features[f'{feat1}*{feat2}*{feat3}'] = X_base[feat1] * X_base[feat2] * X_base[feat3]
        if feat1 in top_5 and feat2 in top_5:
            features[f'({feat1}*{feat2})/{feat3}'] = (X_base[feat1] * X_base[feat2]) / (X_base[feat3].abs() + eps)

    # 4-way interactions
    for feat1, feat2, feat3, feat4 in combinations(top_6, 4):
        features[f'{feat1}*{feat2}*{feat3}*{feat4}'] = X_base[feat1] * X_base[feat2] * X_base[feat3] * X_base[feat4]

    # Meta features
    M_features = ['M4', 'M2', 'M3', 'M12']
    V_features = ['V13', 'V7', 'V9', 'V10']
    P_features = ['P8', 'P7', 'P5', 'P12', 'P10', 'P11']
    S_features = ['S2', 'S5', 'S8']

    for cat_name, cat_feats in [('M', M_features), ('V', V_features), ('P', P_features), ('S', S_features)]:
        cat_data = X_base[cat_feats]
        features[f'{cat_name}_mean'] = cat_data.mean(axis=1)
        features[f'{cat_name}_std'] = cat_data.std(axis=1)
        features[f'{cat_name}_min'] = cat_data.min(axis=1)
        features[f'{cat_name}_max'] = cat_data.max(axis=1)
        features[f'{cat_name}_range'] = features[f'{cat_name}_max'] - features[f'{cat_name}_min']
        features[f'{cat_name}_skew'] = cat_data.apply(lambda x: skew(x, nan_policy='omit'), axis=1)
        features[f'{cat_name}_kurt'] = cat_data.apply(lambda x: kurtosis(x, nan_policy='omit'), axis=1)
        features[f'{cat_name}_cv'] = features[f'{cat_name}_std'] / (features[f'{cat_name}_mean'].abs() + eps)

    # Cross-category
    features['market_strength'] = features['M_mean'] / (features['V_mean'] + eps)
    features['risk_adjusted_return'] = features['P_mean'] / (features['V_mean'] + eps)
    features['sentiment_to_volatility'] = features['S_mean'] / (features['V_mean'] + eps)
    features['market_to_price'] = features['M_mean'] / (features['P_mean'] + eps)
    features['total_volatility'] = np.sqrt(features['V_mean']**2 + features['M_std']**2 + eps)
    features['market_turbulence'] = features['M_std'] * features['V_std']
    features['price_momentum'] = features['P_max'] - features['P_min']
    features['normalized_market'] = features['M_mean'] / (features['total_volatility'] + eps)
    features['normalized_sentiment'] = features['S_mean'] / (features['total_volatility'] + eps)

    # Volatility
    features['vol_mean_V'] = (abs(X_base['V13']) + abs(X_base['V7']) + abs(X_base['V9'])) / 3
    features['vol_composite_V_all'] = np.sqrt(X_base['V13']**2 + X_base['V7']**2 + X_base['V9']**2 + X_base['V10']**2 + eps)
    features['vol_composite_V'] = np.sqrt(X_base['V13']**2 + X_base['V7']**2 + X_base['V9']**2 + eps)
    features['cross_vol_MV'] = np.sqrt(X_base['M4']**2 + X_base['V13']**2 + eps)
    features['sentiment_dispersion'] = abs(X_base['S2'] - X_base['S5'])
    features['market_vol_composite'] = np.sqrt(X_base['M4']**2 + X_base['M2']**2 + X_base['M3']**2 + eps)
    features['price_vol'] = np.sqrt((X_base['P8'] - X_base['P7'])**2 + (X_base['P7'] - X_base['P5'])**2 + eps)

    # Ratio features
    for feat1 in top_5:
        for feat2 in top_5:
            if feat1 != feat2:
                features[f'ratio_{feat1}_to_{feat2}'] = X_base[feat1] / (X_base[feat2].abs() + eps)

    return pd.DataFrame(features).replace([np.inf, -np.inf], np.nan).fillna(0)

# Create all features
print("[2] Creating all features...")
X_all = create_all_features(train)
print(f"All features: {X_all.shape}")

# Load top features
top_100_df = pd.read_csv("experiments/019/results/top_100_features.csv")
top_100_features = top_100_df['feature'].tolist()
top_50_features = top_100_features[:50]
top_30_features = top_100_features[:30]

print(f"Top 100 features loaded")
print()

# Volatility proxy for regime identification
vol_proxy = X_all['vol_mean_V'].to_numpy()
vol_75 = np.percentile(vol_proxy, 75)
vol_25 = np.percentile(vol_proxy, 25)

# Sharpe calculation function
def calculate_sharpe(y_pred, fwd_returns, risk_free, k=250):
    """Calculate Sharpe ratio"""
    positions = np.clip(1.0 + y_pred * k, 0.0, 2.0)
    strategy_returns = risk_free * (1.0 - positions) + fwd_returns * positions
    excess_returns = strategy_returns - risk_free

    if np.std(strategy_returns) > 1e-8:
        sharpe = (np.mean(excess_returns) / np.std(strategy_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    return sharpe

# Model configurations
print("[3] Defining model configurations...")
model_configs = [
    # K-value variations
    {'name': 'k100_top50', 'k': 100, 'features': top_50_features, 'n_estimators': 150, 'max_depth': 7, 'lr': 0.025},
    {'name': 'k150_top50', 'k': 150, 'features': top_50_features, 'n_estimators': 150, 'max_depth': 7, 'lr': 0.025},
    {'name': 'k250_top50', 'k': 250, 'features': top_50_features, 'n_estimators': 150, 'max_depth': 7, 'lr': 0.025},
    {'name': 'k350_top50', 'k': 350, 'features': top_50_features, 'n_estimators': 150, 'max_depth': 7, 'lr': 0.025},
    {'name': 'k500_top50', 'k': 500, 'features': top_50_features, 'n_estimators': 150, 'max_depth': 7, 'lr': 0.025},

    # Feature set variations
    {'name': 'k250_top30', 'k': 250, 'features': top_30_features, 'n_estimators': 150, 'max_depth': 7, 'lr': 0.025},
    {'name': 'k250_top100', 'k': 250, 'features': top_100_features, 'n_estimators': 200, 'max_depth': 8, 'lr': 0.02},

    # Deep models
    {'name': 'k300_deep', 'k': 300, 'features': top_50_features, 'n_estimators': 250, 'max_depth': 10, 'lr': 0.015},

    # Conservative (high vol optimized)
    {'name': 'k100_conservative', 'k': 100, 'features': top_30_features, 'n_estimators': 100, 'max_depth': 5, 'lr': 0.03},

    # Aggressive (low vol optimized)
    {'name': 'k500_aggressive', 'k': 500, 'features': top_100_features, 'n_estimators': 200, 'max_depth': 9, 'lr': 0.02},
]

print(f"Total model configurations: {len(model_configs)}")
print()

# Train all models
print("[4] Training models with 5-fold CV...")
tscv = TimeSeriesSplit(n_splits=5)

model_results = []
trained_models = []

for idx, config in enumerate(model_configs, 1):
    print(f"\nModel {idx}/{len(model_configs)}: {config['name']}")
    print("-" * 40)

    # Select features
    X = X_all[config['features']].copy()

    fold_sharpes = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X)):
        # Split
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        fwd_va = fwd_returns.iloc[va_idx]
        rf_va = risk_free.iloc[va_idx]

        # Scale
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)

        # Train
        model = XGBRegressor(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['lr'],
            subsample=1.0,
            colsample_bytree=0.6,
            reg_lambda=0.5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_tr_scaled, y_tr)

        # Predict
        y_va_pred = model.predict(X_va_scaled)

        # Evaluate
        sharpe = calculate_sharpe(y_va_pred, fwd_va, rf_va, k=config['k'])
        fold_sharpes.append(sharpe)

    # Average performance
    mean_sharpe = np.mean(fold_sharpes)
    std_sharpe = np.std(fold_sharpes)

    print(f"  Sharpe: {mean_sharpe:.4f} ± {std_sharpe:.4f}")
    print(f"  K: {config['k']}, Features: {len(config['features'])}")

    model_results.append({
        'name': config['name'],
        'k': config['k'],
        'n_features': len(config['features']),
        'n_estimators': config['n_estimators'],
        'max_depth': config['max_depth'],
        'mean_sharpe': mean_sharpe,
        'std_sharpe': std_sharpe,
        'min_sharpe': np.min(fold_sharpes),
        'max_sharpe': np.max(fold_sharpes)
    })

    # Train final model on full data
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X)

    model_full = XGBRegressor(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        learning_rate=config['lr'],
        subsample=1.0,
        colsample_bytree=0.6,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1
    )
    model_full.fit(X_scaled_full, y)

    trained_models.append({
        'name': config['name'],
        'model': model_full,
        'scaler': scaler_full,
        'features': config['features'],
        'k': config['k'],
        'mean_sharpe': mean_sharpe
    })

print()
print("="*80)
print("Training Complete!")
print("="*80)
print()

# Save results
results_df = pd.DataFrame(model_results)
results_df = results_df.sort_values('mean_sharpe', ascending=False).reset_index(drop=True)
results_df.to_csv("experiments/019/results/model_comparison.csv", index=False)

print("Model Performance Ranking:")
print(results_df.to_string(index=False))
print()

# Save trained models
print("[5] Saving trained models...")
with open("experiments/019/results/trained_models.pkl", "wb") as f:
    pickle.dump(trained_models, f)

print(f"Saved {len(trained_models)} models to experiments/019/results/trained_models.pkl")
print()

print("="*80)
print("Phase 2 Complete!")
print("="*80)
print()
print(f"Best model: {results_df.iloc[0]['name']}")
print(f"Best Sharpe: {results_df.iloc[0]['mean_sharpe']:.4f} ± {results_df.iloc[0]['std_sharpe']:.4f}")
