#!/usr/bin/env python3
"""
EXP-019 Phase 3: Ensemble with Kelly Criterion and Quantile Strategy

Goal: Combine multiple models with advanced position sizing for 10+ score
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import pickle

print("="*80)
print("EXP-019 Phase 3: Ensemble Strategy with Kelly Criterion")
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

# Helper function to create all features
def create_all_features(df):
    """Create all 284 features"""
    from itertools import combinations
    from scipy.stats import skew, kurtosis

    X_base = df[top_20_base].fillna(0).replace([np.inf, -np.inf], 0)
    features = {}
    eps = 1e-8

    top_10 = top_20_base[:10]
    top_5 = top_20_base[:5]
    top_8 = top_20_base[:8]
    top_6 = top_20_base[:6]

    # 2-way
    for i, feat1 in enumerate(top_10):
        for feat2 in top_10[i+1:]:
            features[f'{feat1}*{feat2}'] = X_base[feat1] * X_base[feat2]
            features[f'{feat1}/{feat2}'] = X_base[feat1] / (X_base[feat2].abs() + eps)
    for feat in top_5:
        features[f'{feat}²'] = X_base[feat] ** 2
        features[f'{feat}³'] = X_base[feat] ** 3
        features[f'{feat}⁴'] = X_base[feat] ** 4

    # 3-way
    for feat1, feat2, feat3 in combinations(top_8, 3):
        features[f'{feat1}*{feat2}*{feat3}'] = X_base[feat1] * X_base[feat2] * X_base[feat3]
        if feat1 in top_5 and feat2 in top_5:
            features[f'({feat1}*{feat2})/{feat3}'] = (X_base[feat1] * X_base[feat2]) / (X_base[feat3].abs() + eps)

    # 4-way
    for feat1, feat2, feat3, feat4 in combinations(top_6, 4):
        features[f'{feat1}*{feat2}*{feat3}*{feat4}'] = X_base[feat1] * X_base[feat2] * X_base[feat3] * X_base[feat4]

    # Meta
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

    # Ratio
    for feat1 in top_5:
        for feat2 in top_5:
            if feat1 != feat2:
                features[f'ratio_{feat1}_to_{feat2}'] = X_base[feat1] / (X_base[feat2].abs() + eps)

    return pd.DataFrame(features).replace([np.inf, -np.inf], np.nan).fillna(0)

# Create features
print("[2] Creating features...")
X_all = create_all_features(train)
vol_proxy = X_all['vol_mean_V'].to_numpy()

# Load trained models
print("[3] Loading trained models...")
with open("experiments/019/results/trained_models.pkl", "rb") as f:
    trained_models = pickle.load(f)

print(f"Loaded {len(trained_models)} models")

# Sort by performance and select top 5
trained_models = sorted(trained_models, key=lambda x: x['mean_sharpe'], reverse=True)
top_5_models = trained_models[:5]

print("\nTop 5 Models for Ensemble:")
for i, m in enumerate(top_5_models, 1):
    print(f"  {i}. {m['name']:20s} Sharpe={m['mean_sharpe']:.4f}")
print()

# Volatility regime thresholds
vol_75 = np.percentile(vol_proxy, 75)
vol_25 = np.percentile(vol_proxy, 25)

print(f"Volatility regime thresholds:")
print(f"  High vol: > {vol_75:.4f}")
print(f"  Low vol:  < {vol_25:.4f}")
print()

# Ensemble prediction function with Kelly Criterion
def ensemble_predict_kelly(models, X_all_row, vol_proxy_val):
    """
    Ensemble prediction with regime-based weighting and Kelly Criterion

    Args:
        models: List of model dicts
        X_all_row: DataFrame with all features (single row)
        vol_proxy_val: Volatility proxy value

    Returns:
        Final position
    """
    eps = 1e-8

    # Get predictions from all models
    predictions = []
    k_values = []

    for model_dict in models:
        model = model_dict['model']
        scaler = model_dict['scaler']
        features = model_dict['features']
        k = model_dict['k']
        sharpe = model_dict['mean_sharpe']

        # Extract features
        X = X_all_row[features].values.reshape(1, -1)
        X_scaled = scaler.transform(X)

        # Predict
        pred = model.predict(X_scaled)[0]

        predictions.append(pred)
        k_values.append(k)

    # Regime-based weighting
    if vol_proxy_val > vol_75:  # High volatility
        # Favor conservative models (lower K, lower index in sorted list)
        weights = np.array([3.0, 2.0, 1.0, 0.5, 0.25])  # Favor first models
    elif vol_proxy_val < vol_25:  # Low volatility
        # More balanced, slightly favor aggressive
        weights = np.array([2.0, 2.0, 1.5, 1.0, 0.5])
    else:  # Normal volatility
        # Balanced weights based on performance
        weights = np.array([m['mean_sharpe'] for m in models])

    weights = weights / weights.sum()

    # Weighted average prediction
    excess_pred = np.average(predictions, weights=weights)

    # Kelly Criterion inspired position sizing
    # Estimate win probability from prediction magnitude
    confidence = abs(excess_pred)
    win_prob = 1.0 / (1.0 + np.exp(-excess_pred * 1000))  # Sigmoid

    # Kelly fraction: f = p - (1-p) = 2p - 1
    kelly_fraction = 2 * win_prob - 1.0

    # Quantile-based K adjustment
    pred_abs = abs(excess_pred)
    percentile_90 = 0.0015  # Approximate based on training data
    percentile_10 = 0.00005

    if pred_abs > percentile_90:  # Very confident
        k_multiplier = 2.0
    elif pred_abs < percentile_10:  # Very uncertain
        k_multiplier = 0.5
    else:  # Normal
        k_multiplier = 1.0

    # Dynamic K based on volatility (inverse relationship)
    vol_adjustment = 1.0 / (1.0 + 1.0 * vol_proxy_val)

    # Base K (weighted average from models)
    k_base = np.average(k_values, weights=weights)

    # Final K
    k_final = k_base * vol_adjustment * k_multiplier

    # Position with Kelly adjustment
    position_base = 1.0 + excess_pred * k_final
    position_kelly_adjusted = 1.0 + (excess_pred * k_final) * abs(kelly_fraction)

    # Clip to [0, 2]
    position = np.clip(position_kelly_adjusted, 0.0, 2.0)

    return position, excess_pred, k_final, kelly_fraction

# Sharpe calculation
def calculate_sharpe(positions, fwd_returns, risk_free):
    """Calculate Sharpe ratio from positions"""
    strategy_returns = risk_free * (1.0 - positions) + fwd_returns * positions
    excess_returns = strategy_returns - risk_free

    if np.std(strategy_returns) > 1e-8:
        sharpe = (np.mean(excess_returns) / np.std(strategy_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    return sharpe

# Evaluate ensemble with 5-fold CV
print("[4] Evaluating ensemble with 5-fold CV...")
tscv = TimeSeriesSplit(n_splits=5)

fold_results = []

for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X_all), 1):
    print(f"\nFold {fold_idx}/5:")
    print("-" * 40)

    # Validation set
    X_va_all = X_all.iloc[va_idx]
    fwd_va = fwd_returns.iloc[va_idx]
    rf_va = risk_free.iloc[va_idx]
    vol_va = vol_proxy[va_idx]

    # Generate predictions
    positions = []

    for i in range(len(X_va_all)):
        X_row = X_va_all.iloc[[i]]
        vol_val = vol_va[i]

        pos, _, _, _ = ensemble_predict_kelly(top_5_models, X_row, vol_val)
        positions.append(pos)

    positions = np.array(positions)

    # Calculate Sharpe
    sharpe = calculate_sharpe(positions, fwd_va.values, rf_va.values)

    fold_results.append({
        'fold': fold_idx,
        'sharpe': sharpe,
        'position_mean': positions.mean(),
        'position_std': positions.std()
    })

    print(f"  Sharpe: {sharpe:.4f}")
    print(f"  Position: {positions.mean():.4f} ± {positions.std():.4f}")

print()
print("="*80)
print("Ensemble Results")
print("="*80)

fold_df = pd.DataFrame(fold_results)

print(f"\nSharpe by fold:")
for idx, row in fold_df.iterrows():
    print(f"  Fold {int(row['fold'])}: {row['sharpe']:.4f}")

print(f"\nOverall Statistics:")
print(f"  Mean Sharpe: {fold_df['sharpe'].mean():.4f} ± {fold_df['sharpe'].std():.4f}")
print(f"  Min Sharpe:  {fold_df['sharpe'].min():.4f}")
print(f"  Max Sharpe:  {fold_df['sharpe'].max():.4f}")
print()

# Comparison
print("Comparison:")
print("-" * 40)
print(f"EXP-016 (baseline):     0.559 ± 0.362")
print(f"EXP-018 (dynamic K):    0.582 ± 0.358")
print(f"Best single model:      {top_5_models[0]['mean_sharpe']:.4f}")
print(f"EXP-019 (ensemble):     {fold_df['sharpe'].mean():.4f} ± {fold_df['sharpe'].std():.4f}")
print()

improvement_vs_016 = (fold_df['sharpe'].mean() - 0.559) / 0.559 * 100
improvement_vs_best = (fold_df['sharpe'].mean() - top_5_models[0]['mean_sharpe']) / top_5_models[0]['mean_sharpe'] * 100

print(f"Improvement vs EXP-016: {improvement_vs_016:+.2f}%")
print(f"Improvement vs best single: {improvement_vs_best:+.2f}%")
print()

# Expected Public Score
print("Expected Public Score (based on EXP-016 CV-to-Public ratio):")
print("-" * 40)
cv_to_public_ratio = 4.440 / 0.559  # From EXP-016
expected_public = fold_df['sharpe'].mean() * cv_to_public_ratio
expected_public_conservative = fold_df['sharpe'].mean() * 6.0  # Conservative estimate
expected_public_optimistic = fold_df['sharpe'].mean() * 10.0  # Optimistic estimate

print(f"  Conservative (6x):   {expected_public_conservative:.2f}")
print(f"  Expected (7.9x):     {expected_public:.2f}")
print(f"  Optimistic (10x):    {expected_public_optimistic:.2f}")
print()

# Save results
fold_df.to_csv("experiments/019/results/ensemble_cv_results.csv", index=False)

# Save configuration
config = {
    'ensemble_models': [m['name'] for m in top_5_models],
    'mean_sharpe': fold_df['sharpe'].mean(),
    'std_sharpe': fold_df['sharpe'].std(),
    'expected_public_score': expected_public,
    'vol_regime_thresholds': {'high': vol_75, 'low': vol_25}
}

pd.DataFrame([config]).to_csv("experiments/019/results/ensemble_config.csv", index=False)

print(f"Results saved to experiments/019/results/")
print()

print("="*80)
print("Phase 3 Complete!")
print("="*80)
