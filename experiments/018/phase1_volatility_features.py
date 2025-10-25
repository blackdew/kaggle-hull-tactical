#!/usr/bin/env python3
"""
EXP-018 Phase 1: Volatility Proxy Features Development

Goal: Create 1-row calculable volatility features for dynamic K strategy
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

print("="*80)
print("EXP-018 Phase 1: Volatility Proxy Features Development")
print("="*80)
print()

# Load data
print("[1] Loading data...")
train = pd.read_csv("data/train.csv")
print(f"Train data shape: {train.shape}")

# Load top 20 base features from EXP-016
top_20_base = [
    'M4', 'V13', 'V7', 'P8', 'S2', 'I2', 'E19', 'S5', 'P5', 'P7',
    'M2', 'V9', 'M3', 'P12', 'P10', 'V10', 'E12', 'P11', 'M12', 'S8'
]

# Fill missing values
X_base = train[top_20_base].fillna(train[top_20_base].median()).replace([np.inf, -np.inf], 0)
y = train['market_forward_excess_returns'].to_numpy()

print(f"Base features: {X_base.shape}")
print()

# Create volatility proxy features (all 1-row calculable)
print("[2] Creating volatility proxy features...")

volatility_features = {}
eps = 1e-8

# 1. Composite Volatility Measures
print("  - Composite volatility measures...")
# V-category composite
volatility_features['vol_composite_V'] = np.sqrt(X_base['V13']**2 + X_base['V7']**2 + X_base['V9']**2 + eps)
volatility_features['vol_composite_V_weighted'] = np.sqrt(3*X_base['V13']**2 + 2*X_base['V7']**2 + X_base['V9']**2 + eps)
volatility_features['vol_mean_V'] = (abs(X_base['V13']) + abs(X_base['V7']) + abs(X_base['V9'])) / 3

# Add V10 to composite
volatility_features['vol_composite_V_all'] = np.sqrt(X_base['V13']**2 + X_base['V7']**2 + X_base['V9']**2 + X_base['V10']**2 + eps)

# 2. Market Turbulence
print("  - Market turbulence measures...")
volatility_features['market_turbulence'] = np.sqrt(X_base['M4']**2 + X_base['M2']**2 + X_base['M3']**2 + eps)
volatility_features['market_turbulence_weighted'] = np.sqrt(3*X_base['M4']**2 + 2*X_base['M2']**2 + X_base['M3']**2 + eps)
volatility_features['market_abs_sum'] = abs(X_base['M4']) + abs(X_base['M2']) + abs(X_base['M3'])

# Add M12
volatility_features['market_turbulence_all'] = np.sqrt(X_base['M4']**2 + X_base['M2']**2 + X_base['M3']**2 + X_base['M12']**2 + eps)

# 3. Price Dispersion
print("  - Price dispersion measures...")
volatility_features['price_dispersion'] = abs(X_base['P8'] - X_base['P7']) + abs(X_base['P7'] - X_base['P5'])
volatility_features['price_range'] = abs(X_base['P8'] - X_base['P5'])
volatility_features['price_volatility'] = np.sqrt((X_base['P8'] - X_base['P7'])**2 + (X_base['P7'] - X_base['P5'])**2 + eps)

# Add more price features
volatility_features['price_dispersion_all'] = (abs(X_base['P8'] - X_base['P7']) + abs(X_base['P8'] - X_base['P5']) +
                                               abs(X_base['P7'] - X_base['P5']) + abs(X_base['P10'] - X_base['P12']))

# 4. Cross-Category Volatility
print("  - Cross-category volatility...")
volatility_features['cross_vol_MV'] = np.sqrt(X_base['M4']**2 + X_base['V13']**2 + eps)
volatility_features['cross_vol_MVP'] = np.sqrt(X_base['M4']**2 + X_base['V13']**2 + X_base['P8']**2 + eps)
volatility_features['cross_vol_all'] = np.sqrt(X_base['M4']**2 + X_base['V13']**2 + X_base['P8']**2 + X_base['S2']**2 + eps)

# 5. Sentiment Dispersion
print("  - Sentiment dispersion...")
volatility_features['sentiment_dispersion'] = abs(X_base['S2'] - X_base['S5'])
volatility_features['sentiment_volatility'] = np.sqrt(X_base['S2']**2 + X_base['S5']**2 + X_base['S8']**2 + eps)
volatility_features['sentiment_abs_sum'] = abs(X_base['S2']) + abs(X_base['S5']) + abs(X_base['S8'])

# 6. Market-Volatility Ratio
print("  - Market-volatility ratios...")
volatility_features['market_vol_ratio'] = abs(X_base['M4']) / (volatility_features['vol_composite_V'] + eps)
volatility_features['vol_market_ratio'] = volatility_features['vol_composite_V'] / (abs(X_base['M4']) + eps)

# 7. Normalized Composite
print("  - Normalized composites...")
volatility_features['normalized_vol'] = volatility_features['vol_composite_V'] / (abs(X_base['M4']) + abs(X_base['P8']) + eps)
volatility_features['total_magnitude'] = np.sqrt(
    X_base['M4']**2 + X_base['V13']**2 + X_base['P8']**2 + X_base['S2']**2 + X_base['I2']**2 + eps
)

# 8. Coefficient of Variation (proxy)
print("  - Coefficient of variation proxies...")
M_features = ['M4', 'M2', 'M3', 'M12']
V_features = ['V13', 'V7', 'V9', 'V10']
P_features = ['P8', 'P7', 'P5', 'P12']

M_mean = X_base[M_features].mean(axis=1)
M_std = X_base[M_features].std(axis=1)
volatility_features['M_cv'] = M_std / (abs(M_mean) + eps)

V_mean = X_base[V_features].mean(axis=1)
V_std = X_base[V_features].std(axis=1)
volatility_features['V_cv'] = V_std / (abs(V_mean) + eps)

P_mean = X_base[P_features].mean(axis=1)
P_std = X_base[P_features].std(axis=1)
volatility_features['P_cv'] = P_std / (abs(P_mean) + eps)

# 9. Feature Stability
print("  - Feature stability measures...")
volatility_features['feature_stability'] = 1.0 / (volatility_features['total_magnitude'] + eps)
volatility_features['market_stability'] = 1.0 / (volatility_features['market_turbulence'] + eps)

print(f"Created {len(volatility_features)} volatility features")
print()

# Convert to DataFrame
X_vol = pd.DataFrame(volatility_features)
X_vol = X_vol.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Volatility features shape: {X_vol.shape}")
print()

# Evaluate importance using RandomForest
print("[3] Evaluating feature importance...")
scaler = StandardScaler()
X_vol_scaled = scaler.fit_transform(X_vol)

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_vol_scaled, y)

importances = rf.feature_importances_
feature_names = list(X_vol.columns)

# Calculate correlation with target
correlations = []
for col in X_vol.columns:
    corr = np.corrcoef(X_vol[col], y)[0, 1]
    correlations.append(abs(corr) if not np.isnan(corr) else 0.0)

# Combine scores
results = pd.DataFrame({
    'feature': feature_names,
    'importance': importances,
    'abs_correlation': correlations,
    'combined_score': importances * 0.7 + np.array(correlations) * 0.3
})

results = results.sort_values('combined_score', ascending=False).reset_index(drop=True)

print("\nTop 15 Volatility Features:")
print(results.head(15).to_string(index=False))
print()

# Save results
print("[4] Saving results...")
results.to_csv("experiments/018/results/volatility_features_ranking.csv", index=False)

# Save top 10 features
top_10_vol = results.head(10)
top_10_vol.to_csv("experiments/018/results/top_10_volatility_features.csv", index=False)

print(f"Saved to experiments/018/results/")
print()

# Summary statistics
print("[5] Summary Statistics")
print("-" * 80)
print(f"Total volatility features created: {len(volatility_features)}")
print(f"Top 10 features average importance: {top_10_vol['importance'].mean():.6f}")
print(f"Top 10 features average correlation: {top_10_vol['abs_correlation'].mean():.6f}")
print()

print("Top 10 Volatility Features:")
for idx, row in top_10_vol.iterrows():
    print(f"  {idx+1:2d}. {row['feature']:30s} (score: {row['combined_score']:.6f})")

print()
print("="*80)
print("Phase 1 Complete!")
print("="*80)
