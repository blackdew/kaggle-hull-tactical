"""
EXP-024 Phase 2: Advanced Feature Engineering

Phase 1 분석 결과를 바탕으로 고품질 features 생성
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EXP-024 Phase 2: Advanced Feature Engineering")
print("="*80)

# ============================================================================
# 1. Load Data & Phase 1 Results
# ============================================================================
print("\n[1] Loading data...")
df = pd.read_csv("data/train.csv")

# Load Phase 1 results
phase1_results = pd.read_csv("experiments/024/results/phase1_feature_analysis.csv")

# Top features by MI
top_mi_features = phase1_results.nlargest(30, 'mi_score')['feature'].tolist()
print(f"   Top 30 MI features: {', '.join(top_mi_features[:10])}, ...")

# Categories
categories = {
    'M': [f for f in df.columns if f.startswith('M') and f not in ['market_forward_excess_returns']],
    'V': [f for f in df.columns if f.startswith('V')],
    'P': [f for f in df.columns if f.startswith('P')],
    'S': [f for f in df.columns if f.startswith('S')],
    'I': [f for f in df.columns if f.startswith('I')],
    'E': [f for f in df.columns if f.startswith('E')],
}

# ============================================================================
# 2. DateTime Cyclical Features
# ============================================================================
print("\n[2] Creating DateTime cyclical features...")

date_id = df['date_id'].values
trading_days_per_year = 252

# Normalize to [0, 1]
day_of_year = (date_id % trading_days_per_year) / trading_days_per_year
week_of_year = (date_id % trading_days_per_year) / (trading_days_per_year / 52)
month_of_year = (date_id % trading_days_per_year) / (trading_days_per_year / 12)

# Cyclical encoding
df['day_sin'] = np.sin(2 * np.pi * day_of_year)
df['day_cos'] = np.cos(2 * np.pi * day_of_year)
df['week_sin'] = np.sin(2 * np.pi * week_of_year)
df['week_cos'] = np.cos(2 * np.pi * week_of_year)
df['month_sin'] = np.sin(2 * np.pi * month_of_year)
df['month_cos'] = np.cos(2 * np.pi * month_of_year)

datetime_features = ['day_sin', 'day_cos', 'week_sin', 'week_cos', 'month_sin', 'month_cos']
print(f"   Created {len(datetime_features)} datetime features")

# ============================================================================
# 3. Category-wise Statistics
# ============================================================================
print("\n[3] Creating category-wise statistics...")

category_stat_features = []

for cat_name, cat_features in categories.items():
    if len(cat_features) == 0:
        continue

    cat_data = df[cat_features].fillna(0).values

    # Statistics (1-row calculation)
    df[f'{cat_name}_mean'] = np.mean(cat_data, axis=1)
    df[f'{cat_name}_std'] = np.std(cat_data, axis=1)
    df[f'{cat_name}_min'] = np.min(cat_data, axis=1)
    df[f'{cat_name}_max'] = np.max(cat_data, axis=1)
    df[f'{cat_name}_range'] = df[f'{cat_name}_max'] - df[f'{cat_name}_min']
    df[f'{cat_name}_median'] = np.median(cat_data, axis=1)

    # Skewness & Kurtosis (scipy not available in InferenceServer, use simple proxies)
    # Skew proxy: (mean - median) / std
    df[f'{cat_name}_skew_proxy'] = (df[f'{cat_name}_mean'] - df[f'{cat_name}_median']) / (df[f'{cat_name}_std'] + 1e-8)

    category_stat_features.extend([
        f'{cat_name}_mean', f'{cat_name}_std', f'{cat_name}_min', f'{cat_name}_max',
        f'{cat_name}_range', f'{cat_name}_median', f'{cat_name}_skew_proxy'
    ])

print(f"   Created {len(category_stat_features)} category statistics")

# ============================================================================
# 4. Nonlinear Transformations (Top MI Features)
# ============================================================================
print("\n[4] Creating nonlinear transformations for top MI features...")

# Select top 20 for nonlinear transforms
top_20_for_transform = phase1_results.nlargest(20, 'mi_score')['feature'].tolist()

nonlinear_features = []
eps = 1e-8

for feat in top_20_for_transform:
    if feat not in df.columns:
        continue

    feat_values = df[feat].fillna(0).values

    # Log transform (handle negative values)
    df[f'{feat}_log'] = np.sign(feat_values) * np.log1p(np.abs(feat_values))
    nonlinear_features.append(f'{feat}_log')

    # Square root (handle negative values)
    df[f'{feat}_sqrt'] = np.sign(feat_values) * np.sqrt(np.abs(feat_values))
    nonlinear_features.append(f'{feat}_sqrt')

    # Square
    df[f'{feat}_squared'] = feat_values ** 2
    nonlinear_features.append(f'{feat}_squared')

print(f"   Created {len(nonlinear_features)} nonlinear features")

# ============================================================================
# 5. Volatility Proxies
# ============================================================================
print("\n[5] Creating volatility proxies...")

V_features = categories['V']
if len(V_features) > 0:
    V_data = df[V_features].fillna(0).values

    # Composite volatility
    df['vol_composite'] = np.sqrt(np.sum(V_data ** 2, axis=1))
    df['vol_mean'] = np.mean(V_data, axis=1)
    df['vol_max'] = np.max(V_data, axis=1)

# Cross-category volatility
M_features = categories['M']
if len(M_features) > 0:
    M_data = df[M_features].fillna(0).values
    df['market_vol'] = np.sqrt(np.sum(M_data ** 2, axis=1) / len(M_features))

P_features = categories['P']
if len(P_features) > 0:
    P_data = df[P_features].fillna(0).values
    df['price_dispersion'] = np.std(P_data, axis=1)

S_features = categories['S']
if len(S_features) >= 2:
    # Sentiment dispersion
    df['sentiment_dispersion'] = np.abs(df.get('S2', 0) - df.get('S5', 0))

E_features = categories['E']
if len(E_features) > 0:
    E_data = df[E_features].fillna(0).values
    df['econ_uncertainty'] = np.std(E_data, axis=1)

volatility_features = ['vol_composite', 'vol_mean', 'vol_max', 'market_vol',
                       'price_dispersion', 'sentiment_dispersion', 'econ_uncertainty']
volatility_features = [f for f in volatility_features if f in df.columns]
print(f"   Created {len(volatility_features)} volatility proxies")

# ============================================================================
# 6. Cross-Category Ratios
# ============================================================================
print("\n[6] Creating cross-category ratios...")

ratio_features = []

# Market-to-Volatility
if 'M4' in df.columns and 'V13' in df.columns:
    df['M4_to_V13'] = df['M4'] / (df['V13'].abs() + eps)
    ratio_features.append('M4_to_V13')

# Price-to-Volatility
if 'P8' in df.columns and 'V7' in df.columns:
    df['P8_to_V7'] = df['P8'] / (df['V7'].abs() + eps)
    ratio_features.append('P8_to_V7')

# Sentiment-to-Economic
if 'S2' in df.columns and 'E19' in df.columns:
    df['S2_to_E19'] = df['S2'] / (df['E19'].abs() + eps)
    ratio_features.append('S2_to_E19')

# Category mean ratios
for cat1, cat2 in [('M', 'V'), ('P', 'V'), ('S', 'E')]:
    col1, col2 = f'{cat1}_mean', f'{cat2}_mean'
    if col1 in df.columns and col2 in df.columns:
        df[f'{cat1}_to_{cat2}_mean'] = df[col1] / (df[col2].abs() + eps)
        ratio_features.append(f'{cat1}_to_{cat2}_mean')

print(f"   Created {len(ratio_features)} ratio features")

# ============================================================================
# 7. Interaction Features (Top MI pairs)
# ============================================================================
print("\n[7] Creating interaction features for top MI features...")

# Top 10 features for interactions
top_10_for_interactions = phase1_results.nlargest(10, 'mi_score')['feature'].tolist()

interaction_features = []

for i, feat1 in enumerate(top_10_for_interactions):
    for feat2 in top_10_for_interactions[i+1:]:
        if feat1 not in df.columns or feat2 not in df.columns:
            continue

        # Multiplication
        df[f'{feat1}*{feat2}'] = df[feat1].fillna(0) * df[feat2].fillna(0)
        interaction_features.append(f'{feat1}*{feat2}')

        # Division
        df[f'{feat1}/{feat2}'] = df[feat1].fillna(0) / (df[feat2].fillna(0).abs() + eps)
        interaction_features.append(f'{feat1}/{feat2}')

print(f"   Created {len(interaction_features)} interaction features")

# ============================================================================
# 8. Summary
# ============================================================================
print("\n[8] Feature summary")
print("="*80)

all_new_features = (
    datetime_features +
    category_stat_features +
    nonlinear_features +
    volatility_features +
    ratio_features +
    interaction_features
)

print(f"\n📊 New Features Created:")
print(f"   DateTime: {len(datetime_features)}")
print(f"   Category stats: {len(category_stat_features)}")
print(f"   Nonlinear transforms: {len(nonlinear_features)}")
print(f"   Volatility proxies: {len(volatility_features)}")
print(f"   Ratios: {len(ratio_features)}")
print(f"   Interactions: {len(interaction_features)}")
print(f"   ---")
print(f"   Total new features: {len(all_new_features)}")

# Original features
original_features = [f for f in phase1_results['feature'].tolist() if f in df.columns]
print(f"   Original features: {len(original_features)}")
print(f"   ---")
print(f"   **Total features: {len(original_features) + len(all_new_features)}**")

# ============================================================================
# 9. Save Engineered Features
# ============================================================================
print("\n[9] Saving engineered features...")

# Combine all features
all_features_combined = original_features + all_new_features

# Save feature names
feature_list = pd.DataFrame({
    'feature': all_features_combined,
    'type': (
        ['original'] * len(original_features) +
        ['datetime'] * len(datetime_features) +
        ['category_stat'] * len(category_stat_features) +
        ['nonlinear'] * len(nonlinear_features) +
        ['volatility'] * len(volatility_features) +
        ['ratio'] * len(ratio_features) +
        ['interaction'] * len(interaction_features)
    )
})

feature_list_path = Path("experiments/024/results/phase2_feature_list.csv")
feature_list.to_csv(feature_list_path, index=False)
print(f"   Feature list saved to: {feature_list_path}")

# Save engineered data (only new features for now)
engineered_data = df[all_new_features].copy()
data_path = Path("experiments/024/results/phase2_engineered_features.parquet")
engineered_data.to_parquet(data_path, index=False)
print(f"   Engineered data saved to: {data_path}")

# Feature type summary
type_counts = feature_list['type'].value_counts()
print(f"\n📈 Feature type distribution:")
for feat_type, count in type_counts.items():
    print(f"   {feat_type}: {count}")

print("\n" + "="*80)
print("Phase 2 Complete! Next: phase3_mi_feature_selection.py")
print("="*80)
