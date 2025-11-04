"""
EXP-024 Phase 1: Data Analysis

Mutual Information, Nonlinear Correlation, Feature Group 분석
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EXP-024 Phase 1: Data Analysis")
print("="*80)

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n[1] Loading data...")
df = pd.read_csv("data/train.csv")
print(f"   Data shape: {df.shape}")

# Get all base features (exclude date_id, targets)
exclude_cols = ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']
all_features = [col for col in df.columns if col not in exclude_cols]
print(f"   Total features: {len(all_features)}")

# Target
y = df['market_forward_excess_returns'].values

# ============================================================================
# 2. Categorize Features
# ============================================================================
print("\n[2] Categorizing features...")

categories = {
    'M': [f for f in all_features if f.startswith('M')],
    'V': [f for f in all_features if f.startswith('V')],
    'P': [f for f in all_features if f.startswith('P')],
    'S': [f for f in all_features if f.startswith('S')],
    'I': [f for f in all_features if f.startswith('I')],
    'E': [f for f in all_features if f.startswith('E')],
    'D': [f for f in all_features if f.startswith('D')],
}

for cat, feats in categories.items():
    print(f"   {cat}: {len(feats)} features")

# ============================================================================
# 3. Missing Value Analysis
# ============================================================================
print("\n[3] Missing value analysis...")

missing_stats = []
for feat in all_features:
    missing_pct = df[feat].isna().mean() * 100
    missing_stats.append({
        'feature': feat,
        'category': feat[0] if feat[0] in categories else 'Other',
        'missing_pct': missing_pct,
    })

df_missing = pd.DataFrame(missing_stats).sort_values('missing_pct', ascending=False)
print(f"   Features with >50% missing: {(df_missing['missing_pct'] > 50).sum()}")
print(f"   Features with >80% missing: {(df_missing['missing_pct'] > 80).sum()}")

# ============================================================================
# 4. Correlation Analysis
# ============================================================================
print("\n[4] Correlation analysis...")

# Fill missing values for correlation calculation
df_filled = df[all_features].fillna(df[all_features].median())

# Pearson correlation with target
pearson_corr = []
for feat in all_features:
    if df_filled[feat].std() > 0:
        corr = np.corrcoef(df_filled[feat].values, y)[0, 1]
    else:
        corr = 0.0
    pearson_corr.append({
        'feature': feat,
        'pearson_corr': abs(corr),
    })

df_pearson = pd.DataFrame(pearson_corr).sort_values('pearson_corr', ascending=False)

print(f"   Top 10 by Pearson correlation:")
for i, row in df_pearson.head(10).iterrows():
    print(f"      {row['feature']}: {row['pearson_corr']:.4f}")

# ============================================================================
# 5. Spearman Correlation (Rank-based, captures monotonic relationships)
# ============================================================================
print("\n[5] Spearman correlation analysis...")

spearman_corr = []
for feat in all_features:
    if df_filled[feat].std() > 0:
        corr, _ = spearmanr(df_filled[feat].values, y, nan_policy='omit')
    else:
        corr = 0.0
    spearman_corr.append({
        'feature': feat,
        'spearman_corr': abs(corr),
    })

df_spearman = pd.DataFrame(spearman_corr).sort_values('spearman_corr', ascending=False)

print(f"   Top 10 by Spearman correlation:")
for i, row in df_spearman.head(10).iterrows():
    print(f"      {row['feature']}: {row['spearman_corr']:.4f}")

# ============================================================================
# 6. Mutual Information (captures nonlinear relationships)
# ============================================================================
print("\n[6] Mutual Information analysis...")
print("   This may take a few minutes...")

# Prepare data (remove rows with NaN in target)
valid_idx = ~np.isnan(y)
X_valid = df_filled.loc[valid_idx].values
y_valid = y[valid_idx]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_valid)

# Calculate MI
mi_scores = mutual_info_regression(X_scaled, y_valid, random_state=42, n_neighbors=5)

mi_results = []
for feat, score in zip(all_features, mi_scores):
    mi_results.append({
        'feature': feat,
        'mi_score': score,
    })

df_mi = pd.DataFrame(mi_results).sort_values('mi_score', ascending=False)

print(f"   Top 10 by Mutual Information:")
for i, row in df_mi.head(10).iterrows():
    print(f"      {row['feature']}: {row['mi_score']:.6f}")

# ============================================================================
# 7. Compare Correlation vs MI
# ============================================================================
print("\n[7] Comparing Pearson, Spearman, and MI...")

# Merge all metrics
df_comparison = df_pearson.merge(df_spearman, on='feature').merge(df_mi, on='feature')

# Features with high MI but low Pearson (nonlinear relationships)
df_comparison['mi_minus_pearson'] = df_comparison['mi_score'] * 100 - df_comparison['pearson_corr']
df_nonlinear = df_comparison.sort_values('mi_minus_pearson', ascending=False)

print(f"\n   Top 10 features with high MI but low Pearson (nonlinear):")
for i, row in df_nonlinear.head(10).iterrows():
    print(f"      {row['feature']}: MI={row['mi_score']:.6f}, Pearson={row['pearson_corr']:.4f}")

# ============================================================================
# 8. Category-wise Statistics
# ============================================================================
print("\n[8] Category-wise statistics...")

category_stats = []
for cat, feats in categories.items():
    if len(feats) == 0:
        continue

    cat_df = df[feats]

    # Overall stats
    missing_mean = cat_df.isna().mean().mean() * 100

    # Correlation stats
    cat_pearson = df_pearson[df_pearson['feature'].isin(feats)]['pearson_corr']
    cat_spearman = df_spearman[df_spearman['feature'].isin(feats)]['spearman_corr']
    cat_mi = df_mi[df_mi['feature'].isin(feats)]['mi_score']

    category_stats.append({
        'category': cat,
        'n_features': len(feats),
        'missing_mean': missing_mean,
        'pearson_mean': cat_pearson.mean(),
        'pearson_max': cat_pearson.max(),
        'spearman_mean': cat_spearman.mean(),
        'spearman_max': cat_spearman.max(),
        'mi_mean': cat_mi.mean(),
        'mi_max': cat_mi.max(),
    })

df_cat_stats = pd.DataFrame(category_stats).sort_values('mi_max', ascending=False)

print(f"\n   Category ranking by MI:")
for i, row in df_cat_stats.iterrows():
    print(f"      {row['category']}: {row['n_features']} features, "
          f"MI_max={row['mi_max']:.6f}, MI_mean={row['mi_mean']:.6f}")

# ============================================================================
# 9. Feature-Feature Correlation (Redundancy)
# ============================================================================
print("\n[9] Feature-feature correlation (redundancy)...")

# Get top 30 features by MI
top_30_features = df_mi.head(30)['feature'].tolist()
X_top30 = df_filled[top_30_features].values

# Calculate pairwise correlation
corr_matrix = np.corrcoef(X_top30.T)

# Find highly correlated pairs (>0.9)
high_corr_pairs = []
for i in range(len(top_30_features)):
    for j in range(i+1, len(top_30_features)):
        corr_val = abs(corr_matrix[i, j])
        if corr_val > 0.9:
            high_corr_pairs.append({
                'feature1': top_30_features[i],
                'feature2': top_30_features[j],
                'correlation': corr_val,
            })

print(f"   Highly correlated pairs (>0.9) in top 30: {len(high_corr_pairs)}")
if len(high_corr_pairs) > 0:
    print(f"   Top 5 redundant pairs:")
    for pair in sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True)[:5]:
        print(f"      {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.4f}")

# ============================================================================
# 10. Save Results
# ============================================================================
print("\n[10] Saving results...")

# Combine all metrics
df_final = df_comparison.copy()
df_final = df_final.merge(df_missing[['feature', 'missing_pct']], on='feature')
df_final['category'] = df_final['feature'].str[0]
df_final = df_final.sort_values('mi_score', ascending=False)

output_path = Path("experiments/024/results/phase1_feature_analysis.csv")
df_final.to_csv(output_path, index=False)
print(f"   Saved to: {output_path}")

# Save category stats
cat_output_path = Path("experiments/024/results/phase1_category_stats.csv")
df_cat_stats.to_csv(cat_output_path, index=False)
print(f"   Saved to: {cat_output_path}")

# Save redundant pairs
if len(high_corr_pairs) > 0:
    df_redundant = pd.DataFrame(high_corr_pairs)
    redundant_path = Path("experiments/024/results/phase1_redundant_pairs.csv")
    df_redundant.to_csv(redundant_path, index=False)
    print(f"   Saved to: {redundant_path}")

# ============================================================================
# 11. Key Insights
# ============================================================================
print("\n[11] Key Insights")
print("="*80)

# Top features by each metric
top_5_pearson = df_final.head(5)['feature'].tolist()
top_5_spearman = df_spearman.head(5)['feature'].tolist()
top_5_mi = df_final.head(5)['feature'].tolist()

print(f"\n📊 Top 5 Features:")
print(f"   Pearson: {', '.join(top_5_pearson)}")
print(f"   Spearman: {', '.join(top_5_spearman)}")
print(f"   MI: {', '.join(top_5_mi)}")

# Nonlinear features
print(f"\n🔄 Nonlinear Relationships:")
print(f"   {len(df_nonlinear[df_nonlinear['mi_minus_pearson'] > 0])} features have higher MI than Pearson")
print(f"   → These features have nonlinear relationships with target")

# Category insights
best_category = df_cat_stats.iloc[0]
print(f"\n🏆 Best Category: {best_category['category']}")
print(f"   MI_max: {best_category['mi_max']:.6f}")
print(f"   MI_mean: {best_category['mi_mean']:.6f}")
print(f"   Features: {int(best_category['n_features'])}")

# Redundancy
print(f"\n⚠️  Redundancy:")
print(f"   {len(high_corr_pairs)} highly correlated pairs (>0.9) in top 30")
if len(high_corr_pairs) > 0:
    print(f"   → Feature selection needed to remove redundancy")

# Recommendations for Phase 2
print(f"\n✅ Recommendations for Phase 2:")

# 1. Nonlinear transformations
nonlinear_features = df_nonlinear.head(10)['feature'].tolist()
print(f"\n   1. Apply nonlinear transforms to: {', '.join(nonlinear_features[:5])}, ...")

# 2. Category combinations
top_categories = df_cat_stats.head(3)['category'].tolist()
print(f"   2. Focus on category combinations: {', '.join(top_categories)}")

# 3. High MI features
high_mi_features = df_final[df_final['mi_score'] > df_final['mi_score'].quantile(0.75)]['feature'].tolist()
print(f"   3. Use {len(high_mi_features)} features with MI > 75th percentile")

# 4. Remove redundancy
if len(high_corr_pairs) > 0:
    print(f"   4. Remove {len(high_corr_pairs)} redundant feature pairs")

print("\n" + "="*80)
print("Phase 1 Complete! Next: phase2_feature_engineering.py")
print("="*80)
