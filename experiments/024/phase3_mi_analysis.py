"""
EXP-024 Phase 3: MI Analysis (All 305 Features)

모든 features의 MI 계산 후, 305개 모두 사용
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EXP-024 Phase 3: MI Analysis (All Features)")
print("="*80)

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n[1] Loading data...")
df = pd.read_csv("data/train.csv")
y = df['market_forward_excess_returns'].values

# Load feature list
feature_list = pd.read_csv("experiments/024/results/phase2_feature_list.csv")
all_features = feature_list['feature'].tolist()

print(f"   Total features: {len(all_features)}")
print(f"   Samples: {len(df)}")

# ============================================================================
# 2. Prepare Feature Matrix
# ============================================================================
print("\n[2] Preparing feature matrix...")

# Load engineered features
engineered_data = pd.read_parquet("experiments/024/results/phase2_engineered_features.parquet")

# Combine with original data
# Get original features
original_features = feature_list[feature_list['type'] == 'original']['feature'].tolist()
original_data = df[original_features].copy()

# Combine
X_all = pd.concat([original_data, engineered_data], axis=1)

# Reorder to match feature_list
X_all = X_all[all_features]

# Fill missing values
X_all = X_all.fillna(0)

print(f"   Feature matrix shape: {X_all.shape}")

# ============================================================================
# 3. Calculate Mutual Information
# ============================================================================
print("\n[3] Calculating Mutual Information for all features...")
print("   This may take 5-10 minutes...")

# Remove rows with NaN in target
valid_idx = ~np.isnan(y)
X_valid = X_all.loc[valid_idx].values
y_valid = y[valid_idx]

print(f"   Valid samples: {len(y_valid)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_valid)

# Calculate MI (this will take time with 305 features)
mi_scores = mutual_info_regression(X_scaled, y_valid, random_state=42, n_neighbors=5)

# ============================================================================
# 4. Analyze Results
# ============================================================================
print("\n[4] Analyzing MI scores...")

mi_results = pd.DataFrame({
    'feature': all_features,
    'mi_score': mi_scores,
    'type': feature_list['type'].values,
})
mi_results = mi_results.sort_values('mi_score', ascending=False)

# Top features overall
print(f"\n📊 Top 20 Features by MI:")
for i, row in mi_results.head(20).iterrows():
    print(f"   {i+1}. {row['feature']}: {row['mi_score']:.6f} ({row['type']})")

# By feature type
print(f"\n📈 MI Statistics by Feature Type:")
type_stats = mi_results.groupby('type')['mi_score'].agg(['count', 'mean', 'max', 'std'])
type_stats = type_stats.sort_values('mean', ascending=False)
for feat_type, row in type_stats.iterrows():
    print(f"   {feat_type}: count={int(row['count'])}, mean={row['mean']:.6f}, max={row['max']:.6f}")

# ============================================================================
# 5. Feature Type Analysis
# ============================================================================
print(f"\n[5] Comparing feature types...")

# Count top features by type
top_20_types = mi_results.head(20)['type'].value_counts()
top_50_types = mi_results.head(50)['type'].value_counts()
top_100_types = mi_results.head(100)['type'].value_counts()

print(f"\n   Top 20 features by type:")
for feat_type, count in top_20_types.items():
    print(f"      {feat_type}: {count}")

print(f"\n   Top 50 features by type:")
for feat_type, count in top_50_types.items():
    print(f"      {feat_type}: {count}")

print(f"\n   Top 100 features by type:")
for feat_type, count in top_100_types.items():
    print(f"      {feat_type}: {count}")

# ============================================================================
# 6. Redundancy Check (Top 100)
# ============================================================================
print(f"\n[6] Checking redundancy in top 100 features...")

top_100_features = mi_results.head(100)['feature'].tolist()
X_top100 = X_all[top_100_features].values

# Calculate pairwise correlation
print("   Calculating correlation matrix...")
corr_matrix = np.corrcoef(X_top100.T)

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(top_100_features)):
    for j in range(i+1, len(top_100_features)):
        corr_val = abs(corr_matrix[i, j])
        if corr_val > 0.95:
            high_corr_pairs.append({
                'feature1': top_100_features[i],
                'feature2': top_100_features[j],
                'correlation': corr_val,
            })

print(f"   Highly correlated pairs (>0.95): {len(high_corr_pairs)}")
if len(high_corr_pairs) > 0 and len(high_corr_pairs) <= 10:
    for pair in high_corr_pairs:
        print(f"      {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.4f}")
elif len(high_corr_pairs) > 10:
    print(f"      (Too many to display, showing top 5)")
    sorted_pairs = sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True)
    for pair in sorted_pairs[:5]:
        print(f"      {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.4f}")

# ============================================================================
# 7. Save Results
# ============================================================================
print(f"\n[7] Saving results...")

output_path = Path("experiments/024/results/phase3_mi_analysis_all_features.csv")
mi_results.to_csv(output_path, index=False)
print(f"   Saved to: {output_path}")

# Save selected features (all 305)
selected_features_path = Path("experiments/024/results/phase3_selected_features_all.csv")
pd.DataFrame({'feature': all_features}).to_csv(selected_features_path, index=False)
print(f"   Selected features (all 305) saved to: {selected_features_path}")

# ============================================================================
# 8. Recommendations
# ============================================================================
print(f"\n[8] Analysis Summary")
print("="*80)

best_type = type_stats.index[0]
worst_type = type_stats.index[-1]

print(f"\n✅ Best Feature Type: {best_type}")
print(f"   Mean MI: {type_stats.loc[best_type, 'mean']:.6f}")
print(f"   Max MI: {type_stats.loc[best_type, 'max']:.6f}")

print(f"\n⚠️  Worst Feature Type: {worst_type}")
print(f"   Mean MI: {type_stats.loc[worst_type, 'mean']:.6f}")

print(f"\n📊 Using ALL 305 features for Phase 4")

# Overfitting warning
print(f"\n⚠️  OVERFITTING RISK:")
print(f"   Using 305 features with 8990 samples")
print(f"   Ratio: {8990 / 305:.1f} samples per feature")
print(f"   Recommended: 10+ samples per feature")
print(f"   → Overfitting risk is MODERATE")
print(f"   → Model regularization (XGBoost) should help")

# Alternative suggestions
if len(high_corr_pairs) > 20:
    print(f"\n💡 Alternative: Remove {len(high_corr_pairs)} redundant features")
    print(f"   This would reduce to ~{305 - len(high_corr_pairs)} features")

print("\n" + "="*80)
print("Phase 3 Complete! Next: phase4_model_training.py")
print("="*80)
