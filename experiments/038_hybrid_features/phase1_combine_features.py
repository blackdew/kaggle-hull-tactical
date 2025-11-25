#!/usr/bin/env python3
"""
EXP-038 v3 Phase 1: Combine Features (Hybrid Set)
Goal: Create a Hybrid Feature Set by taking the union of EXP-016 (MSE) and EXP-038 v1 (Quantile) Top 30 features.
"""
import pandas as pd
import os

print("="*80)
print("EXP-038 v3 Phase 1: Combine Features (Hybrid Set)")
print("="*80)

# Ensure results directory exists
os.makedirs("experiments/038_hybrid_features/results", exist_ok=True)

# 1. Load Feature Lists
print("\n[1] Loading Feature Lists...")

# EXP-016 (MSE)
path_016 = "experiments/016/results/top_30_with_interactions.csv"
df_016 = pd.read_csv(path_016)
features_016 = set(df_016['feature'].tolist())
print(f"   EXP-016 (MSE): {len(features_016)} features")

# EXP-038 v1 (Quantile)
path_038_v1 = "experiments/038_quantile_feature_selection/results/top_30_quantile_features.csv"
df_038_v1 = pd.read_csv(path_038_v1)
features_038_v1 = set(df_038_v1['feature'].tolist())
print(f"   EXP-038 v1 (Quantile): {len(features_038_v1)} features")

# 2. Create Hybrid Set (Union)
print("\n[2] Creating Hybrid Set (Union)...")
hybrid_features = sorted(list(features_016.union(features_038_v1)))
print(f"   Hybrid Features: {len(hybrid_features)}")

# Analyze Overlap
overlap = features_016.intersection(features_038_v1)
only_016 = features_016 - features_038_v1
only_038 = features_038_v1 - features_016

print(f"   - Overlap: {len(overlap)}")
print(f"   - Only in MSE (Trend): {len(only_016)}")
print(f"   - Only in Quantile (Tail): {len(only_038)}")

print("\n   New Quantile Features Added:")
print(f"   {sorted(list(only_038))}")

# 3. Save Results
print("\n[3] Saving Results...")
output_path = "experiments/038_hybrid_features/results/hybrid_features.csv"
pd.DataFrame({'feature': hybrid_features}).to_csv(output_path, index=False)
print(f"   Saved to: {output_path}")

print("\n" + "="*80)
print("Phase 1 Complete!")
print("="*80)
