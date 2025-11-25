#!/usr/bin/env python3
"""
EXP-038 Phase 1: Generate Interaction Features
Goal: Generate candidate interaction features (multiplication, division, polynomial) from Top 20 base features.
These candidates will be evaluated in Phase 2 using Quantile Loss.
"""
import numpy as np
import pandas as pd
import os

print("="*80)
print("EXP-038 Phase 1: Generate Interaction Features")
print("="*80)

# Ensure results directory exists
os.makedirs("experiments/038_quantile_feature_selection/results", exist_ok=True)

# 1. Load Data
print("\n[1] Loading data...")
train = pd.read_csv("data/train.csv")
print(f"   Train shape: {train.shape}")

# 2. Define Base Features (Top 20 from EXP-016)
# These are the proven high-quality base features
top_20 = [
    'M4', 'V13', 'V7', 'P8', 'S2', 'I2', 'E19', 'S5', 'P5', 'P7',
    'M2', 'V9', 'M3', 'P12', 'P10', 'V10', 'E12', 'P11', 'M12', 'S8'
]
print(f"\n[2] Base Features ({len(top_20)}):")
print(f"   {top_20}")

# 3. Generate Interactions
print("\n[3] Generating Interaction Features...")

interactions = []
feature_names = []

# Top 10 for pairwise interactions to keep count reasonable
top_10 = top_20[:10]
print(f"   Using Top 10 for pairwise interactions: {top_10}")

# 3.1 Multiplication
print("   - Generating Multiplication features...")
for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i+1:]:
        feature_names.append(f'{feat1}*{feat2}')

# 3.2 Division
print("   - Generating Division features...")
for i, feat1 in enumerate(top_10):
    for feat2 in top_10[i+1:]:
        feature_names.append(f'{feat1}/{feat2}')

# 3.3 Polynomial
print("   - Generating Polynomial features (Top 5)...")
top_5 = top_20[:5]
for feat in top_5:
    feature_names.append(f'{feat}²')
    feature_names.append(f'{feat}³')

# 4. Save Candidate List
print("\n[4] Saving Candidate Feature List...")
candidate_df = pd.DataFrame({'feature': feature_names, 'type': 'interaction'})
base_df = pd.DataFrame({'feature': top_20, 'type': 'base'})
all_candidates = pd.concat([base_df, candidate_df], ignore_index=True)

output_path = "experiments/038_quantile_feature_selection/results/candidate_features.csv"
all_candidates.to_csv(output_path, index=False)

print(f"   Saved {len(all_candidates)} candidate features to: {output_path}")
print(f"   - Base: {len(base_df)}")
print(f"   - Interaction: {len(candidate_df)}")

print("\n" + "="*80)
print("Phase 1 Complete!")
print("="*80)
