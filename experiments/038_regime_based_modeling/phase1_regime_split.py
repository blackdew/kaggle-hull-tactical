#!/usr/bin/env python3
"""
EXP-038 v2 Phase 1: Regime Split Analysis
Goal: Analyze Volatility (V13) distribution and determine the threshold for splitting High/Low Volatility regimes.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

print("="*80)
print("EXP-038 v2 Phase 1: Regime Split Analysis")
print("="*80)

# Ensure results directory exists
os.makedirs("experiments/038_regime_based_modeling/results", exist_ok=True)

# 1. Load Data
print("\n[1] Loading data...")
train = pd.read_csv("data/train.csv")
print(f"   Train shape: {train.shape}")

# 2. Analyze V13 (Volatility)
print("\n[2] Analyzing V13 (Volatility) Distribution...")
v13 = train['V13'].fillna(0)

# Statistics
stats = v13.describe()
print(f"   V13 Statistics:\n{stats}")

# Determine Threshold (Median)
threshold = v13.median()
print(f"\n   Split Threshold (Median): {threshold:.6f}")

# 3. Split Data
print("\n[3] Splitting Data...")
low_vol_mask = v13 <= threshold
high_vol_mask = v13 > threshold

low_vol_count = low_vol_mask.sum()
high_vol_count = high_vol_mask.sum()

print(f"   Low Volatility Regime (<= {threshold:.6f}): {low_vol_count} samples ({low_vol_count/len(train)*100:.1f}%)")
print(f"   High Volatility Regime (> {threshold:.6f}): {high_vol_count} samples ({high_vol_count/len(train)*100:.1f}%)")

# 4. Save Threshold Info
print("\n[4] Saving Threshold Info...")
threshold_info = pd.DataFrame({
    'threshold_feature': ['V13'],
    'threshold_value': [threshold],
    'low_vol_count': [low_vol_count],
    'high_vol_count': [high_vol_count]
})
output_path = "experiments/038_regime_based_modeling/results/regime_threshold.csv"
threshold_info.to_csv(output_path, index=False)
print(f"   Saved to: {output_path}")

print("\n" + "="*80)
print("Phase 1 Complete!")
print("="*80)
