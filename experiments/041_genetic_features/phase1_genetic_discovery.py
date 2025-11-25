#!/usr/bin/env python3
"""
EXP-041 Phase 1: Genetic Feature Discovery
Goal: Use Symbolic Regression (gplearn) to automatically discover high-signal features.
"""
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer
from sklearn.preprocessing import StandardScaler
import os
import joblib

print("="*80)
print("EXP-041 Phase 1: Genetic Feature Discovery")
print("="*80)

# 1. Load Data
print("\n[1] Loading data...")
train = pd.read_csv("data/train.csv")
y = train['market_forward_excess_returns'].values

# 2. Prepare Base Features
print("\n[2] Preparing Base Features...")
# We use the Top 20 features as the "atoms" for evolution
top_20 = [
    'M4', 'V13', 'V7', 'P8', 'S2', 'I2', 'E19', 'S5', 'P5', 'P7',
    'M2', 'V9', 'M3', 'P12', 'P10', 'V10', 'E12', 'P11', 'M12', 'S8'
]
X = train[top_20].fillna(train[top_20].median()).replace([np.inf, -np.inf], np.nan).fillna(0)
feature_names = top_20

print(f"   Base Features: {len(feature_names)}")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Symbolic Regression (Genetic Programming)
print("\n[3] Running Symbolic Regression (This may take a while)...")

# Function set: Arithmetic + Log, Sqrt, Abs, Max, Min
function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min']

est_gp = SymbolicTransformer(
    generations=20,          # Number of generations
    population_size=2000,    # Population size
    hall_of_fame=100,        # Number of top individuals to keep
    n_components=20,         # Number of features to generate
    function_set=function_set,
    parsimony_coefficient=0.0005, # Penalty for complexity
    max_samples=0.9,         # Subsample data
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fix for gplearn compatibility with newer sklearn
# est_gp.fit(X_scaled, y)
# Manually validate data if needed, or just pass directly as gplearn should handle it.
# The error suggests gplearn is trying to call a deprecated sklearn method.
# We will try to monkeypatch or just use a workaround if possible.
# Actually, let's try to downgrade sklearn or fix the call.
# But we can't easily downgrade. Let's try to bypass the validation or use a different way.

# Workaround: Monkeypatch BaseEstimator._validate_data if it's missing
from sklearn.base import BaseEstimator
if not hasattr(BaseEstimator, "_validate_data"):
    def _validate_data(self, X, y=None, **kwargs):
        from sklearn.utils.validation import check_X_y, check_array
        if y is None:
            return check_array(X, **kwargs)
        else:
            return check_X_y(X, y, **kwargs)
    BaseEstimator._validate_data = _validate_data

est_gp.fit(X_scaled, y)

# Fix for gplearn transform compatibility
if not hasattr(est_gp, 'n_features_in_'):
    est_gp.n_features_in_ = X_scaled.shape[1]

# 4. Extract Discovered Features
print("\n[4] Extracting Discovered Features...")
best_programs = est_gp._best_programs
print(f"   Found {len(best_programs)} programs.")

discovered_features = []
for i, program in enumerate(best_programs):
    # Get the string representation of the formula
    formula = str(program)
    print(f"   Feature {i+1}: {formula}")
    discovered_features.append(formula)

# 5. Transform Data
print("\n[5] Transforming Data...")
X_new = est_gp.transform(X_scaled)
X_new_df = pd.DataFrame(X_new, columns=[f'GP_{i}' for i in range(X_new.shape[1])])

# Calculate correlation with target
correlations = X_new_df.corrwith(pd.Series(y))
print("\n   Correlations with Target:")
print(correlations.sort_values(ascending=False))

# 6. Save Results
print("\n[6] Saving Results...")
output_dir = "experiments/041_genetic_features/results"
os.makedirs(output_dir, exist_ok=True)

# Save the transformer model
joblib.dump(est_gp, f"{output_dir}/genetic_transformer.pkl")

# Save feature definitions
pd.DataFrame({'feature': [f'GP_{i}' for i in range(len(discovered_features))], 'formula': discovered_features}).to_csv(f"{output_dir}/genetic_formulas.csv", index=False)

# Save transformed features (optional, for quick loading in Phase 2)
# X_new_df.to_csv(f"{output_dir}/genetic_features_train.csv", index=False)

print(f"   Model saved to: {output_dir}/genetic_transformer.pkl")
print(f"   Formulas saved to: {output_dir}/genetic_formulas.csv")

print("\n" + "="*80)
print("Phase 1 Complete!")
print("="*80)
