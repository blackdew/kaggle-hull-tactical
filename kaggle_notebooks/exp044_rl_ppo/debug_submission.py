
import os
import sys

print("="*80)
print("DEBUG SCRIPT STARTED")
print("="*80)

# 1. Check Files
print("\n[1] Checking File System:")
try:
    print(f"CWD: {os.getcwd()}")
    if os.path.exists("/kaggle/input"):
        print("/kaggle/input exists. Listing contents:")
        for root, dirs, files in os.walk("/kaggle/input"):
            print(f"  {root}")
            for f in files:
                print(f"    {f}")
    else:
        print("/kaggle/input does NOT exist.")
except Exception as e:
    print(f"Error checking files: {e}")

# 2. Check Env Vars
print("\n[2] Checking Environment Variables:")
for k, v in os.environ.items():
    if "KAGGLE" in k:
        print(f"  {k}: {v}")

# 3. Check Imports
print("\n[3] Checking Imports:")
modules = [
    'pandas', 'numpy', 'torch', 'xgboost', 'lightgbm', 'catboost',
    'kaggle_evaluation', 'kaggle_evaluation.core.templates'
]

for mod in modules:
    try:
        __import__(mod)
        print(f"  [OK] {mod}")
    except ImportError as e:
        print(f"  [FAIL] {mod}: {e}")
    except Exception as e:
        print(f"  [ERROR] {mod}: {e}")

print("\nDEBUG SCRIPT FINISHED")
