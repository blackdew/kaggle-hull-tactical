#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/submissions"
LOG="$ROOT/logs"

echo "[CI] Generating served submissions for A/B..."
python "$ROOT/serve.py" --candidate A >/dev/null
python "$ROOT/serve.py" --candidate B >/dev/null

fail=0
for f in "$OUT"/candidate_*_served.csv; do
  echo "[CI] Checking $f"
  python - "$f" << 'PY'
import sys, pandas as pd, numpy as np
f=sys.argv[1]
df=pd.read_csv(f)
assert 'prediction' in df.columns, 'missing prediction column'
assert len(df)>=1, 'empty predictions'
mn=float(df['prediction'].min()); mx=float(df['prediction'].max())
assert mn>=0.0-1e-9 and mx<=2.0+1e-9, f'predictions out of range: [{mn},{mx}]'
print('OK', f, len(df), mn, mx)
PY
done

echo "[CI] Checking logs..."
for j in "$LOG"/serve_*.json; do
  python - "$j" << 'PY'
import sys, json
with open(sys.argv[1]) as fh:
  js=json.load(fh)
for k in ['candidate','output_file','train_stats','timestamp']:
  assert k in js, f'missing key {k}'
print('OK', sys.argv[1])
PY
done

echo "[CI] All checks passed."

