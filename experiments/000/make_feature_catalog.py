#!/usr/bin/env python3
"""
Build a feature catalog for all predictors in train.csv.

Outputs:
- experiments/000/summary/feature_catalog.csv
- experiments/000/FEATURES.md (grouped summary in Markdown)
"""
from __future__ import annotations

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "summary"
SUMMARY.mkdir(parents=True, exist_ok=True)


def load_train(path: str = "data/train.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def feature_groups(cols: List[str]) -> dict:
    return {c: (c[0] if c and c[0].isalpha() else "_") for c in cols}


def main() -> None:
    df = load_train()
    targets = ["forward_returns", "risk_free_rate", "market_forward_excess_returns"]
    exclude = ["date_id"] + targets

    # numeric-only for stats/corr
    num = df.select_dtypes(include=[np.number])
    # choose features among numeric excluding known non-features
    features = [c for c in num.columns if c not in exclude]

    # compute stats
    miss = df[features].isna().mean()
    desc = df[features].describe(percentiles=[0.25, 0.5, 0.75]).T
    # correlations vs targets (only if present & numeric)
    cat_rows = []
    for c in features:
        s = df[c]
        row = {
            "feature": c,
            "group": (c[0] if c and c[0].isalpha() else "_"),
            "missing_rate": float(miss.get(c, np.nan)),
            "mean": float(desc.loc[c, "mean"]),
            "std": float(desc.loc[c, "std"]),
            "min": float(desc.loc[c, "min"]),
            "p25": float(desc.loc[c, "25%"]),
            "p50": float(desc.loc[c, "50%"]),
            "p75": float(desc.loc[c, "75%"]),
            "max": float(desc.loc[c, "max"]),
        }
        for t in ["market_forward_excess_returns", "forward_returns"]:
            if t in df.columns:
                try:
                    row[f"corr_{t}"] = float(df[[c, t]].corr(numeric_only=True).iloc[0, 1])
                except Exception:
                    row[f"corr_{t}"] = np.nan
            else:
                row[f"corr_{t}"] = np.nan
        cat_rows.append(row)

    cat = pd.DataFrame(cat_rows)
    cat = cat.sort_values(["group", "feature"]).reset_index(drop=True)
    out_csv = SUMMARY / "feature_catalog.csv"
    cat.to_csv(out_csv, index=False)

    # Markdown summary grouped by group letter
    md = ["# Feature Catalog (EXP-000)", ""]
    md.append(f"Total features: {len(features)} (excluding date_id + targets)")
    md.append("")
    for g in sorted(cat["group"].unique()):
        sub = cat[cat["group"] == g]
        md.append(f"## Group {g}")
        md.append("- Columns: " + ", ".join(sub["feature"].tolist()))
        md.append("")
        md.append("feature, missing, mean, std, corr_mkt_excess, corr_fwd")
        for _, r in sub.iterrows():
            md.append(
                f"- {r['feature']}, {r['missing_rate']:.3f}, {r['mean']:.3f}, {r['std']:.3f}, "
                f"{(r['corr_market_forward_excess_returns'] if not pd.isna(r['corr_market_forward_excess_returns']) else 0):+.3f}, "
                f"{(r['corr_forward_returns'] if not pd.isna(r['corr_forward_returns']) else 0):+.3f}"
            )
        md.append("")

    (ROOT / "FEATURES.md").write_text("\n".join(md))
    print(f"Wrote {out_csv}")
    print(f"Wrote {ROOT / 'FEATURES.md'}")


if __name__ == "__main__":
    main()

