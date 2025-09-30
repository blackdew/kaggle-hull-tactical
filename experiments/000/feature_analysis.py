#!/usr/bin/env python3
"""
Feature analysis for Hull Tactical Market Prediction.

Outputs:
- summary/feature_missing.csv: missing rate per feature (desc)
- summary/feature_target_corr.csv: Pearson corr with targets
- summary/group_stats.csv: group-level (D/E/I/M/P/S/V) counts & missing
- plots/missing_top30.png: top-30 highest missing features
- plots/corr_market_excess_top30.png: top-30 |corr| vs market_forward_excess_returns
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "summary"
PLOTS = ROOT / "plots"
SUMMARY.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


def load_train(path: str = "data/train.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def feature_groups(cols: list[str]) -> dict[str, str]:
    groups = {}
    for c in cols:
        g = c[0] if c and c[0].isalpha() else "_"
        groups[c] = g
    return groups


def compute_missing(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False)
    out = miss.rename("missing_rate").reset_index().rename(columns={"index": "feature"})
    return out


def compute_target_corr(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    feats = [c for c in numeric.columns if c not in targets]
    rows = []
    for t in targets:
        if t not in numeric.columns:
            continue
        corr = numeric[feats + [t]].corr(numeric_only=True)[t].drop(index=t)
        rows.append(
            pd.DataFrame({
                "feature": corr.index,
                "target": t,
                "pearson_corr": corr.values,
                "abs_corr": np.abs(corr.values),
            })
        )
    out = pd.concat(rows, ignore_index=True).sort_values(["target", "abs_corr"], ascending=[True, False])
    return out


def compute_group_stats(missing: pd.DataFrame) -> pd.DataFrame:
    groups = feature_groups(missing["feature"].tolist())
    df = missing.assign(group=missing["feature"].map(groups))
    df = df.groupby("group").agg(
        features=("feature", "count"),
        missing_mean=("missing_rate", "mean"),
        missing_median=("missing_rate", "median"),
    ).reset_index().sort_values("group")
    return df


def plot_missing_top(missing: pd.DataFrame, top: int = 30) -> None:
    top_df = missing.head(top)
    plt.figure(figsize=(10, 10))
    sns.barplot(data=top_df, x="missing_rate", y="feature", orient="h")
    plt.title(f"Top {top} Missing Features")
    plt.xlabel("Missing rate")
    plt.tight_layout()
    plt.savefig(PLOTS / "missing_top30.png", dpi=150)
    plt.close()


def plot_corr_top(corr: pd.DataFrame, target: str, top: int = 30) -> None:
    sub = corr[corr["target"] == target].nlargest(top, "abs_corr")[["feature", "pearson_corr"]]
    plt.figure(figsize=(10, 10))
    sns.barplot(data=sub, x="pearson_corr", y="feature", orient="h",
                palette=["#377eb8" if v >= 0 else "#e41a1c" for v in sub["pearson_corr"]])
    plt.title(f"Top {top} Pearson corr vs {target}")
    plt.xlabel("Pearson corr")
    plt.tight_layout()
    fname = f"corr_{target.replace(' ', '_')}_top{top}.png"
    plt.savefig(PLOTS / fname, dpi=150)
    plt.close()


def main() -> None:
    df = load_train()
    targets = ["forward_returns", "market_forward_excess_returns"]

    # Missing
    missing = compute_missing(df)
    missing.to_csv(SUMMARY / "feature_missing.csv", index=False)

    # Corr
    corr = compute_target_corr(df, targets)
    corr.to_csv(SUMMARY / "feature_target_corr.csv", index=False)

    # Group stats
    group_stats = compute_group_stats(missing)
    group_stats.to_csv(SUMMARY / "group_stats.csv", index=False)

    # Plots
    plot_missing_top(missing)
    plot_corr_top(corr, target="market_forward_excess_returns")

    print("Saved summaries to:")
    print(f"- {SUMMARY / 'feature_missing.csv'}")
    print(f"- {SUMMARY / 'feature_target_corr.csv'}")
    print(f"- {SUMMARY / 'group_stats.csv'}")
    print("Saved plots to:")
    print(f"- {PLOTS / 'missing_top30.png'}")
    print(f"- {PLOTS / 'corr_market_forward_excess_returns_top30.png'}")


if __name__ == "__main__":
    main()

