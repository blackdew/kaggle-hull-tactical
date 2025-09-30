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

    # Derive simple notes per feature for human-readable summaries
    med_std = float(cat["std"].median()) if len(cat) else 0.0
    def note_for_row(r: pd.Series) -> str:
        notes = []
        # Missing level
        mr = r.get("missing_rate", 0.0)
        if mr >= 0.50:
            notes.append("결측↑↑")
        elif mr >= 0.30:
            notes.append("결측↑")
        else:
            notes.append("결측양호")
        # Correlation (market excess)
        cm = r.get("corr_market_forward_excess_returns", 0.0)
        if pd.isna(cm):
            pass
        else:
            ac = abs(cm)
            sign = "+" if cm >= 0 else "-"
            if ac >= 0.06:
                notes.append(f"상관 보통({sign})")
            elif ac >= 0.03:
                notes.append(f"상관 약({sign})")
            else:
                notes.append("상관 매우 약")
        # Scale (std)
        sd = r.get("std", 0.0)
        try:
            if sd >= med_std * 1.5:
                notes.append("스케일 큼(변동↑)")
            elif sd <= med_std * 0.5:
                notes.append("스케일 작음")
        except Exception:
            pass
        return ", ".join(notes)

    cat["note"] = cat.apply(note_for_row, axis=1)
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
        md.append("feature, missing, mean, std, corr_mkt_excess, corr_fwd — note")
        for _, r in sub.iterrows():
            md.append(
                f"- {r['feature']}, {r['missing_rate']:.3f}, {r['mean']:.3f}, {r['std']:.3f}, "
                f"{(r['corr_market_forward_excess_returns'] if not pd.isna(r['corr_market_forward_excess_returns']) else 0):+.3f}, "
                f"{(r['corr_forward_returns'] if not pd.isna(r['corr_forward_returns']) else 0):+.3f} — {r['note']}"
            )
        md.append("")

    features_md_path = ROOT / "FEATURES.md"
    features_md_path.write_text("\n".join(md))

    # Build insights markdown (per-feature utilization potential)
    def pot_for_row(r: pd.Series) -> str:
        mr = r.get("missing_rate", 0.0)
        ac = abs(r.get("corr_market_forward_excess_returns", 0.0) or 0.0)
        if mr < 0.30 and ac >= 0.05:
            return "High"
        if mr < 0.50 and ac >= 0.03:
            return "Medium"
        return "Low"

    def actions_for_row(r: pd.Series) -> str:
        parts = []
        mr = r.get("missing_rate", 0.0)
        sd = r.get("std", 0.0)
        ac = abs(r.get("corr_market_forward_excess_returns", 0.0) or 0.0)
        # Missing handling
        if mr >= 0.50:
            parts.append("고결측: 시계열 보간/모델대치, 제거 검토")
        elif mr >= 0.30:
            parts.append("결측↑: 강건 대치/마스킹 피처")
        else:
            parts.append("결측양호")
        # Scale handling
        if sd >= med_std * 1.5:
            parts.append("스케일 큼: winsorize/로그/표준화")
        elif sd <= med_std * 0.5:
            parts.append("스케일 작음: 스케일 통일")
        # Signal handling
        if ac >= 0.05:
            parts.append("신호 후보: 단순/상호작용/롤링")
        elif ac >= 0.03:
            parts.append("약신호: 군집/비선형에서 재검토")
        else:
            parts.append("신호 약: 차원축소/제거 후보")
        return "; ".join(parts)

    cat["potential"] = cat.apply(pot_for_row, axis=1)
    cat["actions"] = cat.apply(actions_for_row, axis=1)

    lines = ["# Feature Insights (EXP-000)", "", "각 피처별 결측/상관/스케일 기반 활용 가능성 요약.", ""]
    for g in sorted(cat["group"].unique()):
        sub = cat[cat["group"] == g]
        lines.append(f"## Group {g}")
        lines.append("feature | potential | note | actions")
        lines.append("--- | --- | --- | ---")
        for _, r in sub.iterrows():
            lines.append(
                f"{r['feature']} | {r['potential']} | {r['note']} | {r['actions']}"
            )
        lines.append("")

    insights_path = ROOT / "FEATURES-INSIGHTS.md"
    insights_path.write_text("\n".join(lines))

    print(f"Wrote {out_csv}")
    print(f"Wrote {features_md_path}")
    print(f"Wrote {insights_path}")


if __name__ == "__main__":
    main()
