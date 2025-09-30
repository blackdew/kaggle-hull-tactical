#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler

EXCLUDE_COLS = {
    "date_id",
    "forward_returns",
    "risk_free_rate",
    "market_forward_excess_returns",
    "is_scored",
    "lagged_forward_returns",
    "lagged_risk_free_rate",
    "lagged_market_forward_excess_returns",
}


def select_base_features(df: pd.DataFrame, high_missing_threshold: float = 0.5) -> List[str]:
    feats = [c for c in df.columns if c not in EXCLUDE_COLS]
    miss = df[feats].isna().mean()
    keep = [c for c in feats if miss.get(c, 0.0) < high_missing_threshold]
    return keep


def top_abs_corr(df: pd.DataFrame, features: List[str], target: str, n: int) -> List[str]:
    corr = df[features + [target]].corr(numeric_only=True)[target].drop(labels=[target], errors="ignore").abs()
    ordered = [f for f, _ in sorted(corr.items(), key=lambda x: -x[1])[: int(n)]]
    return ordered


def preprocess_fit(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    Xp = X.fillna(X.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
    s = StandardScaler()
    Xs = s.fit_transform(Xp)
    return Xs, s


def preprocess_apply(X: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    Xp = X.fillna(X.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
    return scaler.transform(Xp)


def to_positions(pred_excess: np.ndarray, k: float = 50.0) -> np.ndarray:
    pos = 1.0 + pred_excess * (k / 1.0)
    return np.clip(pos, 0.0, 2.0)

