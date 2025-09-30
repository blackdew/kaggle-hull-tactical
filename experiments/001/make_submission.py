#!/usr/bin/env python3
# snapshot: experiments/001
"""
Generate a first submission file for Hull Tactical - Market Prediction.

Trains SimplePositionSizingModel on data/train.csv and predicts for data/test.csv,
outputting submissions/submission.csv with a single 'prediction' column (values in [0, 2]).

Usage:
  python scripts/make_submission.py
  python scripts/make_submission.py --out submissions/my_submission.csv
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class SimplePositionSizingModel:
    """독립형 포지션 사이징 모델(파일 내 정의, kaggle_evaluation 미사용)"""

    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []

    @staticmethod
    def prepare_features(df: pd.DataFrame) -> list[str]:
        exclude_cols = [
            "date_id",
            "forward_returns",
            "risk_free_rate",
            "market_forward_excess_returns",
            "is_scored",
            "lagged_forward_returns",
            "lagged_risk_free_rate",
            "lagged_market_forward_excess_returns",
        ]
        potential = [c for c in df.columns if c not in exclude_cols]
        selected: list[str] = []
        for c in potential:
            missing = float(df[c].isna().sum()) / float(len(df))
            if missing < 0.5:
                selected.append(c)
        return selected

    @staticmethod
    def preprocess(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        X = df[cols].copy()
        X = X.fillna(X.median(numeric_only=True))
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        return X

    @staticmethod
    def to_position(excess_return_pred: float, risk_adjustment: float = 0.5) -> float:
        base = 1.0
        pos = base + excess_return_pred * risk_adjustment * 100
        return float(np.clip(pos, 0.0, 2.0))

    def fit(self, train_df: pd.DataFrame) -> None:
        self.feature_cols = self.prepare_features(train_df)
        X = self.preprocess(train_df, self.feature_cols)
        y = train_df["market_forward_excess_returns"].to_numpy()
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)

    def predict_positions(self, test_df: pd.DataFrame) -> np.ndarray:
        X = self.preprocess(test_df, self.feature_cols)
        Xs = self.scaler.transform(X)
        yhat = self.model.predict(Xs)
        return np.clip(1.0 + yhat * 0.5 * 100, 0.0, 2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--test", default="data/test.csv", help="Path to test.csv")
    parser.add_argument(
        "--out", default="submissions/submission.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading train/test data...")
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    # Train model
    model = SimplePositionSizingModel()
    model.fit(train_df)

    # Prepare features and predict vectorized
    positions = model.predict_positions(test_df)

    # Build submission: single column 'prediction'
    sub = pd.DataFrame({"prediction": positions})

    # Clip just in case
    sub["prediction"] = sub["prediction"].clip(0.0, 2.0)

    sub.to_csv(args.out, index=False)
    print(f"Wrote submission: {args.out}  (rows={len(sub)})")


if __name__ == "__main__":
    main()
