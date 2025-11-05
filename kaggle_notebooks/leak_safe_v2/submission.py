#!/usr/bin/env python3
"""
EXP-036: Leak-Safe Baseline Strategy (17.396 Target)

핵심 전략:
1. Train 데이터 전체로 최적 target 역산
2. ElasticNet으로 features → target 매핑
3. 수학적으로 최적 Sharpe 달성

Reference: https://www.kaggle.com/code/morodertobias/hull-leak-safe-baseline
"""
from kaggle_evaluation.core.templates import InferenceServer
import os
import numpy as np
import pandas as pd

try:
    from default_gateway import DefaultGateway
except Exception:
    from kaggle_evaluation.default_gateway import DefaultGateway

try:
    import polars as pl
except Exception:
    pl = None


class MyServer(InferenceServer):
    """EXP-036: Leak-Safe Baseline with Optimal Target"""

    def __init__(self):
        from sklearn.linear_model import ElasticNet, ElasticNetCV
        from sklearn.preprocessing import StandardScaler

        self.ElasticNet = ElasticNet
        self.ElasticNetCV = ElasticNetCV
        self.StandardScaler = StandardScaler
        self.ready = False
        self.model = None
        self.scaler = None
        self.best_alpha = None

        # Features from leak-safe baseline
        self.features = [
            "S2", "E2", "E3", "P8", "P9", "P10", "P12", "P13",
            "S1", "S5", "I2", "U1", "U2"
        ]

        # ElasticNet params (same as leak-safe baseline)
        self.l1_ratio = 0.5
        self.cv = 10
        self.alphas = np.logspace(-4, 2, 100)
        self.max_iter = 1000000

        def predict(batch):
            return MyServer.predict(self, batch)
        super().__init__(predict)

    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None, *args, **kwargs):
        return DefaultGateway(data_paths)

    def train_if_needed(self):
        """Lazy training on first prediction"""
        if self.ready:
            return

        # Load training data
        try:
            train = pd.read_csv("/kaggle/input/hull-tactical-market-prediction/train.csv")
        except:
            train = pd.read_csv("data/train.csv")

        # Create derived features
        train["U1"] = train["I2"] - train["I1"]
        train["U2"] = train["M11"] / ((train["I2"] + train["I9"] + train["I7"]) / 3)

        # Drop NaN
        train = train.dropna()

        # Calculate optimal target (핵심!)
        forward_returns = train['forward_returns'].values
        risk_free_rate = train['risk_free_rate'].values
        market_excess_returns = forward_returns - risk_free_rate

        # Optimal position calculation
        market_excess_cumulative = (1 + market_excess_returns).prod()
        market_mean_excess_return = (market_excess_cumulative) ** (1 / len(train)) - 1
        positive_rate = (market_excess_returns > 0).mean()
        c = (1 + market_mean_excess_return) ** (1 / positive_rate) - 1

        # Target = c / market_excess_returns, clipped to [0, 2]
        target = (c / market_excess_returns).clip(0, 2)

        print(f"[INFO] Optimal c parameter: {c:.6f}")
        print(f"[INFO] Target stats - Mean: {target.mean():.4f}, Std: {target.std():.4f}")

        # Prepare features
        X_train = train[self.features].values
        y_train = target

        # Scale features
        self.scaler = self.StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # Train ElasticNetCV to find best alpha
        print("[INFO] Training ElasticNetCV to find optimal alpha...")
        model_cv = self.ElasticNetCV(
            l1_ratio=self.l1_ratio,
            cv=self.cv,
            alphas=self.alphas,
            max_iter=self.max_iter,
            n_jobs=-1
        )
        model_cv.fit(X_scaled, y_train)
        self.best_alpha = model_cv.alpha_
        print(f"[INFO] Best alpha found: {self.best_alpha:.6f}")

        # Train final model with best alpha
        self.model = self.ElasticNet(
            alpha=self.best_alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter
        )
        self.model.fit(X_scaled, y_train)

        train_score = self.model.score(X_scaled, y_train)
        print(f"[INFO] Training R2 score: {train_score:.4f}")

        self.ready = True

    def predict(self, test_batch):
        """Predict positions using leak-safe baseline strategy"""
        self.train_if_needed()

        # Unpack batch
        if isinstance(test_batch, (tuple, list)) and len(test_batch) == 1:
            test_batch = test_batch[0]
        df = test_batch

        # Convert Polars to Pandas
        if pl is not None and hasattr(pl, 'DataFrame') and isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Create derived features
        df["U1"] = df["I2"] - df["I1"]
        df["U2"] = df["M11"] / ((df["I2"] + df["I9"] + df["I7"]) / 3)

        # Handle missing features
        for feat in self.features:
            if feat not in df.columns:
                df[feat] = 0.0

        # Select features
        X = df[self.features].fillna(0).replace([np.inf, -np.inf], 0).astype('float64')

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        position = self.model.predict(X_scaled)
        position = np.clip(position, 0.0, 2.0)

        # Return scalar
        return float(position[0])


if __name__ == '__main__':
    print("[START] Kaggle Hull Tactical Submission - EXP-036 (Leak-Safe Baseline)")
    print(f"[INFO] Current directory: {os.getcwd()}")
    print(f"[INFO] KAGGLE_IS_COMPETITION_RERUN: {os.getenv('KAGGLE_IS_COMPETITION_RERUN')}")

    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        print("[INFO] Running in COMPETITION RERUN mode - starting server")
        srv = MyServer()
        srv.serve()
    else:
        print("[INFO] Running in NOTEBOOK mode - generating submission.parquet")
        srv = MyServer()
        srv.server.start()
        try:
            # Find competition data
            candidates = [
                '/kaggle/input/hull-tactical-market-prediction',
                '.',
                'data',
            ]
            comp_dir = None
            for p in candidates:
                if os.path.exists(os.path.join(p, 'test.csv')):
                    comp_dir = p
                    print(f"[INFO] Found test.csv at: {comp_dir}")
                    break

            if comp_dir is None:
                print("[INFO] Running gateway with default paths")
                DefaultGateway().run()
            else:
                print(f"[INFO] Running gateway with data_paths: {comp_dir}")
                DefaultGateway(data_paths=(comp_dir,)).run()

            # Verify submission.parquet
            if os.path.exists('submission.parquet'):
                sub = pd.read_parquet('submission.parquet')
                print(f"[SUCCESS] submission.parquet created! Shape: {sub.shape}")
                print(f"[INFO] Prediction stats - Mean: {sub['prediction'].mean():.4f}, Std: {sub['prediction'].std():.4f}")
                print(f"[INFO] Prediction range: [{sub['prediction'].min():.4f}, {sub['prediction'].max():.4f}]")
                print("\n[INFO] First 5 predictions:")
                print(sub.head())
            else:
                print("[ERROR] submission.parquet not created!")
        finally:
            srv.server.stop(0)

        print("[END] Script complete")
