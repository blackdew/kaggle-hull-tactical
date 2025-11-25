# %% [code]
#!/usr/bin/env python3
"""
EXP-038 v3: Hybrid Feature Set (MSE + Quantile)
Combines EXP-016 (MSE) and EXP-038 v1 (Quantile) features.
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
    """EXP-038 v3: Hybrid Feature Set (MSE + Quantile)"""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBRegressor

        self.StandardScaler = StandardScaler
        self.XGBRegressor = XGBRegressor
        self.ready = False
        self.xgb_q10 = None
        self.xgb_q50 = None
        self.xgb_q90 = None
        self.scaler = None
        self.K = 250  # Fixed K

        # Hybrid Feature Set (Updated with Phase 1 results)
        self.hybrid_features = [
            'E12', 'E19', 'E19*S5', 'E19/P7', 'E19/S5', 'I2*E19', 'I2*S5', 'I2/S5', 'M4*E19', 'M4*I2',
            'M4*P8', 'M4*V7', 'M4/E19', 'M4/I2', 'M4/P7', 'M4/P8', 'M4/S2', 'M4/V7', 'M4²', 'P5',
            'P5/P7', 'P8*E19', 'P8*S2', 'P8/P5', 'P8/P7', 'P8/S2', 'P8²', 'S2*E19', 'S2*I2', 'S2*S5',
            'S2/P5', 'S2/P7', 'S5/P7', 'S8', 'V13*I2', 'V13*V7', 'V13/S2', 'V13/S5', 'V13/P7', 'V13²',
            'V7*E19', 'V7*I2', 'V7*P5', 'V7*P7', 'V7*P8', 'V7*S2', 'V7/I2', 'V7/P7', 'V7/P8', 'V7/S2',
            'V7/S5'
        ]
        
        # Base features needed to generate interactions
        self.top_20 = [
            'M4', 'V13', 'V7', 'P8', 'S2', 'I2', 'E19', 'S5', 'P5', 'P7',
            'M2', 'V9', 'M3', 'P12', 'P10', 'V10', 'E12', 'P11', 'M12', 'S8'
        ]

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
        y = train['market_forward_excess_returns'].values

        # Create features
        X = self.create_features(train)

        # Scale features
        self.scaler = self.StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train quantile models
        common_params = {
            'n_estimators': 150,
            'max_depth': 7,
            'learning_rate': 0.025,
            'subsample': 1.0,
            'colsample_bytree': 0.6,
            'reg_lambda': 0.5,
            'objective': 'reg:quantileerror',
            'random_state': 42,
            'n_jobs': -1
        }

        self.xgb_q10 = self.XGBRegressor(**common_params, quantile_alpha=0.1)
        self.xgb_q10.fit(X_scaled, y)

        self.xgb_q50 = self.XGBRegressor(**common_params, quantile_alpha=0.5)
        self.xgb_q50.fit(X_scaled, y)

        self.xgb_q90 = self.XGBRegressor(**common_params, quantile_alpha=0.9)
        self.xgb_q90.fit(X_scaled, y)

        self.ready = True

    def create_features(self, df):
        """Create features from raw data"""
        # Convert polars to pandas if needed
        if pl is not None and isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Ensure base features exist
        for feat in self.top_20:
            if feat not in df.columns:
                df[feat] = 0.0

        # Fill missing
        df = df.fillna(0).replace([np.inf, -np.inf], 0)

        # Load base features
        X_base = df[self.top_20]

        # Create interactions
        interactions = []
        feature_names = self.top_20.copy()

        top_10 = self.top_20[:10]
        eps = 1e-8

        # Multiplication
        for i, feat1 in enumerate(top_10):
            for feat2 in top_10[i + 1 :]:
                interactions.append(X_base[feat1] * X_base[feat2])
                feature_names.append(f'{feat1}*{feat2}')

        # Division
        for i, feat1 in enumerate(top_10):
            for feat2 in top_10[i + 1 :]:
                interactions.append(X_base[feat1] / (X_base[feat2].abs() + eps))
                feature_names.append(f'{feat1}/{feat2}')

        # Polynomial
        top_5 = self.top_20[:5]
        for feat in top_5:
            interactions.append(X_base[feat] ** 2)
            feature_names.append(f'{feat}²')
            interactions.append(X_base[feat] ** 3)
            feature_names.append(f'{feat}³')

        # Combine
        X_all = pd.concat(
            [X_base] + [pd.Series(feat, name=name) for feat, name in zip(interactions, feature_names[len(self.top_20) :])],
            axis=1
        )
        X_all.columns = feature_names
        X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Select Hybrid Features
        # Ensure all required features exist (if not, fill 0)
        X = X_all[self.hybrid_features]

        return X

    def predict(self, test_batch):
        """Predict positions using quantile regression strategy"""
        self.train_if_needed()

        # Unpack batch
        if isinstance(test_batch, (tuple, list)) and len(test_batch) == 1:
            test_batch = test_batch[0]
        df = test_batch

        # Convert Polars to Pandas
        if pl is not None and hasattr(pl, 'DataFrame') and isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Create features (1-row at a time)
        X = self.create_features(df)

        # Handle missing features
        missing = [f for f in self.hybrid_features if f not in X.columns]
        for f in missing:
            X[f] = 0.0

        # Select Hybrid Features and ensure float64
        X = X[self.hybrid_features].astype('float64')
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict quantiles
        q10_pred = self.xgb_q10.predict(X_scaled)
        q50_pred = self.xgb_q50.predict(X_scaled)
        q90_pred = self.xgb_q90.predict(X_scaled)

        # Calculate confidence interval
        ci_width = q90_pred - q10_pred
        confidence = 1.0 / (np.abs(ci_width) + 0.001)
        confidence = np.clip(confidence, 0.5, 5.0)  # Scaled x5

        # Calculate positions
        position = 1.0 + q50_pred * self.K * confidence
        position = np.clip(position, 0.0, 2.0)

        # Return scalar
        return float(position[0])


if __name__ == '__main__':
    print("[START] Kaggle Hull Tactical Submission - EXP-038 v3")
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
