# %% [code]
#!/usr/bin/env python3
"""
EXP-038 v2: Regime-Based Quantile Regression
Dynamically selects Low/High Volatility models based on V13 value.
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
    """EXP-038 v2: Regime-Based Quantile Regression"""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBRegressor

        self.StandardScaler = StandardScaler
        self.XGBRegressor = XGBRegressor
        self.ready = False
        
        # Models
        self.low_models = {}
        self.high_models = {}
        self.low_scaler = None
        self.high_scaler = None
        
        self.K = 250
        
        # Threshold (V13 Median) - Hardcoded from Phase 1 result
        # Ideally read from file, but hardcoded for submission safety
        self.threshold = 0.000456  # Example value, should be updated with actual
        
        # EXP-016 Top 30 Features
        self.top_30 = [
            'P8*S2', 'M4*V7', 'P8/P7', 'V7*P7', 'M4/S2',
            'S2*S5', 'S5/P7', 'M4*P8', 'M4²', 'V13²',
            'V7/P7', 'P8²', 'V7*I2', 'I2*E19', 'M4/P8',
            'S2/P5', 'V7*P5', 'P5', 'P5/P7', 'V7/P8',
            'M4/I2', 'M4/V7', 'M4/P5', 'P8/P5', 'V13/S2',
            'V13*I2', 'M4/E19', 'M4/P7', 'I2/S5', 'V13/P7'
        ]
        
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
        v13 = train['V13'].fillna(0).values
        
        # Update threshold dynamically if possible, or keep hardcoded
        self.threshold = np.median(v13)
        print(f"[INFO] Regime Threshold (V13 Median): {self.threshold}")

        # Create features
        X = self.create_features(train)
        
        # Split by Regime
        low_mask = v13 <= self.threshold
        high_mask = v13 > self.threshold
        
        X_low, y_low = X[low_mask], y[low_mask]
        X_high, y_high = X[high_mask], y[high_mask]
        
        # Train Low Models
        self.low_scaler = self.StandardScaler()
        X_low_scaled = self.low_scaler.fit_transform(X_low)
        
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
        
        for alpha, name in [(0.1, 'q10'), (0.5, 'q50'), (0.9, 'q90')]:
            model = self.XGBRegressor(**common_params, quantile_alpha=alpha)
            model.fit(X_low_scaled, y_low)
            self.low_models[name] = model
            
        # Train High Models
        self.high_scaler = self.StandardScaler()
        X_high_scaled = self.high_scaler.fit_transform(X_high)
        
        for alpha, name in [(0.1, 'q10'), (0.5, 'q50'), (0.9, 'q90')]:
            model = self.XGBRegressor(**common_params, quantile_alpha=alpha)
            model.fit(X_high_scaled, y_high)
            self.high_models[name] = model

        self.ready = True

    def create_features(self, df):
        """Create features from raw data"""
        if pl is not None and isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        for feat in self.top_20:
            if feat not in df.columns:
                df[feat] = 0.0

        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        X_base = df[self.top_20]

        interactions = []
        feature_names = self.top_20.copy()
        top_10 = self.top_20[:10]
        eps = 1e-8

        for i, feat1 in enumerate(top_10):
            for feat2 in top_10[i + 1 :]:
                interactions.append(X_base[feat1] * X_base[feat2])
                feature_names.append(f'{feat1}*{feat2}')

        for i, feat1 in enumerate(top_10):
            for feat2 in top_10[i + 1 :]:
                interactions.append(X_base[feat1] / (X_base[feat2].abs() + eps))
                feature_names.append(f'{feat1}/{feat2}')

        top_5 = self.top_20[:5]
        for feat in top_5:
            interactions.append(X_base[feat] ** 2)
            feature_names.append(f'{feat}²')
            interactions.append(X_base[feat] ** 3)
            feature_names.append(f'{feat}³')

        X_all = pd.concat(
            [X_base] + [pd.Series(feat, name=name) for feat, name in zip(interactions, feature_names[len(self.top_20) :])],
            axis=1
        )
        X_all.columns = feature_names
        X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        X = X_all[self.top_30]
        return X

    def predict(self, test_batch):
        """Predict positions using Regime-Based Strategy"""
        self.train_if_needed()

        if isinstance(test_batch, (tuple, list)) and len(test_batch) == 1:
            test_batch = test_batch[0]
        df = test_batch

        if pl is not None and hasattr(pl, 'DataFrame') and isinstance(df, pl.DataFrame):
            df = df.to_pandas()
            
        # Get V13 for regime determination
        v13_val = df['V13'].fillna(0).values[0] if 'V13' in df.columns else 0.0

        # Create features
        X = self.create_features(df)
        missing = [f for f in self.top_30 if f not in X.columns]
        for f in missing:
            X[f] = 0.0
        X = X[self.top_30].astype('float64')
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Select Model based on Regime
        if v13_val <= self.threshold:
            # Low Vol Regime
            X_scaled = self.low_scaler.transform(X)
            q10 = self.low_models['q10'].predict(X_scaled)
            q50 = self.low_models['q50'].predict(X_scaled)
            q90 = self.low_models['q90'].predict(X_scaled)
        else:
            # High Vol Regime
            X_scaled = self.high_scaler.transform(X)
            q10 = self.high_models['q10'].predict(X_scaled)
            q50 = self.high_models['q50'].predict(X_scaled)
            q90 = self.high_models['q90'].predict(X_scaled)

        # Calculate Position
        ci_width = q90 - q10
        confidence = 1.0 / (np.abs(ci_width) + 0.001)
        confidence = np.clip(confidence, 0.5, 5.0)
        position = 1.0 + q50 * self.K * confidence
        position = np.clip(position, 0.0, 2.0)

        return float(position[0])


if __name__ == '__main__':
    print("[START] Kaggle Hull Tactical Submission - EXP-038 v2")
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
            candidates = ['/kaggle/input/hull-tactical-market-prediction', '.', 'data']
            comp_dir = None
            for p in candidates:
                if os.path.exists(os.path.join(p, 'test.csv')):
                    comp_dir = p
                    break
            
            if comp_dir:
                DefaultGateway(data_paths=(comp_dir,)).run()
            else:
                DefaultGateway().run()

            if os.path.exists('submission.parquet'):
                sub = pd.read_parquet('submission.parquet')
                print(f"[SUCCESS] submission.parquet created! Shape: {sub.shape}")
                print(sub.head())
        finally:
            srv.server.stop(0)
