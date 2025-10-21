"""
EXP-016 v2: InferenceServer with Original + Interaction Features

Model: XGBoost with Top 30 features (original + interactions)
Features: 1-row calculable (no lag/rolling/ema)
Expected Sharpe: ~0.56 (5-fold CV)
K = 250
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


class EXP016V2Server(InferenceServer):
    """EXP-016 v2: Original + Interaction Features (InferenceServer compatible)"""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBRegressor

        self.StandardScaler = StandardScaler
        self.XGBRegressor = XGBRegressor
        self.ready = False

        # Top 30 features (from Phase 2)
        self.top_30_features = [
            'P8*S2', 'M4*V7', 'P8/P7', 'V7*P7', 'M4/S2',
            'S2*S5', 'S5/P7', 'M4*P8', 'M4²', 'V13²',
            'V7/P7', 'P8²', 'V7*I2', 'I2*E19', 'M4/P8',
            'S2/P5', 'V7*P5', 'P5', 'P5/P7', 'V7/P8',
            'M4/I2', 'M4/V7', 'M4/P5', 'P8/P5', 'V13/S2',
            'V13*I2', 'M4/E19', 'M4/P7', 'I2/S5', 'V13/P7'
        ]

        # Top 20 base features (needed for creating interactions)
        self.top_20_base = [
            'M4', 'V13', 'V7', 'P8', 'S2', 'I2', 'E19', 'S5', 'P5', 'P7',
            'M2', 'V9', 'M3', 'P12', 'P10', 'V10', 'E12', 'P11', 'M12', 'S8'
        ]

        # Model hyperparameters
        self.params = {
            'n_estimators': 150,
            'max_depth': 7,
            'learning_rate': 0.025,
            'subsample': 1.0,
            'colsample_bytree': 0.6,
            'reg_lambda': 0.5,
            'random_state': 42,
            'n_jobs': -1
        }

        # Position sizing
        self.k = 250.0

        def predict(batch):
            return EXP016V2Server.predict(self, batch)
        super().__init__(predict)

    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None, *args, **kwargs):
        return DefaultGateway(data_paths)

    def create_features(self, df):
        """Create interaction features from base features (1-row calculable)"""
        # Ensure base features exist
        for feat in self.top_20_base:
            if feat not in df.columns:
                df[feat] = 0.0

        # Fill missing
        df = df.fillna(0).replace([np.inf, -np.inf], 0)

        eps = 1e-8
        features = {}

        # Extract base features for top 10
        top_10 = self.top_20_base[:10]

        # Create all interactions
        # Multiplication
        for i, feat1 in enumerate(top_10):
            for feat2 in top_10[i+1:]:
                name = f'{feat1}*{feat2}'
                features[name] = df[feat1] * df[feat2]

        # Division
        for i, feat1 in enumerate(top_10):
            for feat2 in top_10[i+1:]:
                name = f'{feat1}/{feat2}'
                features[name] = df[feat1] / (df[feat2].abs() + eps)

        # Polynomial for top 5
        top_5 = self.top_20_base[:5]
        for feat in top_5:
            features[f'{feat}²'] = df[feat] ** 2
            features[f'{feat}³'] = df[feat] ** 3

        # Add base features
        for feat in self.top_20_base:
            features[feat] = df[feat]

        return pd.DataFrame(features)

    def train_if_needed(self):
        if self.ready:
            return

        print("[INFO] Starting training...")

        # Load train.csv
        def _load_train():
            candidates = [
                'train.csv',
                './train.csv',
                '/kaggle/input/hull-tactical-market-prediction/train.csv',
                '/kaggle/working/train.csv',
                'data/train.csv',
            ]
            for path in candidates:
                if os.path.exists(path):
                    print(f"[INFO] Found train.csv at: {path}")
                    return pd.read_csv(path)
            raise FileNotFoundError('train.csv not found')

        train = _load_train()
        print(f"[INFO] Train data shape: {train.shape}")

        # Create features
        X_all = self.create_features(train)
        print(f"[INFO] Features created: {X_all.shape}")

        # Select Top 30
        missing = [f for f in self.top_30_features if f not in X_all.columns]
        if missing:
            print(f"[WARN] Missing features: {missing}")
            for f in missing:
                X_all[f] = 0.0

        X = X_all[self.top_30_features].copy()
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        y = train['market_forward_excess_returns'].to_numpy()

        # Scale
        self.scaler = self.StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train XGBoost
        print("[INFO] Training XGBoost model...")
        self.model = self.XGBRegressor(**self.params)
        self.model.fit(X_scaled, y)
        print(f"[INFO] Training complete. k={self.k}")

        self.ready = True

    def predict(self, test_batch):
        self.train_if_needed()

        # Unpack batch
        if isinstance(test_batch, (tuple, list)) and len(test_batch) == 1:
            test_batch = test_batch[0]
        df = test_batch

        # Convert Polars to Pandas
        if pl is not None and hasattr(pl, 'DataFrame') and isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Create features for this row
        X_all = self.create_features(df)

        # Handle missing features
        missing = [f for f in self.top_30_features if f not in X_all.columns]
        for f in missing:
            X_all[f] = 0.0

        # Select Top 30
        X = X_all[self.top_30_features].astype('float64')
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        excess_return = self.model.predict(X_scaled)

        # Convert to position
        position = np.clip(1.0 + excess_return * self.k, 0.0, 2.0)

        return float(position[0])


if __name__ == '__main__':
    print("[START] Kaggle Hull Tactical Submission - EXP-016 v2 (Sharpe 0.56+)")
    print(f"[INFO] Current directory: {os.getcwd()}")
    print(f"[INFO] KAGGLE_IS_COMPETITION_RERUN: {os.getenv('KAGGLE_IS_COMPETITION_RERUN')}")

    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        print("[INFO] Running in COMPETITION RERUN mode - starting server")
        srv = EXP016V2Server()
        srv.serve()
    else:
        print("[INFO] Running in NOTEBOOK mode - generating submission.parquet")
        srv = EXP016V2Server()
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
