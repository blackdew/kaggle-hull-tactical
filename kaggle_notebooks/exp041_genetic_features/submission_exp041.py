# %% [code]
#!/usr/bin/env python3
"""
EXP-041: Genetic Feature Generation (Symbolic Regression)
Uses gplearn transformer to generate features + XGBoost for prediction.
"""
from kaggle_evaluation.core.templates import InferenceServer
import os
import numpy as np
import pandas as pd
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Install gplearn from dataset (offline)
import os
os.system('pip install /kaggle/input/gplearn-lib/gplearn-*.whl --no-index --find-links /kaggle/input/gplearn-lib')

try:
    from default_gateway import DefaultGateway
except Exception:
    from kaggle_evaluation.default_gateway import DefaultGateway

try:
    import polars as pl
except Exception:
    pl = None

# Import models
try:
    from xgboost import XGBRegressor
    from gplearn.genetic import SymbolicTransformer
except ImportError:
    print("[WARNING] One or more model libraries not found. Submission may fail.")

# Monkeypatch for gplearn compatibility (same as in Phase 1)
from sklearn.base import BaseEstimator
if not hasattr(BaseEstimator, "_validate_data"):
    def _validate_data(self, X, y=None, **kwargs):
        from sklearn.utils.validation import check_X_y, check_array
        if y is None:
            return check_array(X, **kwargs)
        else:
            return check_X_y(X, y, **kwargs)
    BaseEstimator._validate_data = _validate_data


class MyServer(InferenceServer):
    """EXP-041: Genetic Features + XGBoost"""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        
        self.StandardScaler = StandardScaler
        self.ready = False
        self.models = {}
        self.scaler_base = None
        self.scaler_comb = None
        self.est_gp = None
        self.K = 250

        # Base features needed for Genetic Transformer
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

        # 1. Prepare Base Features
        X_base = train[self.top_20].fillna(train[self.top_20].median()).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Scale base features
        self.scaler_base = self.StandardScaler()
        X_base_scaled = self.scaler_base.fit_transform(X_base)

        # 2. Load or Train Genetic Transformer
        # In a real submission, we should load a pre-trained transformer.
        # But here we will assume it's available or train a small one if not.
        transformer_path = "experiments/041_genetic_features/results/genetic_transformer.pkl"
        
        if os.path.exists(transformer_path):
            print(f"Loading Genetic Transformer from {transformer_path}...")
            self.est_gp = joblib.load(transformer_path)
        else:
            print("[WARNING] Genetic Transformer not found. Training a small one (fallback)...")
            # Fallback: Train a small GP model
            self.est_gp = SymbolicTransformer(
                generations=5, population_size=500, hall_of_fame=50, n_components=10,
                function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min'],
                parsimony_coefficient=0.0005, max_samples=0.9, verbose=0, random_state=42, n_jobs=-1
            )
            self.est_gp.fit(X_base_scaled, y)

        # 3. Generate Genetic Features
        X_genetic = self.est_gp.transform(X_base_scaled)
        genetic_cols = [f'GP_{i}' for i in range(X_genetic.shape[1])]
        X_genetic_df = pd.DataFrame(X_genetic, columns=genetic_cols)

        # 4. Combine Features
        X = pd.concat([X_base, X_genetic_df], axis=1)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Scale combined features
        self.scaler_comb = self.StandardScaler()
        X_scaled = self.scaler_comb.fit_transform(X)

        # 5. Train Quantile Models (XGBoost)
        quantiles = [0.1, 0.5, 0.9]
        q_names = ['q10', 'q50', 'q90']
        
        for q, name in zip(quantiles, q_names):
            print(f"Training XGBoost {name}...")
            model = XGBRegressor(
                n_estimators=150, max_depth=7, learning_rate=0.025,
                subsample=1.0, colsample_bytree=0.6, reg_lambda=0.5,
                objective='reg:quantileerror', quantile_alpha=q,
                random_state=42, n_jobs=-1
            )
            model.fit(X_scaled, y)
            self.models[name] = model

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

        # Base features
        X_base = df[self.top_20]
        
        # Scale base features
        X_base_scaled = self.scaler_base.transform(X_base)
        
        # Generate Genetic Features
        X_genetic = self.est_gp.transform(X_base_scaled)
        genetic_cols = [f'GP_{i}' for i in range(X_genetic.shape[1])]
        X_genetic_df = pd.DataFrame(X_genetic, columns=genetic_cols)
        
        # Combine
        X = pd.concat([X_base.reset_index(drop=True), X_genetic_df.reset_index(drop=True)], axis=1)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return X

    def predict(self, test_batch):
        """Predict positions using Genetic Features"""
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

        # Scale combined features
        X_scaled = self.scaler_comb.transform(X)

        # Predict quantiles
        q10_pred = self.models['q10'].predict(X_scaled)
        q50_pred = self.models['q50'].predict(X_scaled)
        q90_pred = self.models['q90'].predict(X_scaled)

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
    print("[START] Kaggle Hull Tactical Submission - EXP-041 (Genetic Features)")
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
