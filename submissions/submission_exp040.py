# %% [code]
#!/usr/bin/env python3
"""
EXP-040: Refined Hybrid Ensemble (Feature Selection on Hybrid Set)
Combines Refined Hybrid Features (Top 35) with EXP-022 (Ensemble).
Models: XGBoost, LightGBM, CatBoost
Features: 35 Refined Hybrid Features
"""
from kaggle_evaluation.core.templates import InferenceServer
import os
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

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
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
except ImportError:
    print("[WARNING] One or more model libraries not found. Submission may fail.")


class MyServer(InferenceServer):
    """EXP-040: Refined Hybrid Ensemble (XGB+LGB+Cat on 35 Refined Features)"""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        
        self.StandardScaler = StandardScaler
        self.ready = False
        self.models = {}  # Store all models here
        self.scaler = None
        self.K = 250  # Fixed K

        # Refined Hybrid Feature Set (35 features)
        self.refined_features = [
            'V7/P8', 'V7/P7', 'E19', 'V7*P5', 'V7/S5', 'S8', 'V7*P8', 'P5/P7', 'P8/S2', 'M4*E19',
            'P8*E19', 'V7*E19', 'E19/S5', 'P8*S2', 'E19/P7', 'M4/P8', 'V7*P7', 'V13/S5', 'P8/P7', 'M4/E19',
            'V13²', 'V13*V7', 'I2*E19', 'V7*S2', 'P8²', 'V13/S2', 'P8/P5', 'M4*P8', 'S5/P7', 'M4/I2',
            'V7/S2', 'V7*I2', 'V13/P7', 'P5', 'S2/P5'
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

        # Train models
        # We need q10, q50, q90 for XGB, LGB, Cat (Total 9 models)
        quantiles = [0.1, 0.5, 0.9]
        q_names = ['q10', 'q50', 'q90']
        model_types = ['xgb', 'lgb', 'cat']
        
        for m_type in model_types:
            self.models[m_type] = {}
            for q, q_name in zip(quantiles, q_names):
                print(f"Training {m_type} {q_name}...")
                
                if m_type == 'xgb':
                    model = XGBRegressor(
                        n_estimators=150, max_depth=7, learning_rate=0.025,
                        subsample=1.0, colsample_bytree=0.6, reg_lambda=0.5,
                        objective='reg:quantileerror', quantile_alpha=q,
                        random_state=42, n_jobs=-1
                    )
                elif m_type == 'lgb':
                    model = LGBMRegressor(
                        n_estimators=150, max_depth=7, learning_rate=0.025,
                        subsample=0.8, colsample_bytree=0.6, reg_lambda=0.5,
                        objective='quantile', alpha=q,
                        random_state=42, n_jobs=-1, verbose=-1
                    )
                elif m_type == 'cat':
                    model = CatBoostRegressor(
                        iterations=150, depth=7, learning_rate=0.025,
                        loss_function=f'Quantile:alpha={q}',
                        random_state=42, verbose=0, thread_count=-1
                    )
                
                model.fit(X_scaled, y)
                self.models[m_type][q_name] = model

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

        # Select Refined Features
        # Ensure all required features exist (if not, fill 0)
        X = X_all[self.refined_features]

        return X

    def predict(self, test_batch):
        """Predict positions using ensemble strategy"""
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
        missing = [f for f in self.refined_features if f not in X.columns]
        for f in missing:
            X[f] = 0.0

        # Select Refined Features and ensure float64
        X = X[self.refined_features].astype('float64')
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict with Ensemble
        q_names = ['q10', 'q50', 'q90']
        preds = {'q10': 0.0, 'q50': 0.0, 'q90': 0.0}
        
        for q_name in q_names:
            p_xgb = self.models['xgb'][q_name].predict(X_scaled)[0]
            p_lgb = self.models['lgb'][q_name].predict(X_scaled)[0]
            p_cat = self.models['cat'][q_name].predict(X_scaled)[0]
            
            # Simple Average
            preds[q_name] = (p_xgb + p_lgb + p_cat) / 3.0

        # Calculate confidence interval
        q10_pred = preds['q10']
        q50_pred = preds['q50']
        q90_pred = preds['q90']
        
        ci_width = q90_pred - q10_pred
        confidence = 1.0 / (np.abs(ci_width) + 0.001)
        confidence = np.clip(confidence, 0.5, 5.0)  # Scaled x5

        # Calculate positions
        position = 1.0 + q50_pred * self.K * confidence
        position = np.clip(position, 0.0, 2.0)

        # Return scalar
        return float(position)


if __name__ == '__main__':
    print("[START] Kaggle Hull Tactical Submission - EXP-040 (Refined Hybrid Ensemble)")
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
