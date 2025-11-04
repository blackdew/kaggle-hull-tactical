#!/usr/bin/env python3
"""
EXP-035 Submission: Best Configuration

CV Sharpe: 0.809 (+27.1% vs EXP-021)
Expected Public: 6.96

Configuration:
- Features: EXP-028 (105 features from phase2 engineering + top by MI)
- Loss: MAE (reg:absoluteerror)
- Target: Rank-based with quantile mapping
- Hyperparameters: lr=0.05, depth=4, K=400
"""

from kaggle_evaluation.core.templates import InferenceServer
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

try:
    from default_gateway import DefaultGateway
except Exception:
    from kaggle_evaluation.default_gateway import DefaultGateway

try:
    import polars as pl
except Exception:
    pl = None


class MyServer(InferenceServer):
    """EXP-035: Optimized MAE + Rank + 105 Features"""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBRegressor

        self.StandardScaler = StandardScaler
        self.XGBRegressor = XGBRegressor
        self.ready = False

        # Model components
        self.model = None
        self.scaler = None
        self.y_train_sorted = None  # For rank-to-value conversion

        # Best hyperparameters from EXP-035
        self.K = 400  # Position sizing parameter

        # EXP-028 top 105 features (hardcoded for Kaggle)
        self.features_105 = [
            'V7_sqrt', 'E19', 'V7_log', 'E19_sqrt', 'E19_log', 'V7', 'V7/E2', 'V7/E3', 'V7/V13', 'V7/P2',
            'E19/E2', 'E19/E3', 'V13_sqrt', 'V13_log', 'S8_sqrt', 'S8_log', 'S8', 'V13', 'V7*E20', 'V13/E2',
            'P5_log', 'S8/E3', 'P5', 'V13/E3', 'E19/P2', 'P5_sqrt', 'V9/E20', 'E2_sqrt', 'E2_log', 'E3_sqrt',
            'E19*E20', 'E3_log', 'E2', 'V9_sqrt', 'V13*E20', 'S8/E20', 'V9_log', 'V9', 'P5*E20', 'E3',
            'V13/P2', 'V7*P2', 'V7*E2', 'S8/P2', 'E19*P2', 'E20', 'E19*S8', 'E_range', 'E19*E3', 'V_mean',
            'vol_mean', 'P10_sqrt', 'I9_log', 'V9/P2', 'E19*P5', 'V1_sqrt', 'E1_sqrt', 'S8/E2', 'econ_uncertainty',
            'E_std', 'P8_sqrt', 'P2_log', 'P2_sqrt', 'I9_sqrt', 'E19_squared', 'E20_log', 'P2', 'I9',
            'E1_log', 'P5_squared', 'P10_log', 'E20_sqrt', 'E19*E2', 'P10', 'E17', 'E19*V13', 'P5*E2', 'V9*E20',
            'V7*E3', 'E1', 'S8*E3', 'E20*P2', 'M4', 'E12', 'V7*V13', 'V7*P5', 'E18', 'P8_log',
            'V1_log', 'V13*E3', 'E2/P2', 'V13_squared', 'E10_squared', 'I4', 'E17_sqrt', 'I2_log', 'vol_composite', 'E3/P2',
            'S2', 'V_skew_proxy', 'I5', 'E17_log', 'S8_squared', 'E1_squared', 'E18_log'
        ]

        # REQUIRED: predict 함수를 super().__init__()에 전달
        def predict(batch):
            return MyServer.predict(self, batch)
        super().__init__(predict)

    # REQUIRED: Gateway 메서드 구현
    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None, *args, **kwargs):
        return DefaultGateway(data_paths)

    def train_if_needed(self):
        """첫 예측 시 lazy training"""
        if self.ready:
            return

        print("[EXP-035] Starting training...")

        # Load train.csv (경로 처리 - Kaggle/로컬 모두 대응)
        try:
            train = pd.read_csv("/kaggle/input/hull-tactical-market-prediction/train.csv")
            print(f"[EXP-035] Loaded train data from Kaggle: {train.shape}")
        except:
            train = pd.read_csv("data/train.csv")
            print(f"[EXP-035] Loaded train data locally: {train.shape}")

        y_excess = train['market_forward_excess_returns'].values

        # Convert to rank (0~1 scale)
        y_rank = rankdata(y_excess, method='average') / len(y_excess)

        # Store sorted values for rank-to-value conversion during prediction
        self.y_train_sorted = np.sort(y_excess)
        print(f"[EXP-035] Stored {len(self.y_train_sorted)} sorted training values")
        print(f"[EXP-035] Using {len(self.features_105)} features")

        # Create features
        X = self.create_features(train)
        print(f"[EXP-035] Created feature matrix: {X.shape}")

        # Scale features
        self.scaler = self.StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train model with MAE loss on rank target
        self.model = self.XGBRegressor(
            n_estimators=150,
            max_depth=4,  # Best from EXP-035
            learning_rate=0.05,  # Best from EXP-035
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='reg:absoluteerror',  # MAE loss
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y_rank, verbose=False)
        print("[EXP-035] Model training complete")

        self.ready = True

    def create_features(self, df):
        """Create EXP-028 features (105 features)"""
        # REQUIRED: Polars → Pandas 변환
        if pl is not None and isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Simplified: Create engineered features from scratch
        # This avoids dependency on pre-computed feature files

        # Get base features (original columns)
        base_cols = [c for c in df.columns if c not in ['date_id', 'market_forward_excess_returns', 'forward_returns', 'risk_free_rate']]
        X_all = df[base_cols].copy()

        # Create nonlinear transforms
        for col in base_cols:
            if col.startswith(('V', 'E', 'P', 'S', 'I', 'M')):
                X_all[f'{col}_sqrt'] = np.sqrt(np.abs(X_all[col]))
                X_all[f'{col}_log'] = np.log1p(np.abs(X_all[col]))
                X_all[f'{col}_squared'] = X_all[col] ** 2

        # Create interactions (multiplication)
        key_features = ['V7', 'E19', 'V13', 'S8', 'P5', 'E2', 'E3', 'E20', 'V9', 'P2']
        for i, col1 in enumerate(key_features):
            if col1 in X_all.columns:
                for col2 in key_features[i+1:]:
                    if col2 in X_all.columns:
                        X_all[f'{col1}*{col2}'] = X_all[col1] * X_all[col2]

        # Create interactions (division)
        for col1 in key_features:
            if col1 in X_all.columns:
                for col2 in key_features:
                    if col2 in X_all.columns and col1 != col2:
                        X_all[f'{col1}/{col2}'] = X_all[col1] / (X_all[col2].abs() + 1e-8)

        # Create category statistics
        if 'E' in ''.join(base_cols):
            e_cols = [c for c in base_cols if c.startswith('E')]
            if len(e_cols) > 0:
                X_all['E_range'] = X_all[e_cols].max(axis=1) - X_all[e_cols].min(axis=1)
                X_all['E_std'] = X_all[e_cols].std(axis=1)

        if 'V' in ''.join(base_cols):
            v_cols = [c for c in base_cols if c.startswith('V')]
            if len(v_cols) > 0:
                X_all['V_mean'] = X_all[v_cols].mean(axis=1)
                X_all['V_skew_proxy'] = (X_all[v_cols].max(axis=1) - X_all[v_cols].min(axis=1)) / (X_all[v_cols].std(axis=1) + 1e-8)

        # Create volatility features
        if 'V7' in X_all.columns and 'V13' in X_all.columns:
            X_all['vol_mean'] = (X_all['V7'] + X_all['V13']) / 2
            X_all['vol_composite'] = np.sqrt(X_all['V7']**2 + X_all['V13']**2)

        if 'E_std' in X_all.columns:
            X_all['econ_uncertainty'] = X_all['E_std']

        # Select top 105 features
        available_features = [f for f in self.features_105 if f in X_all.columns]
        if len(available_features) < len(self.features_105):
            print(f"[WARNING] Only {len(available_features)}/{len(self.features_105)} features available")

        X = X_all[available_features].copy()

        # REQUIRED: Missing value 처리
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        return X

    def predict(self, test_batch):
        """Predict positions for test batch"""
        # Train if not ready
        self.train_if_needed()

        # REQUIRED: Unpack batch (date_id와 DataFrame이 튜플로 전달됨)
        if isinstance(test_batch, tuple):
            date_id, test_df = test_batch
        else:
            test_df = test_batch

        # REQUIRED: Polars → Pandas 변환
        if pl is not None and isinstance(test_df, pl.DataFrame):
            test_df = test_df.to_pandas()

        # Create features
        X_test = self.create_features(test_df)

        # Scale
        X_test_scaled = self.scaler.transform(X_test)

        # Predict rank (0~1)
        y_pred_rank = self.model.predict(X_test_scaled)
        y_pred_rank = np.clip(y_pred_rank, 0.0, 1.0)

        # Convert rank back to value using training distribution
        indices = (y_pred_rank * (len(self.y_train_sorted) - 1)).astype(int)
        y_pred = self.y_train_sorted[indices]

        # Calculate position
        position = 1.0 + y_pred * self.K
        position = np.clip(position, 0.0, 2.0)

        # REQUIRED: Return scalar float (not array)
        return float(position[0])


# REQUIRED: Main 블록 구현
if __name__ == '__main__':
    print("[START] EXP-035 Submission")
    print(f"[INFO] Current directory: {os.getcwd()}")
    print(f"[INFO] KAGGLE_IS_COMPETITION_RERUN: {os.getenv('KAGGLE_IS_COMPETITION_RERUN')}")

    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        # Competition rerun mode
        print("[INFO] Running in COMPETITION RERUN mode")
        srv = MyServer()
        srv.serve()
    else:
        # Notebook mode - generate submission.parquet
        print("[INFO] Running in NOTEBOOK mode")
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
