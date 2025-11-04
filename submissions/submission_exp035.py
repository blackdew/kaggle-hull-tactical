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

        # EXP-028 top 105 features (will be loaded)
        self.features_105 = []

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
        print(f"[EXP-035] Stored {len(self.y_train_sorted)} sorted training values for rank conversion")

        # Load EXP-028 features (top 105)
        try:
            # Try Kaggle paths first
            features_df = pd.read_csv("/kaggle/input/hull-tactical-market-prediction/experiments/028/results/top105_features.csv")
        except:
            # Fall back to local
            features_df = pd.read_csv("experiments/028/results/top105_features.csv")

        self.features_105 = features_df['feature'].tolist()
        print(f"[EXP-035] Loaded {len(self.features_105)} features from EXP-028")

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

        # Load phase2 engineered features
        try:
            # Try Kaggle first
            engineered = pd.read_parquet("/kaggle/input/hull-tactical-market-prediction/experiments/024/results/phase2_engineered_features.parquet")
            phase2_list = pd.read_csv("/kaggle/input/hull-tactical-market-prediction/experiments/024/results/phase2_feature_list.csv")
        except:
            # Fall back to local
            engineered = pd.read_parquet("experiments/024/results/phase2_engineered_features.parquet")
            phase2_list = pd.read_csv("experiments/024/results/phase2_feature_list.csv")

        # Get original features
        original_features = phase2_list[phase2_list['type'] == 'original']['feature'].tolist()
        original_data = df[original_features].copy()

        # Combine original + engineered
        X_all = pd.concat([original_data, engineered], axis=1)

        # Select top 105 features
        # Handle case where some features might be missing
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


# Main execution
if __name__ == "__main__":
    # Create server instance
    server = MyServer()

    # Get Gateway (for local testing)
    data_paths = {
        "train": "data/train.csv",
        "test": "data/test.csv",
    }

    try:
        gateway = server._get_gateway_for_test(data_paths=data_paths)
        print("[Main] Gateway created successfully")

        # Process all test batches
        print("[Main] Processing test batches...")
        for i, batch in enumerate(gateway):
            prediction = server.predict(batch)
            if i < 3:  # Print first 3 predictions
                print(f"[Main] Batch {i}: position = {prediction:.4f}")

        print("[Main] All batches processed successfully")
        print("[Main] submission.parquet should be created")

    except Exception as e:
        print(f"[Main] Error: {e}")
        import traceback
        traceback.print_exc()
