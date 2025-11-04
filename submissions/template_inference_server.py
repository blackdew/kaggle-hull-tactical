#!/usr/bin/env python3
"""
InferenceServer Template for Hull Tactical Market Prediction

이 파일은 Kaggle submission 생성을 위한 필수 구조를 담고 있습니다.
새로운 실험을 제출할 때 이 템플릿을 복사하여 사용하세요.
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
    """InferenceServer 구현 템플릿"""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBRegressor

        self.StandardScaler = StandardScaler
        self.XGBRegressor = XGBRegressor
        self.ready = False

        # TODO: 모델 및 feature 설정
        self.model = None
        self.scaler = None
        self.features = []
        self.K = 250

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

        # train.csv 로드 (경로 처리 - Kaggle/로컬 모두 대응)
        try:
            train = pd.read_csv("/kaggle/input/hull-tactical-market-prediction/train.csv")
        except:
            train = pd.read_csv("data/train.csv")

        y = train['market_forward_excess_returns'].values

        # TODO: Feature 생성
        X = self.create_features(train)

        # TODO: 모델 학습
        self.scaler = self.StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = self.XGBRegressor(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.025,
            subsample=1.0,
            colsample_bytree=0.6,
            reg_lambda=0.5,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)

        self.ready = True

    def create_features(self, df):
        """Feature 생성 (1-row에서 계산 가능해야 함)"""
        # REQUIRED: Polars → Pandas 변환
        if pl is not None and isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # REQUIRED: 기본 features 존재 확인
        for feat in self.features:
            if feat not in df.columns:
                df[feat] = 0.0

        # REQUIRED: Missing value 처리
        df = df.fillna(0).replace([np.inf, -np.inf], 0)

        # TODO: Feature engineering 로직 구현
        # - Interaction features (곱셈, 나눗셈, 다항식)
        # - 모든 feature는 현재 행 데이터만으로 계산되어야 함
        # - lag/rolling features 사용 불가

        return df[self.features]

    def predict(self, test_batch):
        """Row-by-row 예측"""
        self.train_if_needed()

        # REQUIRED: Batch unpacking
        if isinstance(test_batch, (tuple, list)) and len(test_batch) == 1:
            test_batch = test_batch[0]
        df = test_batch

        # REQUIRED: Polars → Pandas 변환
        if pl is not None and hasattr(pl, 'DataFrame') and isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Feature 생성
        X = self.create_features(df)

        # REQUIRED: Missing features 처리
        for feat in self.features:
            if feat not in X.columns:
                X[feat] = 0.0

        # Feature 선택 및 타입 변환
        X = X[self.features].astype('float64')
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 예측
        X_scaled = self.scaler.transform(X)
        excess_return = self.model.predict(X_scaled)

        # Position 계산
        position = np.clip(1.0 + excess_return * self.K, 0.0, 2.0)

        # REQUIRED: Scalar float 반환 (numpy array 아님!)
        return float(position[0])


# REQUIRED: Main 블록 구현
if __name__ == '__main__':
    print("[START] Kaggle Hull Tactical Submission")
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
