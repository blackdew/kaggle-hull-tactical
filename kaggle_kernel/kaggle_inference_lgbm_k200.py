import os
import numpy as np
import pandas as pd

from kaggle_evaluation.core.templates import InferenceServer

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from default_gateway import DefaultGateway
except Exception:
    from kaggle_evaluation.default_gateway import DefaultGateway


class LightGBMServer(InferenceServer):
    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.StandardScaler = StandardScaler
        self.ready = False

        def predict(batch):
            return LightGBMServer.predict(self, batch)

        super().__init__(predict)

    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None, *args, **kwargs):
        return DefaultGateway(data_paths)

    def _load_train(self) -> pd.DataFrame:
        candidates = [
            '/kaggle/input/hull-tactical-market-prediction/train.csv',
            'train.csv',
            './train.csv',
            'data/train.csv',
        ]
        print("[INFO] Searching for train.csv...")
        for path in candidates:
            if os.path.exists(path):
                print(f"[INFO] Found train.csv at: {path}")
                return pd.read_csv(path)
        raise FileNotFoundError('train.csv not found')

    def select_features(self, df: pd.DataFrame):
        exclude = {
            'date_id', 'forward_returns', 'risk_free_rate',
            'market_forward_excess_returns', 'is_scored',
            'lagged_forward_returns', 'lagged_risk_free_rate',
            'lagged_market_forward_excess_returns'
        }
        return [c for c in df.columns if c not in exclude]

    def train_if_needed(self):
        if self.ready:
            return

        if lgb is None:
            raise ImportError("LightGBM not available")

        print("[INFO] Starting training...")
        train = self._load_train()
        print(f"[INFO] Train data shape: {train.shape}")

        target = 'market_forward_excess_returns'
        features = self.select_features(train)
        print(f"[INFO] Features: {len(features)}")

        X = train[features].copy()
        X = X.fillna(X.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)

        self.scaler = self.StandardScaler()
        Xs = self.scaler.fit_transform(X)
        y = train[target].to_numpy()

        print("[INFO] Training LightGBM model...")
        self.model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.01,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
        self.model.fit(Xs, y)

        self.features = features
        self.k = 200.0  # Best from EXP-005: H2 k=200, Sharpe 0.611
        print(f"[INFO] Training complete. k={self.k}")
        self.ready = True

    def predict(self, test_batch):
        self.train_if_needed()

        if isinstance(test_batch, (tuple, list)) and len(test_batch) == 1:
            test_batch = test_batch[0]

        df = test_batch

        # Ensure all features exist
        for c in self.features:
            if c not in df.columns:
                df[c] = 0.0

        X = df[self.features].astype('float64')
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        Xs = self.scaler.transform(X)

        excess = self.model.predict(Xs)
        positions = np.clip(1.0 + excess * self.k, 0.0, 2.0)

        return float(positions[0])


if __name__ == '__main__':
    print("[START] Kaggle Hull Tactical Submission - EXP-005 H2 LightGBM k=200")
    print(f"[INFO] Current directory: {os.getcwd()}")

    is_rerun = os.getenv('KAGGLE_IS_COMPETITION_RERUN')
    print(f"[INFO] KAGGLE_IS_COMPETITION_RERUN: {is_rerun}")

    if is_rerun:
        print("[INFO] Running in RERUN mode - starting server only")
        srv = LightGBMServer()
        srv.serve()
    else:
        print("[INFO] Running in NOTEBOOK mode - generating submission.parquet")
        srv = LightGBMServer()
        srv.server.start()
        try:
            candidates = [
                '/kaggle/input/hull-tactical-market-prediction',
                '.',
                'data',
            ]
            comp_dir = None
            print("[INFO] Searching for test.csv...")
            for p in candidates:
                test_path = os.path.join(p, 'test.csv')
                if os.path.exists(test_path):
                    comp_dir = p
                    print(f"[INFO] Found test.csv at: {comp_dir}")
                    break

            if comp_dir is None:
                print("[WARNING] test.csv not found, using default gateway")
                DefaultGateway().run()
            else:
                print(f"[INFO] Running gateway with data_paths: {comp_dir}")
                DefaultGateway(data_paths=(comp_dir,)).run()

            print("[INFO] Checking for submission.parquet...")
            if os.path.exists('submission.parquet'):
                sub = pd.read_parquet('submission.parquet')
                print(f"[SUCCESS] submission.parquet created! Shape: {sub.shape}")
                print(f"[INFO] Prediction stats - Mean: {sub['prediction'].mean():.4f}, Std: {sub['prediction'].std():.4f}")
                print(f"[INFO] Prediction range: [{sub['prediction'].min():.4f}, {sub['prediction'].max():.4f}]")
                print("\n[INFO] First 5 predictions:")
                print(sub.head())
            else:
                print("[ERROR] submission.parquet was NOT created!")
                print(f"[INFO] Working directory files: {os.listdir('.')}")

        except Exception as e:
            print(f"[ERROR] Exception during execution: {e}")
            import traceback
            traceback.print_exc()
        finally:
            srv.server.stop(0)
            print("[END] Script complete")
