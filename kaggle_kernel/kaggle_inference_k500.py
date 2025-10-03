import os
import numpy as np
import pandas as pd

from kaggle_evaluation.core.templates import InferenceServer

# Try optional Polars for gateway batches
try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None

# Import the competition gateway (path may vary in host)
try:
    from default_gateway import DefaultGateway  # type: ignore
except Exception:  # pragma: no cover
    from kaggle_evaluation.default_gateway import DefaultGateway  # type: ignore


class LassoTop20Server(InferenceServer):
    def __init__(self):
        # Lazy imports to minimize startup time
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import StandardScaler

        self.Lasso = Lasso
        self.StandardScaler = StandardScaler
        self.ready = False
        # Register a named function (not a bound method) per relay requirements
        def predict(batch):  # noqa: D401
            return LassoTop20Server.predict(self, batch)
        super().__init__(predict)

    # Required by InferenceServer: return a gateway instance for local tests
    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None, *args, **kwargs):
        return DefaultGateway(data_paths)

    def _load_train(self) -> pd.DataFrame:
        """Robust train loader for Kaggle + local."""
        candidates = [
            '/kaggle/input/hull-tactical-market-prediction/train.csv',  # Kaggle dataset mount
            'train.csv',
            './train.csv',
            '/kaggle/working/train.csv',
            'data/train.csv',  # local repo
        ]
        print("[INFO] Searching for train.csv...")
        for path in candidates:
            if os.path.exists(path):
                print(f"[INFO] Found train.csv at: {path}")
                return pd.read_csv(path)
        raise FileNotFoundError('train.csv not found in known locations: ' + ', '.join(candidates))

    def train_if_needed(self):
        if self.ready:
            return

        print("[INFO] Starting training...")
        train = self._load_train()
        print(f"[INFO] Train data shape: {train.shape}")

        target = 'market_forward_excess_returns'

        # base feature selection (exclude non-features)
        exclude = {
            'date_id', 'forward_returns', 'risk_free_rate',
            'market_forward_excess_returns', 'is_scored',
            'lagged_forward_returns', 'lagged_risk_free_rate',
            'lagged_market_forward_excess_returns'
        }
        feats = [c for c in train.columns if c not in exclude]

        # Top-20 by absolute Pearson corr (numeric only)
        num = train[feats + [target]].select_dtypes(include=[np.number])
        corr = num.corr(numeric_only=True)[target].drop(index=target).abs().sort_values(ascending=False)
        top20 = [c for c in corr.index[:20] if c in feats]
        print(f"[INFO] Top-20 features: {top20[:5]}... (total: {len(top20)})")

        X = train[top20].copy()
        X = X.fillna(X.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
        self.scaler = self.StandardScaler()
        Xs = self.scaler.fit_transform(X)
        y = train[target].to_numpy()

        print("[INFO] Training Lasso model...")
        self.model = self.Lasso(alpha=1e-4, max_iter=50000)
        self.model.fit(Xs, y)
        self.features = top20
        self.k = 500.0  # Updated from EXP-004: H3a_large_k500 (Sharpe 0.836)
        print(f"[INFO] Training complete. k={self.k}")
        self.ready = True

    def predict(self, test_batch):
        self.train_if_needed()
        # Unpack ((df,), batch_id) -> df
        if isinstance(test_batch, (tuple, list)) and len(test_batch) == 1:
            test_batch = test_batch[0]

        df = test_batch
        # Convert Polars to Pandas if needed
        if pl is not None and hasattr(pl, 'DataFrame') and isinstance(df, pl.DataFrame):
            df = df.to_pandas()
        # Ensure all expected features exist (fill missing with zeros)
        missing = [c for c in self.features if c not in df.columns]
        for c in missing:
            df[c] = 0.0

        X = df[self.features].astype('float64')
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        Xs = self.scaler.transform(X)
        excess = self.model.predict(Xs)
        positions = np.clip(1.0 + excess * (self.k / 1.0), 0.0, 2.0)
        # Return a scalar or a single-value array per batch
        return float(positions[0])


if __name__ == '__main__':
    print("[START] Kaggle Hull Tactical Submission - EXP-004 k=500")
    print(f"[INFO] Current directory: {os.getcwd()}")
    print(f"[INFO] Directory contents: {os.listdir('.')}")

    # Check if we're in competition rerun mode
    is_rerun = os.getenv('KAGGLE_IS_COMPETITION_RERUN')
    print(f"[INFO] KAGGLE_IS_COMPETITION_RERUN: {is_rerun}")

    if is_rerun:
        # In Kaggle rerun (host evaluation), only start the server and block.
        print("[INFO] Running in RERUN mode - starting server only")
        srv = LassoTop20Server()
        srv.serve()
    else:
        # Local/Kaggle notebook execution: start server and run the gateway to write submission.parquet
        print("[INFO] Running in NOTEBOOK mode - generating submission.parquet")
        srv = LassoTop20Server()
        srv.server.start()
        try:
            # Pick competition data directory that actually contains test.csv
            candidates = [
                '/kaggle/input/hull-tactical-market-prediction',  # Kaggle Code environment
                '.',  # working dir
                'data',  # local repo
            ]
            comp_dir = None
            print("[INFO] Searching for test.csv...")
            for p in candidates:
                test_path = os.path.join(p, 'test.csv')
                print(f"[INFO] Checking: {test_path}")
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

            # Check if submission was created
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
