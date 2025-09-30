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
        super().__init__(self.predict)

    # Required by InferenceServer: return a gateway instance for local tests
    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None, *args, **kwargs):
        return DefaultGateway(data_paths)

    def train_if_needed(self):
        if self.ready:
            return
        train = pd.read_csv('train.csv')
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

        X = train[top20].copy()
        X = X.fillna(X.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
        self.scaler = self.StandardScaler()
        Xs = self.scaler.fit_transform(X)
        y = train[target].to_numpy()

        self.model = self.Lasso(alpha=1e-4, max_iter=50000)
        self.model.fit(Xs, y)
        self.features = top20
        self.k = 50.0
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
        X = df[self.features].copy()
        X = X.fillna(X.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
        Xs = self.scaler.transform(X)
        excess = self.model.predict(Xs)
        positions = np.clip(1.0 + excess * (self.k / 1.0), 0.0, 2.0)
        # Return a scalar or a single-value array per batch
        return float(positions.iloc[0] if hasattr(positions, 'iloc') else positions[0])


if __name__ == '__main__':
    # Start server and run the competition gateway so that
    # submission.parquet is produced in the Outputs for this Version.
    srv = LassoTop20Server()
    srv.server.start()
    try:
        DefaultGateway(data_paths=('.',)).run()
    finally:
        srv.server.stop(0)
