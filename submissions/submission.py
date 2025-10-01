from kaggle_evaluation.core.templates import InferenceServer
import os
import numpy as np
import pandas as pd

# Gateway import (host may expose either path)
try:
    from default_gateway import DefaultGateway  # type: ignore
except Exception:
    from kaggle_evaluation.default_gateway import DefaultGateway  # type: ignore

# Optional Polars support (Gateway often passes pl.DataFrame)
try:
    import polars as pl  # type: ignore
except Exception:
    pl = None

class LassoTop20Server(InferenceServer):
    def __init__(self):
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import StandardScaler
        self.Lasso = Lasso
        self.StandardScaler = StandardScaler
        self.ready = False
        # Register an endpoint function literally named 'predict'
        def predict(batch):  # noqa: D401
            return LassoTop20Server.predict(self, batch)
        super().__init__(predict)

    # 이 메서드를 반드시 구현해야 합니다
    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None, *args, **kwargs):
        return DefaultGateway(data_paths)
    
    def train_if_needed(self):
        if self.ready:
            return
        # Robust train.csv loader (works in Kaggle + local)
        def _load_train() -> pd.DataFrame:
            candidates = [
                'train.csv',
                './train.csv',
                '/kaggle/input/hull-tactical-market-prediction/train.csv',
                '/kaggle/working/train.csv',
                'data/train.csv',
            ]
            for path in candidates:
                if os.path.exists(path):
                    return pd.read_csv(path)
            raise FileNotFoundError('train.csv not found in known locations: ' + ', '.join(candidates))

        train = _load_train()
        target = 'market_forward_excess_returns'
        exclude = {
          'date_id','forward_returns','risk_free_rate',
          'market_forward_excess_returns','is_scored',
          'lagged_forward_returns','lagged_risk_free_rate',
          'lagged_market_forward_excess_returns'
        }
        feats = [c for c in train.columns if c not in exclude]
        num = train[feats + [target]].select_dtypes(include=[np.number])
        corr = num.corr(numeric_only=True)[target].drop(index=target).abs().sort_values(ascending=False)
        top20 = [c for c in corr.index[:20] if c in feats]
        
        X = train[top20].copy()
        X = X.fillna(X.median(numeric_only=True)).replace([np.inf,-np.inf], np.nan).fillna(0)
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
        # Ensure all expected features exist in this batch
        missing = [c for c in self.features if c not in df.columns]
        for c in missing:
            df[c] = 0.0

        # Numeric casting and sanitization
        X = df[self.features].astype('float64')
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Transform with scaler (preserve feature names to avoid warnings)
        Xs = self.scaler.transform(X)
        excess = self.model.predict(Xs)
        pos = np.clip(1.0 + excess * self.k, 0.0, 2.0)
        return float(pos[0])

if __name__ == '__main__':
    # Kaggle 주최측 재실행 환경에서는 서버만 띄워야 합니다(게이트웨이는 호스트가 구동).
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        srv = LassoTop20Server()
        srv.serve()
    else:
        # 로컬/커밋 실행에서 Outputs(submission.parquet)를 생성하기 위한 경로
        srv = LassoTop20Server()
        srv.server.start()
        try:
            # 동작 환경별 데이터 경로 선택
            candidates = [
                '/kaggle/input/hull-tactical-market-prediction',  # Kaggle Code env
                '.',                                             # bundle root / working
                'data',                                          # local repo
            ]
            comp_dir = None
            for p in candidates:
                if os.path.exists(os.path.join(p, 'test.csv')):
                    comp_dir = p
                    break

            if comp_dir is None:
                DefaultGateway().run()
            else:
                DefaultGateway(data_paths=(comp_dir,)).run()

            # Fallback: submission.parquet이 없으면 직접 생성
            if not os.path.exists('submission.parquet'):
                import pandas as pd
                test_path = os.path.join(comp_dir or '.', 'test.csv')
                test_df = pd.read_csv(test_path)
                missing = [c for c in srv.features if c not in test_df.columns]
                for c in missing:
                    test_df[c] = 0.0
                X = test_df[srv.features].astype('float64').replace([np.inf, -np.inf], np.nan).fillna(0.0)
                Xs = srv.scaler.transform(X)
                excess = srv.model.predict(Xs)
                pred = np.clip(1.0 + excess * srv.k, 0.0, 2.0)
                row_id = test_df.columns[0]
                pd.DataFrame({row_id: test_df[row_id], 'prediction': pred}).to_parquet('submission.parquet', index=False)
        finally:
            srv.server.stop(0)
