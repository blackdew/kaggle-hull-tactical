"""
Kaggle Debug Inference Script — Lasso Top‑20 with verbose logging

목적
- 호스트 재실행(rerun) 환경에서는 서버만 구동(serve)하여 평가 게이트웨이가 접속하도록 함.
- 로컬/노트북 실행 시 게이트웨이를 함께 실행하여 submission.parquet 생성 + 풍부한 로그 출력.

제출 방법(노트북 Script 커널)
- code_file로 본 파일을 사용하고, competition_sources에 `hull-tactical-market-prediction`을 포함하세요.

로그 내용
- 데이터 경로 탐색 결과, 학습/특징 개수, 상위 피처 일부, 스케일 정보, 예측 스칼라/shape, 최종 제출 파일 요약 등.
"""

from __future__ import annotations

import os
import sys
import json
import time
import traceback
from typing import List

import numpy as np
import pandas as pd

try:
    from kaggle_evaluation.core.templates import InferenceServer
    from kaggle_evaluation.default_gateway import DefaultGateway
except Exception:
    # 로컬 백업 경로
    from kaggle_evaluation.core.templates import InferenceServer  # type: ignore
    from kaggle_evaluation.default_gateway import DefaultGateway  # type: ignore

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None


def log(msg: str) -> None:
    print(f"[DEBUG] {time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}", flush=True)


class LassoTop20DebugServer(InferenceServer):
    def __init__(self):
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import StandardScaler

        self.Lasso = Lasso
        self.StandardScaler = StandardScaler
        self.ready = False
        self.features: List[str] = []
        self.k = 50.0
        # Register a named function (not a bound method) so relay accepts it
        def predict(batch):  # noqa: D401
            return LassoTop20DebugServer.predict(self, batch)
        super().__init__(predict)

    # InferenceServer가 요구하는 테스트용 게이트웨이 생성자 구현
    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None, *args, **kwargs):
        return DefaultGateway(data_paths)

    def _load_train(self) -> pd.DataFrame:
        candidates = [
            '/kaggle/input/hull-tactical-market-prediction/train.csv',
            'train.csv', './train.csv', '/kaggle/working/train.csv', 'data/train.csv',
        ]
        for p in candidates:
            if os.path.exists(p):
                log(f"train.csv found at: {p}")
                return pd.read_csv(p)
        raise FileNotFoundError('train.csv not found in known locations')

    def _select_features(self, df: pd.DataFrame, target: str) -> List[str]:
        exclude = {
            'date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns',
            'is_scored', 'lagged_forward_returns', 'lagged_risk_free_rate', 'lagged_market_forward_excess_returns',
        }
        feats = [c for c in df.columns if c not in exclude]
        num = df[feats + [target]].select_dtypes(include=[np.number])
        corr = num.corr(numeric_only=True)[target].drop(index=target).abs().sort_values(ascending=False)
        top20 = [c for c in corr.index[:20] if c in feats]
        return top20

    def train_if_needed(self) -> None:
        if self.ready:
            return
        t0 = time.time()
        log("training start")
        try:
            train = self._load_train()
            log(f"train shape: {train.shape}")
            target = 'market_forward_excess_returns'
            self.features = self._select_features(train, target)
            log(f"features selected: {len(self.features)} -> {self.features[:8]}{'...' if len(self.features)>8 else ''}")

            X = train[self.features].copy()
            X = X.fillna(X.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            self.scaler = self.StandardScaler()
            Xs = self.scaler.fit_transform(X)
            y = train[target].to_numpy()
            log(f"Xs shape: {Xs.shape}, y shape: {y.shape}")

            self.model = self.Lasso(alpha=1e-4, max_iter=50000)
            self.model.fit(Xs, y)
            self.k = 50.0
            self.ready = True
            log(f"training done in {time.time()-t0:.2f}s")
        except Exception:
            log("EXCEPTION during training:\n" + traceback.format_exc())
            raise

    def predict(self, test_batch):
        self.train_if_needed()
        try:
            # Unpack ((df,), batch_id) -> df
            if isinstance(test_batch, (tuple, list)) and len(test_batch) == 1:
                test_batch = test_batch[0]

            df = test_batch
            if pl is not None and hasattr(pl, 'DataFrame') and isinstance(df, pl.DataFrame):
                df = df.to_pandas()

            # Column safety
            missing = [c for c in self.features if c not in df.columns]
            if missing:
                log(f"missing features in batch: {len(missing)} (filled with 0.0), e.g. {missing[:5]}")
                for c in missing:
                    df[c] = 0.0

            X = df[self.features].astype('float64')
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            Xs = self.scaler.transform(X)
            excess = self.model.predict(Xs)
            pos = np.clip(1.0 + excess * self.k, 0.0, 2.0)
            if hasattr(pos, 'shape'):
                log(f"predict: excess shape={getattr(excess, 'shape', None)}, pos[0]={float(pos[0]):.6f}")
            return float(pos[0])
        except Exception:
            log("EXCEPTION during predict:\n" + traceback.format_exc())
            raise


if __name__ == '__main__':
    log("kernel start (debug mode)")
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        # Host rerun: serve only
        srv = LassoTop20DebugServer()
        log("serve() on rerun")
        srv.serve()
    else:
        # Local / Kaggle notebook run: start server and Gateway to produce submission.parquet with logs
        srv = LassoTop20DebugServer()
        srv.server.start()
        try:
            # choose data dir that has test.csv
            candidates = [
                '/kaggle/input/hull-tactical-market-prediction', '.', 'data'
            ]
            comp_dir = None
            for p in candidates:
                if os.path.exists(os.path.join(p, 'test.csv')):
                    comp_dir = p
                    break
            log(f"selected data dir: {comp_dir or '(DefaultGateway defaults)'}")

            if comp_dir is None:
                DefaultGateway().run()
            else:
                DefaultGateway(data_paths=(comp_dir,)).run()

            # Post-run: summarize submission
            sub_path = 'submission.parquet'
            if os.path.exists(sub_path):
                try:
                    import pyarrow as pa  # noqa: F401
                    import pandas as pd
                    df = pd.read_parquet(sub_path)
                    log(f"submission: shape={df.shape}, columns={list(df.columns)}")
                    log(f"submission.head():\n{df.head()}\n...")
                except Exception:
                    log("EXCEPTION while reading submission.parquet:\n" + traceback.format_exc())
            else:
                log("WARNING: submission.parquet not found after gateway run")
        finally:
            srv.server.stop(0)
            log("server stopped")
