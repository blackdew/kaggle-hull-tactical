#!/usr/bin/env python3
"""
Test: Constant Position = 2.0 (Maximum Leverage)
Hypothesis: 17.396 might be from always betting maximum
"""
from kaggle_evaluation.core.templates import InferenceServer
import os
import pandas as pd

try:
    from default_gateway import DefaultGateway
except:
    from kaggle_evaluation.default_gateway import DefaultGateway

try:
    import polars as pl
except:
    pl = None


class MyServer(InferenceServer):
    """Test: Always return position = 2.0"""

    def __init__(self):
        def predict(batch):
            return MyServer.predict(self, batch)
        super().__init__(predict)

    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None, *args, **kwargs):
        return DefaultGateway(data_paths)

    def predict(self, test_batch):
        """Always return 2.0 (maximum position)"""
        return 2.0


if __name__ == '__main__':
    print("[TEST] Constant Position = 2.0")

    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        srv = MyServer()
        srv.serve()
    else:
        srv = MyServer()
        srv.server.start()
        try:
            candidates = [
                '/kaggle/input/hull-tactical-market-prediction',
                '.',
                'data',
            ]
            comp_dir = None
            for p in candidates:
                if os.path.exists(os.path.join(p, 'test.csv')):
                    comp_dir = p
                    break

            if comp_dir:
                DefaultGateway(data_paths=(comp_dir,)).run()
            else:
                DefaultGateway().run()

            if os.path.exists('submission.parquet'):
                sub = pd.read_parquet('submission.parquet')
                print(f"[SUCCESS] Shape: {sub.shape}")
                print(f"All predictions = 2.0: {(sub['prediction'] == 2.0).all()}")
        finally:
            srv.server.stop(0)
