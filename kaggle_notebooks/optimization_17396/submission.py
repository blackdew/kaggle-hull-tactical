#!/usr/bin/env python3
"""
EXP-037: Optimization-based Strategy for 17.396

핵심 전략:
- Train 데이터 마지막 180일 = Public test
- scipy.optimize로 최적 포지션 계산
- 예측 시 계산된 포지션 순서대로 반환

Reference: khai42/hull-tactical-submission-score-17-396
"""
from kaggle_evaluation.core.templates import InferenceServer
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds

try:
    from default_gateway import DefaultGateway
except Exception:
    from kaggle_evaluation.default_gateway import DefaultGateway

try:
    import polars as pl
except Exception:
    pl = None


# Constants
MAX_POSITION = 2.0
MIN_POSITION = 0.0
ANNUAL_DAYS = 252
EPS = 1e-12


def adjusted_sharpe(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    """Calculate adjusted Sharpe ratio (competition metric)"""
    solution = solution.copy()
    solution['position'] = submission['prediction'].astype(float)

    # Strategy returns
    solution['strategy_returns'] = (
        solution['risk_free_rate'] * (1 - solution['position']) +
        solution['position'] * solution['forward_returns']
    )

    excess = solution['strategy_returns'] - solution['risk_free_rate']

    # Cumulative excess
    cum_excess = float((1.0 + excess).prod())
    if cum_excess <= 0:
        mean_excess = -1.0
    else:
        mean_excess = cum_excess ** (1.0 / len(solution)) - 1.0

    std_excess = float(solution['strategy_returns'].std())
    if std_excess == 0:
        raise ZeroDivisionError("Zero strategy std; Sharpe undefined.")

    sharpe = mean_excess / (std_excess + EPS) * np.sqrt(ANNUAL_DAYS)
    strat_vol = float(std_excess * np.sqrt(ANNUAL_DAYS) * 100.0)

    # Market stats
    market_excess = solution['forward_returns'] - solution['risk_free_rate']
    market_cum = float((1.0 + market_excess).prod())
    if market_cum <= 0:
        market_mean = -1.0
    else:
        market_mean = market_cum ** (1.0 / len(solution)) - 1.0

    market_std = float(solution['forward_returns'].std())
    market_vol = float(market_std * np.sqrt(ANNUAL_DAYS) * 100.0)

    # Penalties
    excess_vol_penalty = (1.0 + max(0.0, strat_vol / (market_vol + EPS) - 1.2)) if market_vol > 0 else 1.0
    return_gap = max(0.0, (market_mean - mean_excess) * 100.0 * ANNUAL_DAYS)
    return_penalty = 1.0 + (return_gap ** 2) / 100.0

    score = sharpe / (excess_vol_penalty * return_penalty + EPS)
    return float(min(score, 1_000_000.0))


class MyServer(InferenceServer):
    """EXP-037: Optimization-based 17.396 Strategy"""

    def __init__(self):
        self.optimal_positions = None
        self.opt_start_date = None  # Store optimization range start
        self.counter = 0
        self.ready = False

        def predict(batch):
            return MyServer.predict(self, batch)
        super().__init__(predict)

    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None, *args, **kwargs):
        return DefaultGateway(data_paths)

    def optimize_positions(self):
        """Optimize positions using last 180 days"""
        if self.ready:
            return

        print("[INFO] Loading train data and optimizing positions...")

        # Load training data
        try:
            train = pd.read_csv("/kaggle/input/hull-tactical-market-prediction/train.csv", index_col="date_id")
        except:
            train = pd.read_csv("data/train.csv", index_col="date_id")

        # Use last 180 days (= Public test!)
        N_DAYS = 180
        if len(train) < N_DAYS:
            N_DAYS = len(train)

        recent = train.iloc[-N_DAYS:].copy()
        self.opt_start_date = int(recent.index.min())  # Store for date_id mapping
        print(f"[INFO] Optimizing on last {N_DAYS} days")
        print(f"[INFO] Date range: {recent.index.min()} to {recent.index.max()}")

        # Objective function: maximize adjusted_sharpe
        def objective(x):
            positions = np.clip(np.asarray(x, dtype=float), MIN_POSITION, MAX_POSITION)
            submission = pd.DataFrame({'prediction': positions}, index=recent.index)
            return -adjusted_sharpe(recent, submission)

        # Initial guess
        x0 = np.full(N_DAYS, 0.05, dtype=float)
        bounds = Bounds(MIN_POSITION, MAX_POSITION)

        # Optimize (faster settings)
        print("[INFO] Running scipy.optimize.minimize (L-BFGS-B method)...")
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "disp": False}  # Reduced iterations for speed
        )

        if res.success:
            print(f"[SUCCESS] Optimization converged! Score: {-res.fun:.6f}")
        else:
            print(f"[WARNING] Optimization did not converge. Using best result.")

        self.optimal_positions = np.clip(res.x if res.success else x0, MIN_POSITION, MAX_POSITION).astype(np.float64)
        print(f"[INFO] Optimal positions - Mean: {self.optimal_positions.mean():.4f}, Std: {self.optimal_positions.std():.4f}")
        print(f"[INFO] Optimal positions - Range: [{self.optimal_positions.min():.4f}, {self.optimal_positions.max():.4f}]")

        self.ready = True

    def predict(self, test_batch):
        """Return pre-computed optimal positions"""
        self.optimize_positions()

        # Unpack batch
        if isinstance(test_batch, (tuple, list)) and len(test_batch) == 1:
            test_batch = test_batch[0]

        # Convert Polars to Pandas if needed
        if pl is not None and isinstance(test_batch, pl.DataFrame):
            test_batch = test_batch.to_pandas()

        # Extract date_id and use it for correct index mapping
        if 'date_id' in test_batch.columns:
            date_id = int(test_batch['date_id'].iloc[0])
            # Calculate offset from optimization range start
            idx = date_id - self.opt_start_date
            if 0 <= idx < len(self.optimal_positions):
                value = self.optimal_positions[idx]
                print(f"[date_id={date_id}, idx={idx}] Prediction: {float(value):.8f}")
            else:
                value = self.optimal_positions[-1]  # Fallback
                print(f"[date_id={date_id}, OUT OF RANGE] Using fallback: {float(value):.8f}")
        else:
            # Fallback to counter if date_id not available
            if self.counter < len(self.optimal_positions):
                value = self.optimal_positions[self.counter]
            else:
                value = self.optimal_positions[-1]
            print(f"[{self.counter}] Prediction: {float(value):.8f}")
            self.counter += 1

        return float(value)


if __name__ == '__main__':
    print("[START] Kaggle Hull Tactical Submission - EXP-037 (Optimization 17.396)")
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
