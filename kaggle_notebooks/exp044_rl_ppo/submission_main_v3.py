
import os
import sys
import base64
import io
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.distributions import Normal
from collections import deque
from kaggle_evaluation.core.templates import InferenceServer
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    from default_gateway import DefaultGateway
except Exception:
    from kaggle_evaluation.default_gateway import DefaultGateway

try:
    import polars as pl
except Exception:
    pl = None

# Import models
try:
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
except ImportError:
    pass

# ------------------------------------------------------------------------
# RL Model Definition (LSTM)
# ------------------------------------------------------------------------
class ActorCriticLSTM(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super(ActorCriticLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError
    
    def get_features(self, x):
        self.lstm.flatten_parameters()
        out, (h_n, c_n) = self.lstm(x)
        return out[:, -1, :]

    def act(self, state):
        features = self.get_features(state)
        action_mean = self.actor(features)
        return action_mean

def load_rl_model():
    decoded = base64.b64decode(BASE64_WEIGHTS)
    buffer = io.BytesIO(decoded)
    state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    model = ActorCriticLSTM(21, 1)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ------------------------------------------------------------------------
# Inference Server (Hybrid)
# ------------------------------------------------------------------------
class MyServer(InferenceServer):
    def __init__(self):
        def predict(batch):
            return MyServer.predict(self, batch)
        super().__init__(predict)
        
        # --- RL Init ---
        self.rl_model = load_rl_model()
        self.current_position = 0.0
        self.window_size = 10
        self.n_rl_features = 20
        self.history = deque(maxlen=self.window_size)
        self.rl_features = [
            'M4', 'V13', 'M1', 'S5', 'S2', 'D1', 'D2', 'M2', 'V10', 'E7',
            'P7', 'P2', 'E1', 'V6', 'V1', 'E16', 'E2', 'P4', 'V5', 'V4'
        ]
        
        # --- Supervised Init ---
        from sklearn.preprocessing import StandardScaler
        self.StandardScaler = StandardScaler
        self.sup_ready = False
        self.models = {}
        self.scaler = None
        self.K = 250
        self.refined_features = [
            'V7/P8', 'V7/P7', 'E19', 'V7*P5', 'V7/S5', 'S8', 'V7*P8', 'P5/P7', 'P8/S2', 'M4*E19',
            'P8*E19', 'V7*E19', 'E19/S5', 'P8*S2', 'E19/P7', 'M4/P8', 'V7*P7', 'V13/S5', 'P8/P7', 'M4/E19',
            'V13²', 'V13*V7', 'I2*E19', 'V7*S2', 'P8²', 'V13/S2', 'P8/P5', 'M4*P8', 'S5/P7', 'M4/I2',
            'V7/S2', 'V7*I2', 'V13/P7', 'P5', 'S2/P5'
        ]
        self.top_20 = [
            'M4', 'V13', 'V7', 'P8', 'S2', 'I2', 'E19', 'S5', 'P5', 'P7',
            'M2', 'V9', 'M3', 'P12', 'P10', 'V10', 'E12', 'P11', 'M12', 'S8'
        ]

    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None, *args, **kwargs):
        return DefaultGateway(data_paths)

    # --- Supervised Methods ---
    def train_if_needed(self):
        if self.sup_ready: return
        
        try:
            train = pd.read_csv("/kaggle/input/hull-tactical-market-prediction/train.csv")
        except:
            train = pd.read_csv("data/train.csv")
        y = train['market_forward_excess_returns'].values
        X = self.create_features(train)
        self.scaler = self.StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        quantiles = [0.1, 0.5, 0.9]
        q_names = ['q10', 'q50', 'q90']
        model_types = ['xgb', 'lgb', 'cat']
        
        for m_type in model_types:
            self.models[m_type] = {}
            for q, q_name in zip(quantiles, q_names):
                if m_type == 'xgb':
                    model = XGBRegressor(n_estimators=150, max_depth=7, learning_rate=0.025, subsample=1.0, colsample_bytree=0.6, reg_lambda=0.5, objective='reg:quantileerror', quantile_alpha=q, random_state=42, n_jobs=-1)
                elif m_type == 'lgb':
                    model = LGBMRegressor(n_estimators=150, max_depth=7, learning_rate=0.025, subsample=0.8, colsample_bytree=0.6, reg_lambda=0.5, objective='quantile', alpha=q, random_state=42, n_jobs=-1, verbose=-1)
                elif m_type == 'cat':
                    model = CatBoostRegressor(iterations=150, depth=7, learning_rate=0.025, loss_function=f'Quantile:alpha={q}', random_state=42, verbose=0, thread_count=-1)
                model.fit(X_scaled, y)
                self.models[m_type][q_name] = model
        self.sup_ready = True

    def create_features(self, df):
        if pl is not None and isinstance(df, pl.DataFrame): df = df.to_pandas()
        for feat in self.top_20:
            if feat not in df.columns: df[feat] = 0.0
        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        X_base = df[self.top_20]
        interactions = []
        feature_names = self.top_20.copy()
        top_10 = self.top_20[:10]
        eps = 1e-8
        for i, feat1 in enumerate(top_10):
            for feat2 in top_10[i + 1 :]:
                interactions.append(X_base[feat1] * X_base[feat2])
                feature_names.append(f'{feat1}*{feat2}')
        for i, feat1 in enumerate(top_10):
            for feat2 in top_10[i + 1 :]:
                interactions.append(X_base[feat1] / (X_base[feat2].abs() + eps))
                feature_names.append(f'{feat1}/{feat2}')
        top_5 = self.top_20[:5]
        for feat in top_5:
            interactions.append(X_base[feat] ** 2)
            feature_names.append(f'{feat}²')
            interactions.append(X_base[feat] ** 3)
            feature_names.append(f'{feat}³')
        X_all = pd.concat([X_base] + [pd.Series(feat, name=name) for feat, name in zip(interactions, feature_names[len(self.top_20) :])], axis=1)
        X_all.columns = feature_names
        X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)
        X = X_all[self.refined_features]
        return X

    def predict(self, test_batch):
        # 1. Supervised Prediction
        self.train_if_needed()
        if isinstance(test_batch, (tuple, list)) and len(test_batch) == 1: test_batch = test_batch[0]
        df = test_batch
        if pl is not None and hasattr(pl, 'DataFrame') and isinstance(df, pl.DataFrame): df = df.to_pandas()
        
        # Supervised Features
        X_sup = self.create_features(df)
        missing = [f for f in self.refined_features if f not in X_sup.columns]
        for f in missing: X_sup[f] = 0.0
        X_sup = X_sup[self.refined_features].astype('float64').replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_scaled = self.scaler.transform(X_sup)
        
        preds = {'q10': 0.0, 'q50': 0.0, 'q90': 0.0}
        for q_name in ['q10', 'q50', 'q90']:
            p_xgb = self.models['xgb'][q_name].predict(X_scaled)[0]
            p_lgb = self.models['lgb'][q_name].predict(X_scaled)[0]
            p_cat = self.models['cat'][q_name].predict(X_scaled)[0]
            preds[q_name] = (p_xgb + p_lgb + p_cat) / 3.0
            
        ci_width = preds['q90'] - preds['q10']
        confidence = 1.0 / (np.abs(ci_width) + 0.001)
        confidence = np.clip(confidence, 0.5, 5.0)
        pos_sup = 1.0 + preds['q50'] * self.K * confidence
        pos_sup = np.clip(pos_sup, 0.0, 2.0)
        
        # 2. RL Prediction
        x_rl = df[self.rl_features].fillna(0).values
        feat = x_rl[0] # Assume batch size 1
        self.history.append(feat)
        current_hist = list(self.history)
        if len(current_hist) < self.window_size:
            padding = [np.zeros(self.n_rl_features) for _ in range(self.window_size - len(current_hist))]
            current_hist = padding + current_hist
        window_feat = np.array(current_hist)
        pos_col = np.full((self.window_size, 1), self.current_position)
        obs = np.hstack([window_feat, pos_col])
        state_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = self.rl_model.act(state_tensor)
            pos_rl_raw = float(action.numpy()[0][0])
        
        # Shift RL to [0, 2] range (Assuming -1 -> 0, 1 -> 2)
        pos_rl = np.clip(pos_rl_raw + 1.0, 0.0, 2.0)
        
        # 3. Ensemble
        final_pos = 0.5 * pos_sup + 0.5 * pos_rl
        
        # Update RL state (using raw RL output or final pos? Usually final pos is what we took)
        # But RL agent expects its own action as state?
        # If we override its action, we should update state with what we actually did?
        # Or what it *wanted* to do?
        # For simplicity, let's update with what it *wanted* to do (pos_rl_raw) to keep its internal consistency?
        # No, 'current_position' in Env is the *actual* position held.
        # If we submit 'final_pos', that's our position.
        # So we should convert final_pos back to [-1, 1] scale for the RL agent's state?
        # final_pos is [0, 2]. RL state expects [-1, 1].
        # So state_pos = final_pos - 1.0.
        self.current_position = final_pos - 1.0
        
        return float(final_pos)

if __name__ == '__main__':
    print("[START] Kaggle Hull Tactical Submission - EXP-044 v3 (Hybrid Ensemble)")
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        print("[INFO] Running in COMPETITION RERUN mode - starting server")
        server = MyServer()
        server.serve()
    else:
        print("[INFO] Running in NOTEBOOK mode - generating submission.parquet")
        server = MyServer()
        # Start the server in a separate thread/process if needed, or just use the gateway which calls the server?
        # In EXP-040, it did: srv.server.start(); DefaultGateway().run(); srv.server.stop(0)
        # But MyServer inherits from InferenceServer.
        # Let's check EXP-040 again.
        # It calls srv.server.start() (which is likely the gRPC server)
        # Then DefaultGateway().run() which connects to it?
        # Actually, InferenceServer wraps the user code.
        # Let's copy the logic exactly from EXP-040.
        
        server.server.start()
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
                
            if os.path.exists('submission.parquet'):
                print("[SUCCESS] submission.parquet created!")
            else:
                print("[ERROR] submission.parquet not created!")
        finally:
            server.server.stop(0)
