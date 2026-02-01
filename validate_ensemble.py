import pandas as pd
import numpy as np
# Import LightGBM before Torch to avoid OpenMP conflict on macOS
from lightgbm import LGBMRegressor
import torch
import torch.nn as nn
import base64
import io
from collections import deque
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings('ignore')

# Load Base64 Weights (from file)
with open("experiments/044_rl_ppo/weights_v2_b64.txt", "r") as f:
    content = f.read().strip()
    # Extract the string inside quotes if present
    if "=" in content and "BASE64_WEIGHTS" in content:
        content = content.split('=')[1].strip().strip("'").strip('"')
    BASE64_WEIGHTS = content

# --- RL Model ---
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

# --- Validation Logic ---
def validate():
    print("Loading Data...")
    df = pd.read_csv("data/train.csv")
    
    # Split Validation (Last 20%)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # --- 1. Supervised Model Training ---
    print("Training Supervised Models...")
    # Features
    top_20 = [
        'M4', 'V13', 'V7', 'P8', 'S2', 'I2', 'E19', 'S5', 'P5', 'P7',
        'M2', 'V9', 'M3', 'P12', 'P10', 'V10', 'E12', 'P11', 'M12', 'S8'
    ]
    refined_features = [
        'V7/P8', 'V7/P7', 'E19', 'V7*P5', 'V7/S5', 'S8', 'V7*P8', 'P5/P7', 'P8/S2', 'M4*E19',
        'P8*E19', 'V7*E19', 'E19/S5', 'P8*S2', 'E19/P7', 'M4/P8', 'V7*P7', 'V13/S5', 'P8/P7', 'M4/E19',
        'V13²', 'V13*V7', 'I2*E19', 'V7*S2', 'P8²', 'V13/S2', 'P8/P5', 'M4*P8', 'S5/P7', 'M4/I2',
        'V7/S2', 'V7*I2', 'V13/P7', 'P5', 'S2/P5'
    ]
    
    def create_features(df):
        df = df.copy()
        for feat in top_20:
            if feat not in df.columns: df[feat] = 0.0
        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        X_base = df[top_20]
        interactions = []
        feature_names = top_20.copy()
        top_10 = top_20[:10]
        eps = 1e-8
        for i, feat1 in enumerate(top_10):
            for feat2 in top_10[i + 1 :]:
                interactions.append(X_base[feat1] * X_base[feat2])
                feature_names.append(f'{feat1}*{feat2}')
        for i, feat1 in enumerate(top_10):
            for feat2 in top_10[i + 1 :]:
                interactions.append(X_base[feat1] / (X_base[feat2].abs() + eps))
                feature_names.append(f'{feat1}/{feat2}')
        top_5 = top_20[:5]
        for feat in top_5:
            interactions.append(X_base[feat] ** 2)
            feature_names.append(f'{feat}²')
            interactions.append(X_base[feat] ** 3)
            feature_names.append(f'{feat}³')
        X_all = pd.concat([X_base] + [pd.Series(feat, name=name) for feat, name in zip(interactions, feature_names[len(top_20) :])], axis=1)
        X_all.columns = feature_names
        X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)
        X = X_all[refined_features]
        return X

    X_train = create_features(train_df)
    y_train = train_df['market_forward_excess_returns'].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    models = {'xgb': {}, 'lgb': {}, 'cat': {}}
    quantiles = [0.1, 0.5, 0.9]
    q_names = ['q10', 'q50', 'q90']
    
    for m_type in ['xgb', 'lgb', 'cat']:
        for q, q_name in zip(quantiles, q_names):
            if m_type == 'xgb':
                model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, objective='reg:quantileerror', quantile_alpha=q, n_jobs=-1)
            elif m_type == 'lgb':
                model = LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, objective='quantile', alpha=q, n_jobs=-1, verbose=-1)
            elif m_type == 'cat':
                model = CatBoostRegressor(iterations=100, depth=5, learning_rate=0.05, loss_function=f'Quantile:alpha={q}', verbose=0, thread_count=-1)
            model.fit(X_train_scaled, y_train)
            models[m_type][q_name] = model
            
    # --- 2. RL Model Init ---
    print("Initializing RL Model...")
    rl_model = load_rl_model()
    window_size = 10
    n_rl_features = 20
    rl_features = [
        'M4', 'V13', 'M1', 'S5', 'S2', 'D1', 'D2', 'M2', 'V10', 'E7',
        'P7', 'P2', 'E1', 'V6', 'V1', 'E16', 'E2', 'P4', 'V5', 'V4'
    ]
    history = deque(maxlen=window_size)
    current_position = 0.0
    
    # --- 3. Validation Loop ---
    print("Running Validation...")
    X_val_sup = create_features(val_df)
    X_val_sup_scaled = scaler.transform(X_val_sup)
    
    # Batch prediction for Supervised
    preds_sup = {'q10': np.zeros(len(val_df)), 'q50': np.zeros(len(val_df)), 'q90': np.zeros(len(val_df))}
    for q_name in q_names:
        p_xgb = models['xgb'][q_name].predict(X_val_sup_scaled)
        p_lgb = models['lgb'][q_name].predict(X_val_sup_scaled)
        p_cat = models['cat'][q_name].predict(X_val_sup_scaled)
        preds_sup[q_name] = (p_xgb + p_lgb + p_cat) / 3.0
        
    # Calculate Supervised Positions
    K = 250
    ci_width = preds_sup['q90'] - preds_sup['q10']
    confidence = 1.0 / (np.abs(ci_width) + 0.001)
    confidence = np.clip(confidence, 0.5, 5.0)
    pos_sup_all = 1.0 + preds_sup['q50'] * K * confidence
    pos_sup_all = np.clip(pos_sup_all, 0.0, 2.0)
    
    # RL Loop (Sequential)
    pos_rl_all = []
    val_rl_data = val_df[rl_features].fillna(0).values
    
    # Pre-fill history with end of train data to avoid cold start
    train_rl_data = train_df[rl_features].fillna(0).values
    for i in range(max(0, len(train_rl_data) - window_size), len(train_rl_data)):
        history.append(train_rl_data[i])
        
    for i in range(len(val_rl_data)):
        feat = val_rl_data[i]
        history.append(feat)
        
        current_hist = list(history)
        if len(current_hist) < window_size:
            padding = [np.zeros(n_rl_features) for _ in range(window_size - len(current_hist))]
            current_hist = padding + current_hist
            
        window_feat = np.array(current_hist)
        pos_col = np.full((window_size, 1), current_position)
        obs = np.hstack([window_feat, pos_col])
        state_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            action = rl_model.act(state_tensor)
            pos_rl_raw = float(action.numpy()[0][0])
            
        pos_rl = np.clip(pos_rl_raw + 1.0, 0.0, 2.0)
        pos_rl_all.append(pos_rl)
        
        # Ensemble for state update
        final_pos = 0.5 * pos_sup_all[i] + 0.5 * pos_rl
        current_position = final_pos - 1.0
        
    pos_rl_all = np.array(pos_rl_all)
    final_positions = 0.5 * pos_sup_all + 0.5 * pos_rl_all
    
    # Calculate Sharpe
    returns = val_df['forward_returns'].values
    # Note: 'forward_returns' might be different from what we want?
    # Usually we use 'market_forward_excess_returns' or similar for target, but for Sharpe we use actual returns.
    # Let's use 'forward_returns' as in previous scripts.
    
    # Shift positions? No, prediction at t is for return at t (forward).
    # So Reward = Position * Return
    # But wait, final_pos is [0, 2]. Market neutral is 1.
    # So effective position is final_pos - 1.
    effective_positions = final_positions - 1.0
    
    daily_rewards = effective_positions * returns
    sharpe = np.mean(daily_rewards) / (np.std(daily_rewards) + 1e-9) * np.sqrt(252)
    
    print(f"Validation Sharpe: {sharpe:.4f}")

if __name__ == "__main__":
    validate()
