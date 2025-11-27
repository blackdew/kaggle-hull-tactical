
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

try:
    from default_gateway import DefaultGateway
except Exception:
    from kaggle_evaluation.default_gateway import DefaultGateway

# ------------------------------------------------------------------------
# Model Definition (LSTM)
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

# ------------------------------------------------------------------------
# Load Model
# ------------------------------------------------------------------------
def load_model():
    decoded = base64.b64decode(BASE64_WEIGHTS)
    buffer = io.BytesIO(decoded)
    state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    
    # Input dim = 20 features + 1 position = 21
    model = ActorCriticLSTM(21, 1)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ------------------------------------------------------------------------
# Inference Server
# ------------------------------------------------------------------------
class MyServer(InferenceServer):
    def __init__(self):
        def predict(batch):
            return MyServer.predict(self, batch)
        super().__init__(predict)
        
        self.model = load_model()
        self.current_position = 0.0
        self.window_size = 10
        self.n_features = 20
        
        # History buffer: Store last N features
        # We need to store features ONLY, not position (position is appended dynamically)
        self.history = deque(maxlen=self.window_size)
        
        self.features = [
            'M4', 'V13', 'M1', 'S5', 'S2', 'D1', 'D2', 'M2', 'V10', 'E7',
            'P7', 'P2', 'E1', 'V6', 'V1', 'E16', 'E2', 'P4', 'V5', 'V4'
        ]

    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None, *args, **kwargs):
        return DefaultGateway(data_paths)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        # Extract features
        x = data[self.features].fillna(0).values
        
        predictions = []
        for i in range(len(x)):
            feat = x[i] # (F,)
            
            # Update history
            self.history.append(feat)
            
            # Construct Window
            # If history < window_size, pad with zeros (or repeat first?)
            # Padding with zeros is safer for cold start.
            current_hist = list(self.history)
            if len(current_hist) < self.window_size:
                # Pad with zeros at the beginning
                padding = [np.zeros(self.n_features) for _ in range(self.window_size - len(current_hist))]
                current_hist = padding + current_hist
            
            window_feat = np.array(current_hist) # (W, F)
            
            # Append current position to each step in window?
            # In training, we did: pos_col = np.full((window, 1), current_position)
            pos_col = np.full((self.window_size, 1), self.current_position)
            
            obs = np.hstack([window_feat, pos_col]) # (W, F+1)
            
            # Prepare tensor
            state_tensor = torch.FloatTensor(obs).unsqueeze(0) # (1, W, F+1)
            
            with torch.no_grad():
                action = self.model.act(state_tensor)
                position = float(action.numpy()[0][0])
            
            # Clip position
            position = np.clip(position, -1.0, 1.0)
            
            # Update state
            self.current_position = position
            predictions.append(position)
            
        return pd.DataFrame({'prediction': predictions}, index=data.index)

if __name__ == '__main__':
    server = MyServer()
    server.serve()
