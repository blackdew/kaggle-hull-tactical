
import os
import sys
import base64
import io
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.distributions import Normal

# ------------------------------------------------------------------------
# Model Definition (Must match training)
# ------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_mean = self.actor(state)
        return action_mean # Deterministic action for inference

# ------------------------------------------------------------------------
# Load Model
# ------------------------------------------------------------------------
def load_model():
    # Decode weights
    decoded = base64.b64decode(BASE64_WEIGHTS)
    buffer = io.BytesIO(decoded)
    state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    
    # Initialize model
    # 20 features + 1 position = 21 inputs
    model = ActorCritic(21, 1)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ------------------------------------------------------------------------
# Inference Server
# ------------------------------------------------------------------------
from kaggle_evaluation.core.templates import InferenceServer

try:
    from default_gateway import DefaultGateway
except Exception:
    from kaggle_evaluation.default_gateway import DefaultGateway

class MyServer(InferenceServer):
    def __init__(self):
        def predict(batch):
            return MyServer.predict(self, batch)
        super().__init__(predict)
        self.model = load_model()
        self.current_position = 0.0
        self.features = [
            'M4', 'V13', 'M1', 'S5', 'S2', 'D1', 'D2', 'M2', 'V10', 'E7',
            'P7', 'P2', 'E1', 'V6', 'V1', 'E16', 'E2', 'P4', 'V5', 'V4'
        ]

    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None, *args, **kwargs):
        return DefaultGateway(data_paths)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        # Extract features
        x = data[self.features].fillna(0).values
        
        # Add current position to state
        # Note: data is a batch, but for this competition it's usually 1 row per step in loop?
        # The InferenceServer passes a batch. We need to handle batch size.
        # But we also need to maintain state (current_position).
        # If batch size > 1, this stateful logic is tricky.
        # Assuming sequential processing or batch=1 for stateful RL.
        # However, the competition might pass a batch of assets for the SAME time step?
        # No, it's a single asset. So batch is time steps?
        # If batch is time steps, we can't easily do stateful updates inside the batch without a loop.
        
        predictions = []
        for i in range(len(x)):
            feat = x[i]
            state = np.append(feat, self.current_position)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
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
