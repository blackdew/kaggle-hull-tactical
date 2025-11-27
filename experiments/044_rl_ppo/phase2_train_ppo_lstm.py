#!/usr/bin/env python3
"""
EXP-044 Phase 2: Train PPO Agent (LSTM + Sharpe Reward)
Goal: Train a PPO agent with LSTM to capture temporal patterns and optimize Sharpe Ratio.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os
import sys

print("="*80)
print("EXP-044 Phase 2: Train PPO Agent (LSTM + Sharpe Reward)")
print("="*80)

# Hyperparameters
LR = 0.0001 # Lower learning rate for LSTM
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4
BATCH_SIZE = 64
T_HORIZON = 2048
TOTAL_TIMESTEPS = 100000 # Increased training steps
WINDOW_SIZE = 10 # Lookback window

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load Data
print("\n[1] Loading data...")
train_df = pd.read_csv("data/train.csv")

# Feature Selection (Top 20 from EXP-040)
TOP_FEATURES = [
    'M4', 'V13', 'M1', 'S5', 'S2', 'D1', 'D2', 'M2', 'V10', 'E7',
    'P7', 'P2', 'E1', 'V6', 'V1', 'E16', 'E2', 'P4', 'V5', 'V4'
]
features = train_df[TOP_FEATURES].fillna(0).values
returns = train_df['forward_returns'].fillna(0).values

# Split Train/Val
split_idx = int(len(train_df) * 0.8)
train_features = features[:split_idx]
train_returns = returns[:split_idx]
val_features = features[split_idx:]
val_returns = returns[split_idx:]

print(f"   Train: {len(train_features)}, Val: {len(val_features)}")

# 2. Environment V2
class HullTradingEnvV2:
    def __init__(self, features, returns, window_size=10):
        self.features = features
        self.returns = returns
        self.window_size = window_size
        self.n_features = features.shape[1]
        self.reset()

    def reset(self):
        self.current_step = self.window_size # Start after window
        self.current_position = 0.0
        return self._next_observation()

    def _next_observation(self):
        # Windowed Observation: (Window, Features+1)
        # We append current position to EACH step in window? 
        # Or just append it as a separate scalar?
        # For simplicity in LSTM, let's append it to the features.
        # But position changes every step. Past positions were different.
        # Ideally we should track past positions too.
        # For this version, let's just append the *current* position to the last step's features
        # and maybe 0 or past positions to previous steps?
        # Simpler: Just concatenate position to the features. 
        # Let's assume the agent knows its past positions (it took them).
        # We will append 'current_position' to the last time step features only?
        # No, let's append it to all, effectively telling it "this is the current state".
        # Actually, standard way is: Obs = [Window_Features, Current_Position]
        # But for LSTM input (B, W, F), we need consistent dims.
        # Let's make Obs: (Window, F). And we treat Position as a separate input or append to F.
        # Let's append to F. So F becomes F+1.
        
        start = self.current_step - self.window_size
        end = self.current_step
        
        window_feat = self.features[start:end] # (W, F)
        
        # Append current position as a feature to all timesteps? 
        # Or just the last one?
        # Let's append a column of 'current_position' to the window.
        # This implies "I am currently at pos X".
        pos_col = np.full((self.window_size, 1), self.current_position)
        
        obs = np.hstack([window_feat, pos_col]) # (W, F+1)
        return obs

    def step(self, action):
        target_position = np.clip(action, -1, 1)
        
        # Reward: Sharpe Proxy
        # R = r - lambda * r^2
        # Lambda = 0.1 (Arbitrary penalty factor)
        current_return = self.returns[self.current_step]
        raw_reward = target_position * current_return
        
        # Penalty for high variance (proxy)
        # We want consistent positive rewards.
        # If raw_reward is negative, we punish more?
        # Sharpe Reward usually requires batch calculation.
        # Per-step proxy: Differential Sharpe or just Mean - Variance penalty.
        # Let's use: Reward = Return * 100 - 0.1 * |Action_Change|? No, that's transaction cost.
        # Let's stick to simple Return for now, but maybe scale it better.
        # Or: Reward = Return if Return > 0 else Return * 1.5 (Loss aversion)
        # Let's try the Sharpe Proxy: R = r - 0.5 * r^2 (if we assume mean=0, var=E[r^2])
        # Actually, let's just use raw return * 100 for stability first. 
        # LSTM itself helps smoothing.
        reward = raw_reward * 100 
        
        self.current_position = target_position
        self.current_step += 1
        
        done = self.current_step >= len(self.features) - 1
        
        obs = self._next_observation() if not done else np.zeros((self.window_size, self.n_features + 1))
        return obs, reward, done, {}

# 3. PPO Agent with LSTM
class ActorCriticLSTM(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super(ActorCriticLSTM, self).__init__()
        
        # LSTM Feature Extractor
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError
    
    def get_features(self, x):
        # x: (Batch, Window, Features)
        self.lstm.flatten_parameters()
        out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        return out[:, -1, :] # (Batch, Hidden)

    def act(self, state):
        # state: (Window, Features) -> Add batch dim
        if state.ndim == 2:
            state = state.unsqueeze(0)
        
        features = self.get_features(state)
        action_mean = self.actor(features)
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        # state: (Batch, Window, Features)
        features = self.get_features(state)
        
        action_mean = self.actor(features)
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(features)
        
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, input_dim, action_dim):
        self.policy = ActorCriticLSTM(input_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.policy_old = ActorCriticLSTM(input_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach().squeeze(1)
        old_logprobs = torch.stack(memory.logprobs).to(device).detach().squeeze(1)
        
        for _ in range(K_EPOCHS):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages

            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# 4. Training Loop
print("\n[3] Training PPO Agent (LSTM)...")
env = HullTradingEnvV2(train_features, train_returns, window_size=WINDOW_SIZE)
input_dim = env.n_features + 1
action_dim = 1
ppo = PPO(input_dim, action_dim)
memory = Memory()

time_step = 0
i_episode = 0

while time_step < TOTAL_TIMESTEPS:
    state = env.reset()
    current_ep_reward = 0
    done = False
    
    while not done:
        time_step += 1
        
        state_tensor = torch.FloatTensor(state).to(device) # (Window, F)
        action, action_logprob = ppo.policy_old.act(state_tensor)
        action_val = action.cpu().data.numpy()[0]
        
        next_state, reward, done, _ = env.step(action_val)
        
        memory.states.append(state_tensor)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        
        state = next_state
        current_ep_reward += reward
        
        if time_step % T_HORIZON == 0:
            ppo.update(memory)
            memory.clear_memory()
            print(f"   Step {time_step}/{TOTAL_TIMESTEPS} - Updated PPO")
            
    i_episode += 1
    if i_episode % 10 == 0:
        # Convert to scalar if needed
        if isinstance(current_ep_reward, np.ndarray):
            current_ep_reward = current_ep_reward.item()
        print(f"   Episode {i_episode} - Reward: {current_ep_reward:.2f}")

# Save Model
torch.save(ppo.policy.state_dict(), "experiments/044_rl_ppo/ppo_lstm_agent.pth")
print("   Model saved to experiments/044_rl_ppo/ppo_lstm_agent.pth")

# 5. Evaluate
print("\n[4] Evaluating on Validation Set...")
val_env = HullTradingEnvV2(val_features, val_returns, window_size=WINDOW_SIZE)
state = val_env.reset()
done = False
rewards = []
positions = []

while not done:
    state_tensor = torch.FloatTensor(state).to(device)
    # Deterministic Eval
    if state_tensor.ndim == 2:
        state_tensor = state_tensor.unsqueeze(0)
    features = ppo.policy.get_features(state_tensor)
    action_mean = ppo.policy.actor(features)
    action_val = action_mean.cpu().data.numpy()[0]
    action_val = np.clip(action_val, -1, 1)
    
    state, reward, done, _ = val_env.step(action_val)
    rewards.append(reward/100)
    positions.append(action_val)

rewards = np.array(rewards)
sharpe = np.mean(rewards) / (np.std(rewards) + 1e-9) * np.sqrt(252)
print(f"   Validation Sharpe Ratio: {sharpe:.4f}")

# Save Results
results_df = pd.DataFrame({
    'position': np.array(positions).flatten(),
    'reward': np.array(rewards).flatten()
})
results_df.to_csv("experiments/044_rl_ppo/results/val_results_v2.csv", index=False)
print("   Results saved to experiments/044_rl_ppo/results/val_results_v2.csv")
