#!/usr/bin/env python3
"""
EXP-044 Phase 3: Train PPO Agent (LSTM + Differential Sharpe Reward)
Goal: Train a PPO agent with LSTM and Differential Sharpe Ratio to directly optimize risk-adjusted returns.
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
print("EXP-044 Phase 3: Train PPO Agent (LSTM + Differential Sharpe Reward)")
print("="*80)

# Hyperparameters
LR = 0.0001
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4
BATCH_SIZE = 64
T_HORIZON = 2048
TOTAL_TIMESTEPS = 100000
WINDOW_SIZE = 10

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

# 2. Environment V3 (Differential Sharpe)
class HullTradingEnvV3:
    def __init__(self, features, returns, window_size=10, decay=0.99):
        self.features = features
        self.returns = returns
        self.window_size = window_size
        self.n_features = features.shape[1]
        self.decay = decay # Decay rate for moving averages
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.current_position = 0.0
        
        # DSR State Variables
        # A: Moving Average of Returns
        # B: Moving Average of Squared Returns
        self.A = 0.0
        self.B = 0.0
        
        return self._next_observation()

    def _next_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step
        window_feat = self.features[start:end]
        pos_col = np.full((self.window_size, 1), self.current_position)
        obs = np.hstack([window_feat, pos_col])
        return obs

    def step(self, action):
        target_position = np.clip(action, -1, 1)
        
        # Calculate Return
        current_return = self.returns[self.current_step]
        r_t = target_position * current_return * 100 # Scale up
        
        # Update DSR Moving Averages
        # Standard DSR update:
        # A_t = eta * r_t + (1-eta) * A_{t-1}
        # B_t = eta * r_t^2 + (1-eta) * B_{t-1}
        eta = 1.0 - self.decay
        
        prev_A = self.A
        prev_B = self.B
        
        self.A = eta * r_t + (1 - eta) * prev_A
        self.B = eta * (r_t**2) + (1 - eta) * prev_B
        
        # Differential Sharpe Ratio (D_t)
        # D_t = (B_{t-1} * Delta A_t - 0.5 * A_{t-1} * Delta B_t) / (B_{t-1} - A_{t-1}^2)^(3/2)
        # Delta A_t = r_t - A_{t-1}
        # Delta B_t = r_t^2 - B_{t-1}
        
        delta_A = r_t - prev_A
        delta_B = (r_t**2) - prev_B
        
        variance = prev_B - (prev_A**2)
        std_dev = np.sqrt(variance + 1e-9)
        
        # Avoid division by zero or complex numbers
        if variance <= 1e-9:
            dsr = 0.0
        else:
            term1 = prev_B * delta_A
            term2 = 0.5 * prev_A * delta_B
            dsr = (term1 - term2) / (variance * std_dev)
            
        # Reward is the Differential Sharpe Ratio
        # Clip reward to prevent instability
        reward = np.clip(dsr, -10, 10)
        
        self.current_position = target_position
        self.current_step += 1
        
        done = self.current_step >= len(self.features) - 1
        
        obs = self._next_observation() if not done else np.zeros((self.window_size, self.n_features + 1))
        return obs, reward, done, {}

# 3. PPO Agent with LSTM (Same as v2)
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
print("\n[3] Training PPO Agent (LSTM + DSR)...")
env = HullTradingEnvV3(train_features, train_returns, window_size=WINDOW_SIZE)
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
        state_tensor = torch.FloatTensor(state).to(device)
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
        if isinstance(current_ep_reward, np.ndarray):
            current_ep_reward = current_ep_reward.item()
        print(f"   Episode {i_episode} - Reward (DSR Sum): {current_ep_reward:.2f}")

# Save Model
torch.save(ppo.policy.state_dict(), "experiments/044_rl_ppo/ppo_dsr_agent.pth")
print("   Model saved to experiments/044_rl_ppo/ppo_dsr_agent.pth")

# 5. Evaluate
print("\n[4] Evaluating on Validation Set...")
val_env = HullTradingEnvV3(val_features, val_returns, window_size=WINDOW_SIZE)
state = val_env.reset()
done = False
rewards = []
positions = []

while not done:
    state_tensor = torch.FloatTensor(state).to(device)
    if state_tensor.ndim == 2:
        state_tensor = state_tensor.unsqueeze(0)
    features = ppo.policy.get_features(state_tensor)
    action_mean = ppo.policy.actor(features)
    action_val = action_mean.cpu().data.numpy()[0]
    action_val = np.clip(action_val, -1, 1)
    
    # Note: We want to evaluate actual returns, not DSR reward
    current_return = val_env.returns[val_env.current_step]
    actual_reward = action_val * current_return
    
    state, _, done, _ = val_env.step(action_val)
    rewards.append(actual_reward)
    positions.append(action_val)

rewards = np.array(rewards)
sharpe = np.mean(rewards) / (np.std(rewards) + 1e-9) * np.sqrt(252)
print(f"   Validation Sharpe Ratio: {sharpe:.4f}")

# Save Results
results_df = pd.DataFrame({
    'position': np.array(positions).flatten(),
    'reward': np.array(rewards).flatten()
})
results_df.to_csv("experiments/044_rl_ppo/results/val_results_v3.csv", index=False)
print("   Results saved to experiments/044_rl_ppo/results/val_results_v3.csv")
