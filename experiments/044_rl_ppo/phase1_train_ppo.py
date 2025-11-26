#!/usr/bin/env python3
"""
EXP-044 Phase 1: Train PPO Agent (Custom PyTorch Implementation)
Goal: Train a Reinforcement Learning agent (PPO) to maximize Sharpe Ratio.
Note: Implemented from scratch in PyTorch to avoid external dependencies.
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
print("EXP-044 Phase 1: Train PPO Agent (PyTorch)")
print("="*80)

# Hyperparameters
LR = 0.0003
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4
BATCH_SIZE = 64
T_HORIZON = 2048
TOTAL_TIMESTEPS = 50000

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

# 2. Environment
class HullTradingEnv:
    def __init__(self, features, returns):
        self.features = features
        self.returns = returns
        self.n_features = features.shape[1]
        self.reset()

    def reset(self):
        self.current_step = 0
        self.current_position = 0.0
        return self._next_observation()

    def _next_observation(self):
        feat = self.features[self.current_step]
        # Normalize features roughly if needed, but tree models didn't need it.
        # For NN, it's better. Simple standardization here?
        # Assuming features are already somewhat scaled or we rely on LayerNorm.
        obs = np.append(feat, self.current_position)
        return obs

    def step(self, action):
        # Action is continuous [-1, 1]
        target_position = np.clip(action, -1, 1)
        
        # Reward
        current_return = self.returns[self.current_step]
        reward = target_position * current_return * 100 # Scale up reward for stability
        
        self.current_position = target_position
        self.current_step += 1
        
        done = self.current_step >= len(self.features) - 1
        
        obs = self._next_observation() if not done else np.zeros(self.n_features + 1)
        return obs, reward, done, {}

# 3. PPO Agent
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Shared features? Or separate. Let's separate for simplicity.
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh() # Output [-1, 1]
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim)) # Learnable std
        
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
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device)
        
        # Optimize policy for K epochs
        for _ in range(K_EPOCHS):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
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
print("\n[3] Training PPO Agent...")
env = HullTradingEnv(train_features, train_returns)
state_dim = env.n_features + 1
action_dim = 1
ppo = PPO(state_dim, action_dim)
memory = Memory()

time_step = 0
i_episode = 0

while time_step < TOTAL_TIMESTEPS:
    state = env.reset()
    current_ep_reward = 0
    done = False
    
    while not done:
        time_step += 1
        
        # Select action
        state_tensor = torch.FloatTensor(state).to(device)
        action, action_logprob = ppo.policy_old.act(state_tensor)
        action_val = action.cpu().data.numpy()[0]
        
        # Step
        state, reward, done, _ = env.step(action_val)
        
        # Save in memory
        memory.states.append(state_tensor)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        
        current_ep_reward += reward
        
        # Update PPO
        if time_step % T_HORIZON == 0:
            ppo.update(memory)
            memory.clear_memory()
            print(f"   Step {time_step}/{TOTAL_TIMESTEPS} - Updated PPO")
            
    i_episode += 1
    if i_episode % 10 == 0:
        print(f"   Episode {i_episode} - Reward: {current_ep_reward:.2f}")

# Save Model
torch.save(ppo.policy.state_dict(), "experiments/044_rl_ppo/ppo_actor_critic.pth")
print("   Model saved to experiments/044_rl_ppo/ppo_actor_critic.pth")

# 5. Evaluate
print("\n[4] Evaluating on Validation Set...")
val_env = HullTradingEnv(val_features, val_returns)
state = val_env.reset()
done = False
rewards = []
positions = []

while not done:
    state_tensor = torch.FloatTensor(state).to(device)
    action, _ = ppo.policy.act(state_tensor)
    action_val = action.cpu().data.numpy()[0]
    # Deterministic for eval? PPO samples. 
    # Usually for eval we take mean, but here we sample. 
    # Let's use mean (actor output) directly for deterministic eval.
    action_mean = ppo.policy.actor(state_tensor)
    action_val = action_mean.cpu().data.numpy()[0]
    action_val = np.clip(action_val, -1, 1) # Tanh already does this but good to be safe
    
    state, reward, done, _ = val_env.step(action_val)
    rewards.append(reward/100) # Scale back
    positions.append(action_val)

rewards = np.array(rewards)
sharpe = np.mean(rewards) / (np.std(rewards) + 1e-9) * np.sqrt(252)
print(f"   Validation Sharpe Ratio: {sharpe:.4f}")

# Save Results
results_df = pd.DataFrame({
    'position': np.array(positions).flatten(),
    'reward': rewards
})
results_df.to_csv("experiments/044_rl_ppo/results/val_results.csv", index=False)
print("   Results saved to experiments/044_rl_ppo/results/val_results.csv")
