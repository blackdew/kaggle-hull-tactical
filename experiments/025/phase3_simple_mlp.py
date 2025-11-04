"""
EXP-025 Phase 3: Simple MLP with Strong Regularization

간단한 구조 + 강한 regularization으로 안정적인 학습
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EXP-025 Phase 3: Simple MLP with Strong Regularization")
print("="*80)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# ============================================================================
# 1. Simple MLP Model
# ============================================================================

class SimpleMLP(nn.Module):
    """Simple 3-layer MLP with strong regularization"""
    def __init__(self, input_dim=305, hidden_dims=[128, 64], dropout=0.5):
        super(SimpleMLP, self).__init__()

        # Layer 1: 305 → 128
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.dropout1 = nn.Dropout(dropout)

        # Layer 2: 128 → 64
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout2 = nn.Dropout(dropout)

        # Output: 64 → 1
        self.fc_out = nn.Linear(hidden_dims[1], 1)

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Output
        x = self.fc_out(x)
        return x


# ============================================================================
# 2. Dataset
# ============================================================================

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# 3. Load Data
# ============================================================================

print("\n[1] Loading data...")
df = pd.read_csv("data/train.csv")

# Target
y_excess = df['market_forward_excess_returns'].values
fwd_returns = df['forward_returns'].values
risk_free = df['risk_free_rate'].values

# Load EXP-024 features (305)
feature_list = pd.read_csv("experiments/024/results/phase3_selected_features_all.csv")
all_features = feature_list['feature'].tolist()

# Load engineered features
engineered_data = pd.read_parquet("experiments/024/results/phase2_engineered_features.parquet")

# Get original features
phase2_list = pd.read_csv("experiments/024/results/phase2_feature_list.csv")
original_features = phase2_list[phase2_list['type'] == 'original']['feature'].tolist()
original_data = df[original_features].copy()

# Combine
X_all = pd.concat([original_data, engineered_data], axis=1)
X_all = X_all[all_features]
X_all = X_all.fillna(0).replace([np.inf, -np.inf], 0)

print(f"   Features: {X_all.shape[1]}")
print(f"   Samples: {len(X_all)}")

# ============================================================================
# 4. Training Functions
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            predictions.extend(outputs.cpu().numpy().flatten())

    return total_loss / len(dataloader), np.array(predictions)


def calculate_sharpe(positions, fwd_returns, risk_free):
    positions = np.clip(positions, 0.0, 2.0)
    strategy_returns = risk_free * (1.0 - positions) + fwd_returns * positions
    excess_returns = strategy_returns - risk_free

    if np.std(strategy_returns) > 0:
        sharpe = (np.mean(excess_returns) / np.std(strategy_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    return sharpe


# ============================================================================
# 5. Cross-Validation Training
# ============================================================================

print("\n[2] Training Simple MLP with 5-fold CV...")
print("   Architecture: 305 → 128 → 64 → 1")
print("   Dropout: 0.5 (strong regularization)")
print("   Learning rate: 0.0001 (conservative)")
print("   Weight decay: 1e-3")

K = 250
n_epochs = 100  # More epochs since LR is low
batch_size = 64
learning_rate = 0.0001

tscv = TimeSeriesSplit(n_splits=5)
results = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_all), 1):
    print(f"\n--- Fold {fold_idx}/5 ---")

    X_train, X_val = X_all.iloc[train_idx].values, X_all.iloc[val_idx].values
    y_train, y_val = y_excess[train_idx], y_excess[val_idx]
    fwd_ret_val = fwd_returns[val_idx]
    rf_val = risk_free[val_idx]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Datasets
    train_dataset = TabularDataset(X_train_scaled, y_train)
    val_dataset = TabularDataset(X_val_scaled, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model - Simple architecture
    model = SimpleMLP(
        input_dim=X_all.shape[1],
        hidden_dims=[128, 64],
        dropout=0.5
    ).to(device)

    # Optimizer & Loss with strong regularization
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-3  # Strong L2 regularization
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )

    # Training
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15  # More patience since LR is low

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_preds = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}/{n_epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best predictions
            best_preds = val_preds
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break

    # Calculate Sharpe with best predictions
    positions = np.clip(1.0 + best_preds * K, 0.0, 2.0)
    sharpe = calculate_sharpe(positions, fwd_ret_val, rf_val)

    pos_mean = np.mean(positions)
    pos_std = np.std(positions)
    pred_std = np.std(best_preds)

    print(f"   Best Val Loss: {best_val_loss:.6f}")
    print(f"   Sharpe: {sharpe:.3f}")
    print(f"   Position: {pos_mean:.3f} ± {pos_std:.3f}")
    print(f"   Pred std: {pred_std:.6f}")

    results.append({
        'fold': fold_idx,
        'sharpe': sharpe,
        'val_loss': best_val_loss,
        'pos_mean': pos_mean,
        'pos_std': pos_std,
        'pred_std': pred_std,
    })

# ============================================================================
# 6. Results Analysis
# ============================================================================

print("\n[3] Results Analysis")
print("="*80)

df_results = pd.DataFrame(results)

cv_sharpe_mean = df_results['sharpe'].mean()
cv_sharpe_std = df_results['sharpe'].std()

print(f"\n📊 Cross-Validation Results:")
print(f"   CV Sharpe: {cv_sharpe_mean:.3f} ± {cv_sharpe_std:.3f}")
print(f"   Range: [{df_results['sharpe'].min():.3f}, {df_results['sharpe'].max():.3f}]")

# Comparison
baselines = {
    'EXP-016 (XGBoost)': 0.559,
    'EXP-021 (XGBoost)': 0.637,
    'EXP-024 (XGBoost over-reg)': 0.559,
    'Phase 1 (ResNet)': 0.547,
}

print(f"\n📈 Comparison:")
for exp, sharpe in baselines.items():
    improvement = (cv_sharpe_mean - sharpe) / sharpe * 100
    symbol = "✅" if improvement > 0 else "❌"
    print(f"   vs {exp} ({sharpe:.3f}): {improvement:+.1f}% {symbol}")

# Expected Public Score
avg_ratio = 8.6
expected_public = cv_sharpe_mean * avg_ratio
print(f"\n🎯 Expected Public Score: {expected_public:.2f}")

# Stability check
print(f"\n🔍 Stability Analysis:")
print(f"   Fold std: {cv_sharpe_std:.3f}")
if cv_sharpe_std < 0.2:
    print(f"   → ✅ Very stable model")
elif cv_sharpe_std < 0.3:
    print(f"   → ✅ Stable model")
else:
    print(f"   → ⚠️  High variance")

# Save
output_path = Path("experiments/025/results/phase3_simple_mlp_results.csv")
df_results.to_csv(output_path, index=False)
print(f"\n💾 Saved to: {output_path}")

print("\n" + "="*80)
print("Phase 3 Complete!")
print("="*80)
