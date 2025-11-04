"""
EXP-025 Phase 3: 2D U-Net with Feature Grouping

Reshape features into 2D grid and apply 2D U-Net
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
print("EXP-025 Phase 3: 2D U-Net with Feature Grouping")
print("="*80)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# ============================================================================
# 1. 2D U-Net Components
# ============================================================================

class ResidualBlock2D(nn.Module):
    """2D Residual Block for U-Net"""
    def __init__(self, channels, dropout=0.3):
        super(ResidualBlock2D, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Skip connection
        out = F.relu(out)

        return out


class EncoderBlock2D(nn.Module):
    """2D Encoder block with downsampling"""
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super(EncoderBlock2D, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.res_block = ResidualBlock2D(out_channels, dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.res_block(x)
        return x


class DecoderBlock2D(nn.Module):
    """2D Decoder block with upsampling and skip connection"""
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super(DecoderBlock2D, self).__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.res_block = ResidualBlock2D(out_channels, dropout)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = self.bn(x)
        x = F.relu(x)

        # Handle size mismatch
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            diff_h = skip.size(2) - x.size(2)
            diff_w = skip.size(3) - x.size(3)
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2])

        # Add skip connection
        x = x + skip
        x = self.res_block(x)
        return x


class UNet2D(nn.Module):
    """2D U-Net for tabular data"""
    def __init__(self, input_dim=305, grid_h=17, grid_w=18, channels=[32, 64, 128, 256], dropout=0.3):
        super(UNet2D, self).__init__()

        self.input_dim = input_dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.pad_size = grid_h * grid_w - input_dim  # Padding needed

        # Input projection
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            ResidualBlock2D(channels[0], dropout)
        )

        # Encoder
        self.encoder1 = EncoderBlock2D(channels[0], channels[1], dropout)
        self.encoder2 = EncoderBlock2D(channels[1], channels[2], dropout)
        self.encoder3 = EncoderBlock2D(channels[2], channels[3], dropout)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock2D(channels[3], dropout),
            ResidualBlock2D(channels[3], dropout)
        )

        # Decoder
        self.decoder3 = DecoderBlock2D(channels[3], channels[2], dropout)
        self.decoder2 = DecoderBlock2D(channels[2], channels[1], dropout)
        self.decoder1 = DecoderBlock2D(channels[1], channels[0], dropout)

        # Output
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.output_layer = nn.Linear(channels[0], 1)

    def forward(self, x):
        # x: (batch, 305)
        batch_size = x.size(0)

        # Pad to grid size (17×18=306, need 1 zero)
        if self.pad_size > 0:
            x = F.pad(x, (0, self.pad_size))

        # Reshape to 2D grid
        x = x.view(batch_size, 1, self.grid_h, self.grid_w)  # (batch, 1, 17, 18)

        # Encoder
        e0 = self.input_conv(x)        # (batch, 32, 17, 18)
        e1 = self.encoder1(e0)         # (batch, 64, 9, 9)
        e2 = self.encoder2(e1)         # (batch, 128, 5, 5)
        e3 = self.encoder3(e2)         # (batch, 256, 3, 3)

        # Bottleneck
        b = self.bottleneck(e3)        # (batch, 256, 3, 3)

        # Decoder with skip connections
        d3 = self.decoder3(b, e2)      # (batch, 128, 5, 5) + skip from e2
        d2 = self.decoder2(d3, e1)     # (batch, 64, 9, 9) + skip from e1
        d1 = self.decoder1(d2, e0)     # (batch, 32, 17, 18) + skip from e0

        # Global pooling and output
        pooled = self.global_pool(d1)  # (batch, 32, 1, 1)
        pooled = pooled.view(batch_size, -1)  # (batch, 32)
        out = self.output_layer(pooled)  # (batch, 1)

        return out


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
print(f"   2D Grid: 17×18 = 306 (pad 1 zero)")

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

print("\n[2] Training 2D U-Net with 5-fold CV...")

K = 250
n_epochs = 50
batch_size = 64
learning_rate = 0.001

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

    # Model
    model = UNet2D(
        input_dim=X_all.shape[1],
        grid_h=17,
        grid_w=18,
        channels=[32, 64, 128, 256],
        dropout=0.3
    ).to(device)

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    # Training
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_preds = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
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
    'EXP-016': 0.559,
    'EXP-021': 0.637,
    'EXP-024': 0.559,
    'Phase 1 ResNet': 0.547,
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

# Save
output_path = Path("experiments/025/results/phase3_unet2d_results.csv")
df_results.to_csv(output_path, index=False)
print(f"\n💾 Saved to: {output_path}")

print("\n" + "="*80)
print("Phase 3 Complete!")
print("="*80)
