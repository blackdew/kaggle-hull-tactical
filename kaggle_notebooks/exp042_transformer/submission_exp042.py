# %% [code]
#!/usr/bin/env python3
"""
EXP-042: Time Series Transformer (Deep Learning)
Uses a Transformer Encoder to predict returns based on past 30 days sequence.
"""
from kaggle_evaluation.core.templates import InferenceServer
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
from collections import deque

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

# Hyperparameters (Must match training)
SEQ_LEN = 30
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DROPOUT = 0.0  # No dropout for inference

# Model Definition
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        # src: [batch, seq_len, input_dim]
        x = self.input_proj(src) # [batch, seq_len, d_model]
        x = x + self.pos_encoder # Add positional encoding
        x = self.transformer_encoder(x) # [batch, seq_len, d_model]
        # Take the last time step
        x = x[:, -1, :] # [batch, d_model]
        output = self.decoder(x) # [batch, 1]
        return output.squeeze()

class MyServer(InferenceServer):
    """EXP-042: Transformer with History Buffer"""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        
        self.StandardScaler = StandardScaler
        self.ready = False
        self.model = None
        self.scaler = None
        self.K = 250
        self.history = deque(maxlen=SEQ_LEN)
        self.device = torch.device("cpu") # Use CPU for inference

        # Top 20 Base Features
        self.top_20 = [
            'M4', 'V13', 'V7', 'P8', 'S2', 'I2', 'E19', 'S5', 'P5', 'P7',
            'M2', 'V9', 'M3', 'P12', 'P10', 'V10', 'E12', 'P11', 'M12', 'S8'
        ]

        def predict(batch):
            return MyServer.predict(self, batch)
        super().__init__(predict)

    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None, *args, **kwargs):
        return DefaultGateway(data_paths)

    def train_if_needed(self):
        """Lazy training on first prediction"""
        if self.ready:
            return

        # Load training data to fit scaler and train model
        try:
            train = pd.read_csv("/kaggle/input/hull-tactical-market-prediction/train.csv")
        except:
            train = pd.read_csv("data/train.csv")
        
        y_excess = train['market_forward_excess_returns'].values
        
        # Prepare Features
        X = train[self.top_20].fillna(train[self.top_20].median()).replace([np.inf, -np.inf], np.nan).fillna(0).values
        
        # Fit Scaler
        self.scaler = self.StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Model (or Load Pretrained)
        # Here we train from scratch for simplicity in the submission script
        # In a real scenario, we should load weights.
        print("Training Transformer Model...")
        
        # Create Dataset
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_excess, dtype=torch.float32)
        
        # Simple training loop
        self.model = TransformerModel(input_dim=len(self.top_20), d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        BATCH_SIZE = 64
        EPOCHS = 5 # Reduced epochs for submission speed
        
        # Create batches manually or use DataLoader
        num_samples = len(X_tensor) - SEQ_LEN
        indices = np.arange(num_samples)
        
        for epoch in range(EPOCHS):
            np.random.shuffle(indices)
            total_loss = 0
            count = 0
            
            for i in range(0, num_samples, BATCH_SIZE):
                batch_idx = indices[i : i + BATCH_SIZE]
                if len(batch_idx) == 0: continue
                
                # Create batch sequences
                batch_X = []
                batch_y = []
                for idx in batch_idx:
                    batch_X.append(X_tensor[idx : idx + SEQ_LEN])
                    batch_y.append(y_tensor[idx + SEQ_LEN])
                
                batch_X = torch.stack(batch_X).to(self.device)
                batch_y = torch.stack(batch_y).to(self.device)
                
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1
            
            print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss/count:.6f}")
            
        self.model.eval()
        
        # Initialize history with the last SEQ_LEN data points from training
        # This ensures we have a valid sequence for the first test point
        last_history = X_scaled[-SEQ_LEN:]
        for row in last_history:
            self.history.append(row)
            
        self.ready = True

    def create_features(self, df):
        """Create features from raw data"""
        if pl is not None and isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        for feat in self.top_20:
            if feat not in df.columns:
                df[feat] = 0.0

        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        X = df[self.top_20].values
        return X

    def predict(self, test_batch):
        """Predict positions using Transformer"""
        self.train_if_needed()

        # Unpack batch
        if isinstance(test_batch, (tuple, list)) and len(test_batch) == 1:
            test_batch = test_batch[0]
        df = test_batch
        
        if pl is not None and hasattr(pl, 'DataFrame') and isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Create features (1 row)
        X_raw = self.create_features(df)
        
        # Scale
        X_scaled = self.scaler.transform(X_raw)
        
        # Update History
        # Assuming test_batch comes sequentially
        for row in X_scaled:
            self.history.append(row)
            
        # If history is not full (should not happen if initialized correctly), pad
        while len(self.history) < SEQ_LEN:
            self.history.appendleft(np.zeros(len(self.top_20)))
            
        # Create Sequence Tensor
        seq_array = np.array(self.history) # [SEQ_LEN, input_dim]
        seq_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(self.device) # [1, SEQ_LEN, input_dim]
        
        # Predict
        with torch.no_grad():
            pred = self.model(seq_tensor).item()
            
        # Strategy
        # Position = 1 + Pred * K
        position = 1.0 + pred * self.K
        position = np.clip(position, 0.0, 2.0)

        return float(position)


if __name__ == '__main__':
    print("[START] Kaggle Hull Tactical Submission - EXP-042 (Transformer)")
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
