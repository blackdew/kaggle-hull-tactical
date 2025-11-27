import torch
import base64
import io

# Load weights
weights = torch.load("experiments/044_rl_ppo/ppo_lstm_agent.pth")

# Save to buffer
buffer = io.BytesIO()
torch.save(weights, buffer)
buffer.seek(0)

# Encode
encoded = base64.b64encode(buffer.read()).decode('utf-8')

print("BASE64_WEIGHTS = " + repr(encoded))
