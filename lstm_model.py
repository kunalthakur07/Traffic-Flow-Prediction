import torch
import torch.nn as nn
import numpy as np

# LSTM Model definition
class SimpleLSTM(nn.Module):
    def __init__(self, n_sensors, hidden=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden, n_sensors * 3)  # horizon = 3 by default
        self.n_sensors = n_sensors

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last time step
        out = out.view(-1, 3, self.n_sensors)  # reshape to (batch, horizon, sensors)
        return out


# Prediction helper function
def predict_future(data_input, model, steps=3, device=None):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data_input, dtype=torch.float32).unsqueeze(0)
        if device:
            x = x.to(device)
        pred = model(x).cpu().numpy()
    return pred[0, -steps:, :]
