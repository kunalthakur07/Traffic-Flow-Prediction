import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import h5py
data_path = '../data/metr-la.h5'
f = h5py.File(data_path, 'r')
print(list(f.keys()))   # see dataset keys
# typical key: 'speed' or 'data'
arr = np.array(f['df'])   # shape (time_steps, num_sensors)
print('shape', arr.shape)
import pandas as pd

data_path = '../data/metr-la.h5'
df = pd.read_hdf(data_path)
print(df.shape)
print(df.head())
# Convert DataFrame to NumPy array first
data = df.values.astype(float)

# 1) Handle missing values (interpolate, then forward/backfill)
for i in range(data.shape[1]):
    col = data[:, i]
    if np.isnan(col).any():
        col = pd.Series(col).interpolate(limit_direction='both').fillna(method='ffill').fillna(method='bfill').values
        data[:, i] = col

# 2) Split by time (train/val/test)
n = data.shape[0]
train_end = int(n * 0.7)
val_end = int(n * 0.8)
train = data[:train_end]
val = data[train_end:val_end]
test = data[val_end:]

# 3) Scale (z-score using train stats)
mean = train.mean(axis=0)
std = train.std(axis=0) + 1e-6
data_norm = (data - mean) / std

# 4) Save scaler
import os, pickle
os.makedirs('../models', exist_ok=True)
with open('../models/scaler.pkl', 'wb') as f:
    pickle.dump({'mean': mean, 'std': std}, f)

print('✅ train/val/test shapes:', train.shape, val.shape, test.shape)
# Convert DataFrame to NumPy array first
data = df.values.astype(float)

# 1) Handle missing values (interpolate, then forward/backfill)
for i in range(data.shape[1]):
    col = data[:, i]
    if np.isnan(col).any():
        col = pd.Series(col).interpolate(limit_direction='both').fillna(method='ffill').fillna(method='bfill').values
        data[:, i] = col

# 2) Split by time (train/val/test)
n = data.shape[0]
train_end = int(n * 0.7)
val_end = int(n * 0.8)
train = data[:train_end]
val = data[train_end:val_end]
test = data[val_end:]

# 3) Scale (z-score using train stats)
mean = train.mean(axis=0)
std = train.std(axis=0) + 1e-6
data_norm = (data - mean) / std

# 4) Save scaler
import os, pickle
os.makedirs('../models', exist_ok=True)
with open('../models/scaler.pkl', 'wb') as f:
    pickle.dump({'mean': mean, 'std': std}, f)

print('✅ train/val/test shapes:', train.shape, val.shape, test.shape)
plt.figure(figsize=(12, 4))
plt.plot(data_norm[:288, 0])  # first sensor, first day
plt.title('Traffic Speed (Sensor 0 - 1st Day)')
plt.xlabel('Time steps (5-min intervals)')
plt.ylabel('Normalized speed')
plt.show()
def persistence_forecast(data_norm, horizon=1):
    # data_norm shape (time, sensors)
    return np.roll(data_norm, -horizon, axis=0)

h = 3  # horizon (e.g., 3 steps ahead)
preds = persistence_forecast(data_norm, horizon=h)
# evaluate on test range only
y_true = data_norm[val_end:-h]
y_pred = preds[val_end:-h]

mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
rmse = mean_squared_error(y_true.flatten(), y_pred.flatten())
print('Persistence MAE:', mae, 'RMSE:', rmse)

# compute mean per sensor per time-of-day index (if 5-min bins, idx = minute_of_day // 5)
# Simplest: compute mean across training set for each time index modulo day length
day_steps = 288  # if 5-min bins (24*60/5)
train_days = train.shape[0] // day_steps
train_trim = train[:train_days*day_steps]
train_td = train_trim.reshape(train_days, day_steps, train.shape[1])  # (days, steps, sensors)
hist_avg = train_td.mean(axis=0)  # (steps, sensors)

# to forecast for test, map each test time to step of day:
test_len = test.shape[0]
preds_hist = np.zeros_like(test)
for i in range(test_len):
    step = ( (val_end + i) % day_steps )
    preds_hist[i] = (hist_avg[step] - mean) / std  # scale using train mean/std

mae_hist = mean_absolute_error(data_norm[val_end:val_end+test_len].flatten(), preds_hist.flatten())
print('Historical average MAE:', mae_hist)
import pandas as pd
import numpy as np
import os

# Load original dataset again
df = pd.read_hdf('../data/metr-la.h5')  # or wherever your dataset is
data = df.values.astype(float)

# Normalize it again
mean = data.mean(axis=0)
std = data.std(axis=0) + 1e-6
data_norm = (data - mean) / std

# Create folder if it doesn’t exist
os.makedirs('./data', exist_ok=True)

# Save it
np.save('./data/data_norm.npy', data_norm)
print("✅ data_norm.npy saved successfully at ./data/")
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# hyperparams
INPUT_LEN = 12   # e.g., last 12 steps (1 hour if 5-min bins)
HORIZON = 3
BATCH = 64
EPOCHS = 100
LR = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset class
class TSdataset(Dataset):
    def _init_(self, data_norm, start_idx, end_idx, input_len=INPUT_LEN, horizon=HORIZON):
        X, Y = [], []
        arr = data_norm[start_idx:end_idx]
        for i in range(len(arr) - input_len - horizon + 1):
            X.append(arr[i:i+input_len])
            Y.append(arr[i+input_len:i+input_len+horizon])
        self.X = np.stack(X).astype(np.float32)
        self.Y = np.stack(Y).astype(np.float32)
    def _len_(self):
        return len(self.X)
    def _getitem_(self, idx):
        return self.X[idx], self.Y[idx]

train_ds = TSdataset(data_norm, 0, train_end, INPUT_LEN, HORIZON)
val_ds = TSdataset(data_norm, train_end, val_end, INPUT_LEN, HORIZON)
test_ds = TSdataset(data_norm, val_end, data_norm.shape[0], INPUT_LEN, HORIZON)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH)

# model
class SimpleLSTM(nn.Module):
    def _init_(self, n_sensors, hidden=64, num_layers=1):
        super()._init_()
        self.lstm = nn.LSTM(input_size=n_sensors, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, n_sensors * HORIZON)
        self.n_sensors = n_sensors
    def forward(self, x):
        out, _ = self.lstm(x)   # out: (batch, seq, hidden)
        out = out[:, -1, :]     # take last time step
        out = self.fc(out)
        out = out.view(-1, HORIZON, self.n_sensors)
        return out

n_sensors = data_norm.shape[1]
model = SimpleLSTM(n_sensors).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.L1Loss()   # MAE loss
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device); yb = yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device); yb = yb.to(device)
            val_loss += loss_fn(model(xb), yb).item()
        val_loss /= len(val_loader)
    print(f'Epoch {epoch+1}/{EPOCHS} train_loss={train_loss:.4f} val_loss={val_loss:.4f}')
import os
import numpy as np

# Create 'data' folder if it doesn't exist
os.makedirs('./data', exist_ok=True)

# Now save your normalized array
np.save('./data/data_norm.npy', data_norm)
print("✅ File saved successfully at ./data/data_norm.npy")
# evaluate on test dataset (corrected to work on all sklearn versions)
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

model.eval()
preds_list, trues_list = [], []
with torch.no_grad():
    for xb, yb in DataLoader(test_ds, batch_size=BATCH):
        xb = xb.to(device)
        p = model(xb).cpu().numpy()        # (batch, horizon, sensors)
        preds_list.append(p)
        trues_list.append(yb.numpy())

# concatenate batches -> arrays shaped (samples, horizon, sensors)
preds = np.concatenate(preds_list, axis=0)
trues = np.concatenate(trues_list, axis=0)

# use the last predicted timestep (or whatever target you trained for)
# here preds and trues are (samples, horizon, sensors). If you used horizon prediction,
# choose the horizon index you want to evaluate (e.g. last: -1). If preds already
# match your Y shape, adapt accordingly.
y_true = trues.reshape(-1)   # flatten (samples*horizon*sensors)
y_pred = preds.reshape(-1)

mae_test = mean_absolute_error(y_true, y_pred)
rmse_test = np.sqrt(mean_squared_error(y_true, y_pred))   # manual RMSE (works always)

print('LSTM test MAE:', mae_test, 'RMSE:', rmse_test)
torch.save(model.state_dict(), '../models/lstm_baseline.pt')
with open('../models/metrics.txt','w') as f:
    f.write(f'LSTM MAE: {mae_test}\nRMSE: {rmse_test}\n')
print('LSTM test MAE:', mae_test, 'RMSE:', rmse_test)
print("arr shape:", arr.shape)
print("arr dtype:", arr.dtype)
print("First 5 elements:", arr[:5])
import matplotlib.pyplot as plt

# compare true vs predicted for a few sensors (e.g. 0 to 2)
sensors_to_plot = [0, 1, 2]

plt.figure(figsize=(12, 6))
for i, sensor in enumerate(sensors_to_plot):
    plt.subplot(len(sensors_to_plot), 1, i + 1)
    plt.plot(y_true.reshape(-1, trues.shape[2])[:, sensor][:288], label='Actual', color='blue')
    plt.plot(y_pred.reshape(-1, preds.shape[2])[:, sensor][:288], label='Predicted', color='red', alpha=0.7)
    plt.title(f'Sensor {sensor} - Predicted vs Actual (First Day)')
    plt.xlabel('Time steps (5-min intervals)')
    plt.ylabel('Normalized speed')
    plt.legend()

plt.tight_layout()
plt.show()
mae_persistence = 0.34   # (example: replace with your persistence MAE)
mae_historical = 0.28    # (example: replace with your hist avg MAE)
mae_lstm = mae_test

rmse_persistence = 0.61
rmse_historical = 0.59
rmse_lstm = rmse_test

import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Model': ['Persistence', 'Historical Avg', 'LSTM'],
    'MAE': [mae_persistence, mae_historical, mae_lstm],
    'RMSE': [rmse_persistence, rmse_historical, rmse_lstm]
}

df = pd.DataFrame(data)

plt.figure(figsize=(8, 5))
plt.bar(df['Model'], df['MAE'], color=['gray', 'orange', 'green'])
plt.title('Model Comparison (MAE)')
plt.ylabel('Mean Absolute Error')
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(df['Model'], df['RMSE'], color=['gray', 'orange', 'green'])
plt.title('Model Comparison (RMSE)')
plt.ylabel('Root Mean Squared Error')
plt.show()
def predict_future(data_input, model, steps=3):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data_input, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(x).cpu().numpy()
    return pred[0, -steps:, :]

# Example usage:
sample_input = test[:12]   # last 12 time steps
predicted_speed = predict_future(sample_input, model)
print(predicted_speed)
model.load_state_dict(torch.load('../models/lstm_baseline.pt'))
model.eval()
import numpy as np
np.save('./data/data_norm.npy', data_norm)
import os
import numpy as np

# Create 'data' folder if it doesn't exist
os.makedirs('./data', exist_ok=True)

# Now save your normalized array
np.save('./data/data_norm.npy', data_norm)
print("✅ File saved successfully at ./data/data_norm.npy")