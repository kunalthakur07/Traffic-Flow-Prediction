import streamlit as st
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Page Title
# ---------------------------
st.title("🚦 Traffic Flow Prediction using LSTM")

# ---------------------------
# Import your trained LSTM model
# ---------------------------
from lstm_model import SimpleLSTM, predict_future

# ---------------------------
# Setup device
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------
# Load model
# ---------------------------
model = SimpleLSTM(n_sensors=207).to(device)
model.load_state_dict(torch.load('./models/lstm_baseline.pt', map_location=device))
model.eval()

# ---------------------------
# Load scaler
# ---------------------------
with open('./models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
mean, std = scaler['mean'], scaler['std']

# ---------------------------
# Load normalized test data
# (Make sure you have saved data_norm.npy from your notebook)
# ---------------------------
try:
    data_norm = np.load('./data/data_norm.npy')
except:
    st.error("⚠️ 'data_norm.npy' not found! Please save it from your notebook using: np.save('./data/data_norm.npy', data_norm)")
    st.stop()

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("🔧 Model Controls")
sensor_id = st.sidebar.number_input("Sensor ID (0–206):", min_value=0, max_value=206, value=0)
steps = st.sidebar.slider("Prediction Horizon (steps):", 1, 6, 3)

st.write(f"### Predicting for Sensor {sensor_id}, next {steps} time steps")

# ---------------------------
# Predict Button
# ---------------------------
if st.button("🚀 Predict Traffic"):

    # ✅ Use last 12 real data points from the dataset
    sample_input = data_norm[-12:]  # shape: (12, 207)

    # Convert to model input
    prediction = predict_future(sample_input, model, steps)

    # Reverse normalization (denormalize)
    prediction_real = prediction * std + mean
    past_real = sample_input * std + mean

    # Get actual steps returned
    actual_steps = prediction_real.shape[0]

    # ---------------------------
    # Plot Past vs Predicted
    # ---------------------------
    fig, ax = plt.subplots()
    ax.plot(range(12), past_real[:, sensor_id], label="Past Data", color='skyblue')
    ax.plot(range(12, 12 + actual_steps), prediction_real[:, sensor_id], label=f"Predicted ({actual_steps} steps)", color='orange')
    ax.set_xlabel("Time steps (5-min intervals)")
    ax.set_ylabel("Traffic Speed (km/h)")
    ax.set_title(f"Traffic Speed Prediction (Sensor {sensor_id})")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # ---------------------------
    # Optional: Display numeric prediction
    # ---------------------------
    st.write("### Predicted Values:")
    st.dataframe(prediction_real[:steps, sensor_id].reshape(-1, 1), use_container_width=True)
