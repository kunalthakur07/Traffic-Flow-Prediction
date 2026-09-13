# 🚦 Traffic Flow Prediction using LSTM

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red.svg)
![Dataset](https://img.shields.io/badge/Dataset-METR--LA-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> A deep learning pipeline that predicts real-world traffic speed across 207 road sensors using an LSTM neural network trained on the METR-LA dataset — with a live interactive Streamlit web application.

---

## 📌 Project Summary

Traffic congestion is one of the most studied problems in smart city research. This project builds an end-to-end traffic speed forecasting system using **Long Short-Term Memory (LSTM)** networks trained on real sensor data from Los Angeles highways.

The model takes the **last 12 time steps (1 hour of traffic data)** as input and predicts the **next 3 time steps (15 minutes ahead)** across all 207 road sensors simultaneously.

---

## 🏆 Key Results

| Model | MAE | RMSE |
|-------|-----|------|
| Persistence Baseline | 0.34 | 0.61 |
| Historical Average Baseline | 0.28 | 0.59 |
| **LSTM (Ours)** | **Lower** | **Lower** |

**LSTM outperforms both classical baselines on the METR-LA test set.**

---

## 🔬 Complete Pipeline
METR-LA Dataset (HDF5)
↓
Data Preprocessing
(Missing value interpolation + Z-score normalization)
↓
Train / Val / Test Split
(70% / 10% / 20% by time)
↓
Baseline Models
(Persistence forecast + Historical average)
↓
LSTM Model Training
(PyTorch — 100 epochs, Adam optimizer, MAE loss)
↓
Evaluation
(MAE + RMSE on test set)
↓
Streamlit Web Application
(Live sensor selection + future prediction + graph)


---

## 📦 Dataset

**METR-LA — Los Angeles Metropolitan Traffic Dataset**

| Property | Value |
|----------|-------|
| Source | METR-LA benchmark dataset |
| Sensors | 207 road sensors |
| Time range | 4 months |
| Interval | 5-minute bins |
| Metric | Traffic speed (km/h) |
| File format | HDF5 (.h5) |

> Note: The dataset file `metr-la.h5` is not included in this repository due to size. Download it from the official source and place it in the `data/` folder.

---

## 🤖 LSTM Model Architecture

Input: (batch, 12 time steps, 207 sensors)
↓
LSTM Layer
(input_size=207, hidden_size=64, num_layers=1)
↓
Last Hidden State
↓
Fully Connected Layer
(64 → 207 × 3)
↓
Reshape
(batch, 3 horizon steps, 207 sensors)
↓
Output: Predicted traffic speed for next 15 minutes


| Parameter | Value |
|-----------|-------|
| Input length | 12 time steps (1 hour) |
| Prediction horizon | 3 time steps (15 minutes) |
| Hidden size | 64 |
| LSTM layers | 1 |
| Optimizer | Adam (lr=0.001) |
| Loss function | MAE (L1 Loss) |
| Batch size | 64 |
| Epochs | 100 |

---

## 🌐 Web Application

Built with **Streamlit**

**Features:**
- Select any sensor ID from 0 to 206
- Choose prediction horizon from 1 to 6 steps
- Click Predict to run the LSTM model live
- View past data vs predicted traffic speed on an interactive graph
- See numeric predicted values in a table

**To launch:**
```bash
streamlit run app.py
```


---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.13 | Core language |
| PyTorch | LSTM model training and inference |
| NumPy | Array operations and data handling |
| Pandas | Dataset loading and preprocessing |
| Scikit-learn | MAE and RMSE evaluation metrics |
| Matplotlib | Visualization of predictions vs actuals |
| Streamlit | Interactive web application |
| h5py | Reading HDF5 dataset files |
| SciPy | Signal processing utilities |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/traffic-prediction.git
cd traffic-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add dataset
Download `metr-la.h5` and place it in the `data/` folder.

### 4. Run the training pipeline
```bash
python project.py
```
This will:
- Load and preprocess the data
- Train the LSTM model
- Evaluate against baselines
- Save model weights to `models/`
- Save `data_norm.npy` to `data/`

### 5. Launch the web app
```bash
streamlit run app.py
```

---

## 📊 Baseline Comparisons

**Persistence Forecast** — assumes next value equals current value. Simple but reasonable for short horizons.

**Historical Average** — uses the average traffic speed at the same time of day across all training days. Captures daily patterns but misses real-time variations.

**LSTM** — learns temporal dependencies across 12 time steps and 207 sensors simultaneously. Outperforms both baselines by capturing complex non-linear traffic patterns.

---

## 📚 References

- METR-LA Dataset — Li et al., Diffusion Convolutional Recurrent Neural Network
- PyTorch LSTM Documentation — pytorch.org
- Streamlit — streamlit.io

---

## 👤 Author

**Kunal Thakur**
Roll No: 1024240018 | Batch: 2X11

---

## 📄 License

This project is licensed under the MIT License.







