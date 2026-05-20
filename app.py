from fastapi import FastAPI, UploadFile, File
import joblib
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import mne
import tempfile

app = FastAPI()

# Load model and scaler
model = joblib.load("sleep_model.pkl")
scaler = joblib.load("scaler.pkl")

sleep_labels = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'Deep',
    4: 'REM'
}

@app.get("/")
def home():
    return {"message": "Sleep Detection API Running"}

@app.post("/predict")
async def predict_sleep(file: UploadFile = File(...)):

    # Save uploaded EDF file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    # Read EDF file
    raw = mne.io.read_raw_edf(temp_path, preload=True)

    # Filter
    raw.filter(0.5, 30)

    # First EEG channel
    signal = raw.get_data()[0]

    # Feature extraction
    freqs, psd = welch(signal)

    delta = np.mean(psd[(freqs >= 0.5) & (freqs < 4)])
    theta = np.mean(psd[(freqs >= 4) & (freqs < 8)])
    alpha = np.mean(psd[(freqs >= 8) & (freqs < 13)])
    beta = np.mean(psd[(freqs >= 13) & (freqs < 30)])

    features = [[
        np.mean(signal),
        np.std(signal),
        np.var(signal),
        skew(signal),
        kurtosis(signal),
        delta,
        theta,
        alpha,
        beta
    ]]

    # Scale
    X_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(X_scaled)[0]

    result = sleep_labels.get(int(prediction), "Unknown")

    return {
        "predicted_stage": result
    }