from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import mne
import tempfile

# ============================================================
# CREATE FASTAPI APP
# ============================================================

app = FastAPI()

# ============================================================
# ENABLE CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

model = joblib.load("sleep_model.pkl")
scaler = joblib.load("scaler.pkl")

# ============================================================
# SLEEP LABELS
# ============================================================

sleep_labels = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'Deep',
    4: 'REM'
}

# ============================================================
# HOME ROUTE
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Sleep Detection API Running Successfully"
    }

# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.post("/predict")
async def predict_sleep(file: UploadFile = File(...)):

    try:

        # Save uploaded EDF file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:

            tmp.write(await file.read())

            temp_path = tmp.name

        # ====================================================
        # LOAD EDF FILE
        # ====================================================

        raw = mne.io.read_raw_edf(
            temp_path,
            preload=True,
            verbose=False
        )

        # ====================================================
        # FILTER EEG SIGNAL
        # ====================================================

        raw.filter(
            0.5,
            30,
            verbose=False
        )

        # ====================================================
        # EXTRACT FIRST EEG CHANNEL
        # ====================================================

        signal = raw.get_data()[0]

        # ====================================================
        # FEATURE EXTRACTION
        # ====================================================

        freqs, psd = welch(
            signal,
            fs=raw.info['sfreq']
        )

        delta = np.mean(
            psd[(freqs >= 0.5) & (freqs < 4)]
        )

        theta = np.mean(
            psd[(freqs >= 4) & (freqs < 8)]
        )

        alpha = np.mean(
            psd[(freqs >= 8) & (freqs < 13)]
        )

        beta = np.mean(
            psd[(freqs >= 13) & (freqs < 30)]
        )

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

        # ====================================================
        # SCALE FEATURES
        # ====================================================

        X_scaled = scaler.transform(features)

        # ====================================================
        # PREDICT SLEEP STAGE
        # ====================================================

        prediction = model.predict(X_scaled)[0]

        result = sleep_labels.get(
            int(prediction),
            "Unknown"
        )

        # ====================================================
        # RETURN RESULT
        # ====================================================

        return {
            "predicted_stage": result
        }

    except Exception as e:

        return {
            "error": str(e)
        }

# ============================================================
# RUN SERVER LOCALLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=10000
    )