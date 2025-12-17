from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import mlflow
import os

from app.schemas import PredictionResponse

app = FastAPI(
    title="Anomaly Detection API",
    description="Detect anomalies in cybersecurity network logs",
    version="1.0.0",
)

# model configs
MODEL_NAME = "AnomalyDetector"
MODEL_ALIAS = "production"
ARTIFACT_PATH = "artifacts/deployable_model.joblib"

model = None
preprocessor = None


#startup
@app.on_event("startup")
def load_model():
    global model, preprocessor

    print("Loading model from MLflow Registry...")
    model = mlflow.pyfunc.load_model(
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    print("Loading preprocessor artifact...")
    artifact = joblib.load(ARTIFACT_PATH)
    preprocessor = artifact["preprocessor"]

    print("Model and preprocessor loaded successfully.")


# health check
@app.get("/health")
def health():
    return {"status": "ok"}


# prediction
@app.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)):

    df = pd.read_csv(file.file)

    # Drop non-feature columns if present
    df = df.drop(columns=["attack_cat", "id"], errors="ignore")

    #fixing categorical column
    df['service'] = df['service'].replace('-', 'unknown', regex=True)

    X = df.drop(columns=["label"], errors="ignore")

    X_transformed = preprocessor.transform(X)

    preds = model.predict(X_transformed)

    anomalies = (preds == -1).sum()
    normals = (preds == 1).sum()

    return PredictionResponse(
        total_records=len(df),
        anomalies_detected=int(anomalies),
        normal_records=int(normals),
    )
