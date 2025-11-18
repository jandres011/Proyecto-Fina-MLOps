import os
import json
import logging
from typing import List

import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_URI = os.getenv("MODEL_URI", "models:/ml-classic-wine-classification/latest")

app = FastAPI(title="ML Classic Service", version="2.0.0")

model = None
FEATURE_NAMES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"
]

def load_model_from_mlflow():
    global model
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"Cargando modelo desde MLflow: {MODEL_URI}")
        model = mlflow.sklearn.load_model(MODEL_URI)
        logger.info("Modelo cargado correctamente desde MLflow.")
    except Exception as e:
        logger.error(f"Error cargando modelo desde MLflow: {e}")
        model = None

load_model_from_mlflow()

class PredictionRequest(BaseModel):
    features: List[float]


class PredictionResponse(BaseModel):
    prediction: int | None
    probabilities: List[float] | None
    feature_names: List[str]
    error: str | None = None

@app.get("/health")
def health():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    if model is None:
        logger.error("Modelo no cargado desde MLflow.")
        return PredictionResponse(
            prediction=None,
            probabilities=None,
            feature_names=FEATURE_NAMES,
            error="Modelo no cargado. Verifica que MLflow esté corriendo y que el modelo exista."
        )

    if len(request.features) != len(FEATURE_NAMES):
        return PredictionResponse(
            prediction=None,
            probabilities=None,
            feature_names=FEATURE_NAMES,
            error=f"Se esperaban {len(FEATURE_NAMES)} características."
        )

    try:
        logger.info("Realizando predicción...")

        pred = model.predict([request.features])[0]
        probs = model.predict_proba([request.features])[0].tolist()

        logger.info(f"Predicción OK: clase={pred}, prob_max={max(probs):.4f}")

        return PredictionResponse(
            prediction=int(pred),
            probabilities=probs,
            feature_names=FEATURE_NAMES
        )

    except Exception as e:
        logger.error(f"Error durante predicción: {e}")
        return PredictionResponse(
            prediction=None,
            probabilities=None,
            feature_names=FEATURE_NAMES,
            error=str(e)
        )
