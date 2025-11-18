import os
import json
import logging
from typing import Any, Dict

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.sklearn

from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "ml-classic-wine-classification"


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def load_data():
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y


def train_model() -> Dict[str, Any]:
    logger.info("Cargando datos...")
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info("Creando Pipeline sklearn (Scaler + RandomForest)...")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100, max_depth=10, min_samples_split=5, random_state=42
                ),
            ),
        ]
    )

    logger.info("Entrenando modelo...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    logger.info(f"Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")

    setup_mlflow()

    with mlflow.start_run(run_name="wine-classification-run"):
        mlflow.log_params(
            {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "random_state": 42,
            }
        )

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="ml-classic-wine-classification",
        )

        report = classification_report(y_test, y_pred, output_dict=True)
        with open("classification_report.json", "w") as f:
            json.dump(report, f, indent=2)

        mlflow.log_artifact("classification_report.json", artifact_path="metrics")

    logger.info("Pipeline y métricas registradas en MLflow correctamente.")

    return {
        "pipeline": pipeline,
        "accuracy": accuracy,
        "f1_score": f1,
        "feature_names": list(X.columns),
    }
