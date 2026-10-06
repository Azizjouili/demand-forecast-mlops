"""Load the production model from the MLflow registry and predict demand."""

import mlflow
import pandas as pd

from demand_forecast.config import (
    FEATURES,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    PRODUCTION_ALIAS,
)

_model = None


def _load():
    global _model
    if _model is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        _model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}")
    return _model


def predict_one(record: dict) -> float:
    model = _load()
    X = pd.DataFrame([record])[FEATURES]
    return float(model.predict(X)[0])

def predict_frame(df: pd.DataFrame):
    model = _load()
    return model.predict(df[FEATURES])