"""Central config: paths, dataset URL, feature definitions, and MLflow settings."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
BATCHES_DIR = DATA_DIR / "batches"

DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
RAW_FILE = RAW_DIR / "hour.csv"

TARGET = "cnt"
# Dropped: identifiers + leakage (casual + registered == cnt).
DROP_COLS = ["instant", "dteday", "casual", "registered"]
CATEGORICAL = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
NUMERIC = ["temp", "atemp", "hum", "windspeed"]
FEATURES = CATEGORICAL + NUMERIC

# MLflow: SQLite backend enables the model registry locally.
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)
MODEL_NAME = "demand-forecaster"
PRODUCTION_ALIAS = "production"