"""Train a LightGBM demand forecaster, track it with MLflow, and register + promote it."""

import argparse

import mlflow
import mlflow.lightgbm
import pandas as pd
from lightgbm import LGBMRegressor
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from demand_forecast.config import (
    CATEGORICAL,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    PRODUCTION_ALIAS,
    REFERENCE_DIR,
)
from demand_forecast.features import make_xy

PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def train(data_path=None, promote: bool = True) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("demand-forecasting")

    path = data_path or (REFERENCE_DIR / "reference.csv")
    df = pd.read_csv(path)
    X, y = make_xy(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        model = LGBMRegressor(**PARAMS)
        model.fit(X_train, y_train, categorical_feature=CATEGORICAL)

        preds = model.predict(X_val)
        metrics = {
            "mae": mean_absolute_error(y_val, preds),
            "rmse": mean_squared_error(y_val, preds) ** 0.5,
            "r2": r2_score(y_val, preds),
        }

        mlflow.log_params(PARAMS)
        mlflow.log_metrics(metrics)
        signature = infer_signature(X_val, preds)
        info = mlflow.lightgbm.log_model(
            model, name="model", signature=signature, registered_model_name=MODEL_NAME
        )

        print(f"run_id={run.info.run_id}")
        print(f"MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  R2={metrics['r2']:.3f}")
        print(f"registered: {info.model_uri}")

    if promote:
        _promote_if_better(metrics["rmse"])
    return metrics


def _promote_if_better(rmse: float) -> None:
    """Point the 'production' alias at the newest version if it beats the current one."""
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
    newest = max(versions, key=lambda v: int(v.version))

    try:
        current = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    except MlflowException:
        current = None
        
    if current is None:
        client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, newest.version)
        print(f"Promoted v{newest.version} to @{PRODUCTION_ALIAS} (first model).")
        return

    current_rmse = client.get_run(current.run_id).data.metrics.get("rmse", float("inf"))
    if rmse <= current_rmse:
        client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, newest.version)
        print(f"Promoted v{newest.version} @{PRODUCTION_ALIAS} (rmse {rmse:.2f} <= {current_rmse:.2f}).")
    else:
        print(f"Kept v{current.version} @{PRODUCTION_ALIAS} (new {rmse:.2f} > {current_rmse:.2f}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-promote", action="store_true")
    train(promote=not parser.parse_args().no_promote)