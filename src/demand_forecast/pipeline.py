"""Prefect pipeline: monitor a batch, train a challenger, and promote it only if it beats production."""

import argparse

import mlflow
import pandas as pd
from mlflow import MlflowClient
from prefect import flow, task
from prefect.logging import get_run_logger
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from demand_forecast.config import (
    BATCHES_DIR,
    FEATURES,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    PRODUCTION_ALIAS,
    PROJECT_ROOT,
    REFERENCE_DIR,
    TARGET,
)
from demand_forecast.monitor import monitor
from demand_forecast.train import train

RMSE_THRESHOLD = 60.0
DRIFT_THRESHOLD = 0.5


def _rmse_on(model_uri: str, df: pd.DataFrame) -> float:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model(model_uri)
    preds = model.predict(df[FEATURES])
    return float(mean_squared_error(df[TARGET], preds) ** 0.5)


@task
def monitor_batch(batch_name: str) -> dict:
    return monitor(batch_name)


@task
def should_retrain(summary: dict) -> bool:
    logger = get_run_logger()
    rmse = summary["performance"]["rmse"]
    drift = summary.get("drift_share") or 0.0
    decision = rmse > RMSE_THRESHOLD or drift > DRIFT_THRESHOLD
    logger.info(f"{summary['batch']}: rmse={rmse:.1f} drift={drift:.2f} -> retrain={decision}")
    return decision


@task
def retrain_and_maybe_promote(batch_name: str) -> dict:
    logger = get_run_logger()
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    batch = pd.read_csv(BATCHES_DIR / f"{batch_name}.csv")
    batch_train, batch_holdout = train_test_split(batch, test_size=0.3, random_state=42)

    reference = pd.read_csv(REFERENCE_DIR / "reference.csv")
    combined = pd.concat([reference, batch_train], ignore_index=True)
    retrain_csv = PROJECT_ROOT / "data" / "retrain_current.csv"
    combined.to_csv(retrain_csv, index=False)
    logger.info(f"Training challenger on reference + {batch_name} ({len(combined)} rows).")

    train(data_path=retrain_csv, promote=False)

    versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
    newest = max(versions, key=lambda v: int(v.version))
    champion_uri = f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}"
    challenger_uri = f"models:/{MODEL_NAME}/{newest.version}"

    champion_rmse = _rmse_on(champion_uri, batch_holdout)
    challenger_rmse = _rmse_on(challenger_uri, batch_holdout)
    logger.info(f"Holdout RMSE  champion={champion_rmse:.1f}  challenger={challenger_rmse:.1f}")

    promoted = challenger_rmse < champion_rmse
    if promoted:
        client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, newest.version)
        logger.info(f"Promoted v{newest.version} to @{PRODUCTION_ALIAS}.")
    else:
        logger.info("Challenger did not beat champion; keeping current production model.")

    return {
        "challenger_version": newest.version,
        "champion_rmse": champion_rmse,
        "challenger_rmse": challenger_rmse,
        "promoted": promoted,
    }


@flow(name="demand-monitoring")
def monitoring_pipeline(batch_name: str = "2012-10") -> dict:
    logger = get_run_logger()
    summary = monitor_batch(batch_name)
    if should_retrain(summary):
        summary["retrain"] = retrain_and_maybe_promote(batch_name)
    else:
        logger.info("No retrain needed.")
        summary["retrain"] = None
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", nargs="?", default="2012-10")
    parser.add_argument("--serve", action="store_true", help="run on an hourly schedule instead of once")
    args = parser.parse_args()
    if args.serve:
        monitoring_pipeline.serve(
            name="demand-monitoring",
            cron="0 * * * *",
            parameters={"batch_name": args.batch},
        )
    else:
        monitoring_pipeline(batch_name=args.batch)