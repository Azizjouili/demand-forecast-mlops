"""Monitor an incoming batch: data drift (Evidently) + live model performance."""

import argparse
import json

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from demand_forecast.config import (
    BATCHES_DIR,
    CATEGORICAL,
    FEATURES,
    NUMERIC,
    PROJECT_ROOT,
    REFERENCE_DIR,
    TARGET,
)
from demand_forecast.predict import predict_frame

REPORTS_DIR = PROJECT_ROOT / "reports"


def _drift_share(result_dict: dict) -> float | None:
    """Pull the share of drifted columns out of the Evidently result, defensively."""
    for metric in result_dict.get("metrics", []):
        name = str(metric.get("metric_id") or metric.get("id") or "")
        value = metric.get("value")
        if "DriftedColumnsCount" in name and isinstance(value, dict):
            return value.get("share")
    for metric in result_dict.get("metrics", []):
        value = metric.get("value")
        if isinstance(value, dict) and "share" in value and "count" in value:
            return value.get("share")
    return None


def monitor(batch_name: str) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(REFERENCE_DIR / "reference.csv")
    current = pd.read_csv(BATCHES_DIR / f"{batch_name}.csv")

    cols = FEATURES + [TARGET]
    schema = DataDefinition(
        numerical_columns=NUMERIC + [TARGET],
        categorical_columns=CATEGORICAL,
    )
    ref_ds = Dataset.from_pandas(reference[cols], data_definition=schema)
    cur_ds = Dataset.from_pandas(current[cols], data_definition=schema)

    report = Report([DataDriftPreset()])
    result = report.run(current_data=cur_ds, reference_data=ref_ds)

    html_path = REPORTS_DIR / f"drift_{batch_name}.html"
    result.save_html(str(html_path))
    drift_share = _drift_share(result.dict())

    # Live performance vs ground truth (batches are historical, so actuals exist).
    preds = predict_frame(current)
    performance = {
        "mae": float(mean_absolute_error(current[TARGET], preds)),
        "rmse": float(mean_squared_error(current[TARGET], preds) ** 0.5),
        "r2": float(r2_score(current[TARGET], preds)),
    }

    summary = {
        "batch": batch_name,
        "rows": len(current),
        "drift_share": drift_share,
        "performance": performance,
        "report_html": str(html_path),
    }
    (REPORTS_DIR / f"summary_{batch_name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", help="batch name, e.g. 2012-10")
    monitor(parser.parse_args().batch)