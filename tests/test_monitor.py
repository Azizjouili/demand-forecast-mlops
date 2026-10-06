from demand_forecast.monitor import _drift_share


def test_drift_share_extracts_value():
    d = {"metrics": [{"metric_id": "DriftedColumnsCount", "value": {"count": 3, "share": 0.25}}]}
    assert _drift_share(d) == 0.25


def test_drift_share_handles_missing():
    assert _drift_share({"metrics": []}) is None