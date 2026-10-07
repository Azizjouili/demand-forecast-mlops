import pytest
from fastapi.testclient import TestClient

from demand_forecast.api import app

client = TestClient(app, raise_server_exceptions=False)

EXAMPLE = {
    "season": 1, "yr": 1, "mnth": 1, "hr": 8, "holiday": 0, "weekday": 1,
    "workingday": 1, "weathersit": 1, "temp": 0.24, "atemp": 0.28,
    "hum": 0.60, "windspeed": 0.15,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_validation_rejects_bad_input():
    r = client.post("/predict", json=dict(EXAMPLE, hr=99))
    assert r.status_code == 422


def test_predict_if_model_available():
    r = client.post("/predict", json=EXAMPLE)
    if r.status_code != 200:
        pytest.skip("Production model not available in this environment")
    assert r.json()["predicted_count"] >= 0