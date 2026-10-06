"""FastAPI service: serve the production demand-forecasting model."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from demand_forecast.predict import predict_one

app = FastAPI(title="demand-forecast-mlops", version="0.1.0")

_INDEX = Path(__file__).resolve().parents[2] / "static" / "index.html"


class RideHour(BaseModel):
    season: int = Field(..., ge=1, le=4)
    yr: int = Field(..., ge=0, le=1)
    mnth: int = Field(..., ge=1, le=12)
    hr: int = Field(..., ge=0, le=23)
    holiday: int = Field(..., ge=0, le=1)
    weekday: int = Field(..., ge=0, le=6)
    workingday: int = Field(..., ge=0, le=1)
    weathersit: int = Field(..., ge=1, le=4)
    temp: float = Field(..., ge=0, le=1)
    atemp: float = Field(..., ge=0, le=1)
    hum: float = Field(..., ge=0, le=1)
    windspeed: float = Field(..., ge=0, le=1)


class Prediction(BaseModel):
    predicted_count: float


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    if _INDEX.exists():
        return HTMLResponse(_INDEX.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>demand-forecast-mlops</h1><p>POST /predict</p>")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(ride: RideHour) -> Prediction:
    return Prediction(predicted_count=round(predict_one(ride.model_dump()), 1))