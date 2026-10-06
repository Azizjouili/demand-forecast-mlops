FROM python:3.11-slim

ENV UV_HTTP_TIMEOUT=300 \
    UV_CONCURRENT_DOWNLOADS=2 \
    PYTHONUNBUFFERED=1 \
    DO_NOT_TRACK=1

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

# Dependency layer (cached across code changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project --frozen || uv sync --no-dev --no-install-project

# Project code
COPY README.md ./
COPY src ./src
COPY static ./static
RUN uv sync --no-dev --frozen || uv sync --no-dev

EXPOSE 8000
CMD [".venv/bin/uvicorn", "demand_forecast.api:app", "--host", "0.0.0.0", "--port", "8000"]