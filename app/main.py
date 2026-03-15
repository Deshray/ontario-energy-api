"""
main.py — FastAPI application entry point.

Run locally:
    uvicorn app.main:app --reload

API docs available at:
    http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.model import ModelBundle, load_models, predict
from app.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    IntervalForecast,
    MetricsResponse,
    MonteCarloSummary,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ─────────────────────────────────────────────
# Application state
# ─────────────────────────────────────────────
_state: Dict[str, Any] = {"bundle": None, "loaded": False}

APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup; release on shutdown."""
    try:
        _state["bundle"] = load_models()
        _state["loaded"] = True
        logger.info("Model bundle loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning("Could not load models at startup: %s", exc)
        _state["loaded"] = False
    yield
    _state["bundle"] = None
    _state["loaded"] = False


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Ontario Electricity Demand Forecast API",
    description=(
        "Probabilistic 24-hour ahead electricity demand forecasting for Ontario. "
        "Provides prediction intervals (10th / 50th / 90th quantile) and optional "
        "Monte Carlo risk simulation."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Check server logs."},
    )


# ─────────────────────────────────────────────
# Dependency helper
# ─────────────────────────────────────────────
def _get_bundle() -> ModelBundle:
    if not _state["loaded"] or _state["bundle"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Models are not loaded. "
                "Train first: python -m app.model --data <csv_path>"
            ),
        )
    return _state["bundle"]


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Liveness check — always returns 200 if the server is running."""
    return HealthResponse(
        status="ok",
        models_loaded=_state["loaded"],
        version=APP_VERSION,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
def metrics() -> MetricsResponse:
    """Returns held-out test-set evaluation metrics from the last training run."""
    bundle = _get_bundle()
    m = bundle.metrics
    return MetricsResponse(
        linear_regression_mae=m.get("linear_regression_mae"),
        linear_regression_rmse=m.get("linear_regression_rmse"),
        quantile_median_mae=m.get("quantile_median_mae"),
        quantile_median_rmse=m.get("quantile_median_rmse"),
        empirical_coverage_80pct=m.get("empirical_coverage_80pct"),
    )


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
def forecast(req: ForecastRequest) -> ForecastResponse:
    """
    Generate a probabilistic demand forecast.

    **Inputs:**
    - `recent_demand` — list of ≥168 hourly demand values (kWh)
    - `recent_price`  — list of ≥168 hourly price values (cents/kWh)
    - `forecast_horizon` — hours ahead to forecast (1–168, default 24)
    - `run_monte_carlo`  — include Monte Carlo peak-risk summary (default true)
    - `n_simulations`    — number of MC paths (100–5000, default 1000)

    **Returns:**
    - Per-hour lower / median / upper prediction intervals
    - Optional Monte Carlo peak demand risk statistics
    """
    bundle = _get_bundle()

    try:
        raw_forecast, raw_mc = predict(
            bundle=bundle,
            recent_demand=req.recent_demand,
            recent_price=req.recent_price,
            forecast_horizon=req.forecast_horizon,
            n_simulations=req.n_simulations,
            run_monte_carlo=req.run_monte_carlo,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(exc)) from exc

    return ForecastResponse(
        horizon_hours=req.forecast_horizon,
        forecast=[IntervalForecast(**h) for h in raw_forecast],
        monte_carlo=MonteCarloSummary(**raw_mc) if raw_mc else None,
    )