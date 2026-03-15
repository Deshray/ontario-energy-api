"""
tests/test_api.py

Run with:
    pytest tests/ -v

NOTE: These tests use a mock model bundle so no CSV data or training is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app, _state
from app.model import ModelBundle
from app.schemas import ForecastResponse, HealthResponse


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────
HISTORY_LEN = 200  # ≥168 required by validator
FAKE_DEMAND = [15_000_000 + np.random.normal(0, 500_000) for _ in range(HISTORY_LEN)]
FAKE_PRICE  = [8.0 + np.random.normal(0, 1.0) for _ in range(HISTORY_LEN)]


def _make_mock_bundle() -> ModelBundle:
    """Build a ModelBundle where every sklearn model is mocked."""
    def _predict_fn(X):
        return np.full(len(X), 15_000_000.0)

    lr = MagicMock()
    lr.predict.side_effect = _predict_fn
    lr.coef_ = np.zeros(18)

    qr_low = MagicMock()
    qr_low.predict.side_effect = lambda X: np.full(len(X), 14_000_000.0)

    qr_med = MagicMock()
    qr_med.predict.side_effect = _predict_fn

    qr_high = MagicMock()
    qr_high.predict.side_effect = lambda X: np.full(len(X), 16_000_000.0)

    scaler = MagicMock()
    scaler.transform.side_effect = lambda X: X  # identity

    from app.features import get_feature_columns
    return ModelBundle(
        lr=lr,
        qr_low=qr_low,
        qr_med=qr_med,
        qr_high=qr_high,
        scaler=scaler,
        feature_columns=get_feature_columns(),
        metrics={
            "linear_regression_mae": 123.4,
            "linear_regression_rmse": 200.1,
            "quantile_median_mae": 115.0,
            "quantile_median_rmse": 190.5,
            "empirical_coverage_80pct": 0.812,
        },
    )


@pytest.fixture(autouse=True)
def inject_mock_bundle():
    """Inject mock bundle into app state before each test."""
    _state["bundle"] = _make_mock_bundle()
    _state["loaded"] = True
    yield
    _state["bundle"] = None
    _state["loaded"] = False


@pytest.fixture
def client():
    return TestClient(app)


# ─────────────────────────────────────────────
# Health & Metrics
# ─────────────────────────────────────────────
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True


def test_metrics_ok(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["linear_regression_mae"] == pytest.approx(123.4)
    assert body["empirical_coverage_80pct"] == pytest.approx(0.812)


# ─────────────────────────────────────────────
# Forecast endpoint — happy paths
# ─────────────────────────────────────────────
def test_forecast_default_horizon(client):
    payload = {
        "recent_demand": FAKE_DEMAND,
        "recent_price": FAKE_PRICE,
    }
    r = client.post("/forecast", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["horizon_hours"] == 24
    assert len(body["forecast"]) == 24
    assert body["monte_carlo"] is not None


def test_forecast_custom_horizon(client):
    payload = {
        "recent_demand": FAKE_DEMAND,
        "recent_price": FAKE_PRICE,
        "forecast_horizon": 48,
    }
    r = client.post("/forecast", json=payload)
    assert r.status_code == 200
    assert r.json()["horizon_hours"] == 48
    assert len(r.json()["forecast"]) == 48


def test_forecast_no_monte_carlo(client):
    payload = {
        "recent_demand": FAKE_DEMAND,
        "recent_price": FAKE_PRICE,
        "run_monte_carlo": False,
    }
    r = client.post("/forecast", json=payload)
    assert r.status_code == 200
    assert r.json()["monte_carlo"] is None


def test_forecast_interval_ordering(client):
    """lower_80 ≤ median ≤ upper_80 for all hours."""
    payload = {"recent_demand": FAKE_DEMAND, "recent_price": FAKE_PRICE}
    body = client.post("/forecast", json=payload).json()
    for h in body["forecast"]:
        assert h["lower_80"] <= h["median"] <= h["upper_80"]


# ─────────────────────────────────────────────
# Forecast endpoint — validation errors
# ─────────────────────────────────────────────
def test_forecast_too_short_history(client):
    payload = {
        "recent_demand": [15_000_000.0] * 50,   # < 168
        "recent_price":  [8.0] * 50,
    }
    r = client.post("/forecast", json=payload)
    assert r.status_code == 422


def test_forecast_invalid_horizon(client):
    payload = {
        "recent_demand": FAKE_DEMAND,
        "recent_price": FAKE_PRICE,
        "forecast_horizon": 200,   # > 168
    }
    r = client.post("/forecast", json=payload)
    assert r.status_code == 422


def test_forecast_invalid_simulations(client):
    payload = {
        "recent_demand": FAKE_DEMAND,
        "recent_price": FAKE_PRICE,
        "n_simulations": 9999,   # > 5000
    }
    r = client.post("/forecast", json=payload)
    assert r.status_code == 422


# ─────────────────────────────────────────────
# 503 when models not loaded
# ─────────────────────────────────────────────
def test_forecast_503_when_no_models(client):
    _state["bundle"] = None
    _state["loaded"] = False
    payload = {"recent_demand": FAKE_DEMAND, "recent_price": FAKE_PRICE}
    r = client.post("/forecast", json=payload)
    assert r.status_code == 503