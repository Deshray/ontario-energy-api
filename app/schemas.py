from pydantic import BaseModel, field_validator
from typing import List, Optional


# ─────────────────────────────────────────────
# Request
# ─────────────────────────────────────────────
class ForecastRequest(BaseModel):
    """
    Supply the last ≥168 hours of demand and price history so that all
    lag/rolling features can be computed correctly.
    """
    recent_demand: List[float]
    recent_price: List[float]
    forecast_horizon: int = 24          # hours (1–168)
    n_simulations: int = 1000           # Monte Carlo paths
    run_monte_carlo: bool = True

    @field_validator("recent_demand", "recent_price")
    @classmethod
    def min_history(cls, v: List[float]) -> List[float]:
        if len(v) < 168:
            raise ValueError(
                "At least 168 hours of history are required "
                "to compute 168-hour lag features."
            )
        return v

    @field_validator("forecast_horizon")
    @classmethod
    def valid_horizon(cls, v: int) -> int:
        if not (1 <= v <= 168):
            raise ValueError("forecast_horizon must be between 1 and 168.")
        return v

    @field_validator("n_simulations")
    @classmethod
    def valid_simulations(cls, v: int) -> int:
        if not (100 <= v <= 5000):
            raise ValueError("n_simulations must be between 100 and 5000.")
        return v


# ─────────────────────────────────────────────
# Response
# ─────────────────────────────────────────────
class IntervalForecast(BaseModel):
    hour: int
    lower_80: float
    median: float
    upper_80: float


class MonteCarloSummary(BaseModel):
    mean_peak: float
    std_peak: float
    peak_95th_percentile: float
    peak_99th_percentile: float
    p_exceed_18000: float
    p_exceed_19000: float
    p_exceed_20000: float
    p_exceed_21000: float
    p_exceed_22000: float


class ForecastResponse(BaseModel):
    horizon_hours: int
    forecast: List[IntervalForecast]
    monte_carlo: Optional[MonteCarloSummary] = None


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
class MetricsResponse(BaseModel):
    linear_regression_mae: Optional[float]
    linear_regression_rmse: Optional[float]
    quantile_median_mae: Optional[float]
    quantile_median_rmse: Optional[float]
    empirical_coverage_80pct: Optional[float]
    note: str = (
        "Metrics are computed on the test split during training "
        "and cached at model load time."
    )