"""
model.py — training, serialization, loading, and inference.

Training:
    python -m app.model --data path/to/ontario_electricity_demand.csv

Inference:
    Called by FastAPI routes via load_models() + predict().
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from app.features import build_features, get_feature_columns

logger = logging.getLogger(__name__)

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
TRAIN_SPLIT_DATE = "2019-01-01"


# ─────────────────────────────────────────────
# Model bundle
# ─────────────────────────────────────────────
@dataclass
class ModelBundle:
    lr: LinearRegression
    qr_low: QuantileRegressor
    qr_med: QuantileRegressor
    qr_high: QuantileRegressor
    scaler: StandardScaler
    feature_columns: List[str]
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)


# ─────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────
def save_models(bundle: ModelBundle, directory: Path = MODELS_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.lr,       directory / "linear_regression.pkl")
    joblib.dump(bundle.qr_low,   directory / "qr_low.pkl")
    joblib.dump(bundle.qr_med,   directory / "qr_med.pkl")
    joblib.dump(bundle.qr_high,  directory / "qr_high.pkl")
    joblib.dump(bundle.scaler,   directory / "scaler.pkl")
    joblib.dump(bundle.feature_columns, directory / "feature_columns.pkl")
    joblib.dump(bundle.metrics,  directory / "metrics.pkl")
    logger.info("Models saved to %s", directory)


def load_models(directory: Path = MODELS_DIR) -> ModelBundle:
    def _load(name: str):
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {path}\n"
                "Run training first:  python -m app.model --data <csv_path>"
            )
        return joblib.load(path)

    bundle = ModelBundle(
        lr=_load("linear_regression.pkl"),
        qr_low=_load("qr_low.pkl"),
        qr_med=_load("qr_med.pkl"),
        qr_high=_load("qr_high.pkl"),
        scaler=_load("scaler.pkl"),
        feature_columns=_load("feature_columns.pkl"),
        metrics=_load("metrics.pkl"),
    )
    logger.info("Models loaded from %s", directory)
    return bundle


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train(csv_path: str) -> ModelBundle:
    logger.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"] - 1, unit="h")
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df[["hourly_demand", "hourly_average_price"]].rename(
        columns={"hourly_demand": "demand_kwh", "hourly_average_price": "price_cents"}
    )

    # Fill gaps
    expected_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(expected_index).interpolate(method="time")
    assert df.isna().sum().sum() == 0

    # Feature engineering
    df = build_features(df)
    df["target_demand_24h"] = df["demand_kwh"].shift(-24)
    df = df.dropna()

    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["target_demand_24h"]

    X_train = X[X.index < TRAIN_SPLIT_DATE]
    X_test  = X[X.index >= TRAIN_SPLIT_DATE]
    y_train = y[y.index < TRAIN_SPLIT_DATE]
    y_test  = y[y.index >= TRAIN_SPLIT_DATE]

    logger.info("Train rows: %d  |  Test rows: %d", len(X_train), len(X_test))

    # ── Linear Regression ──────────────────────────────────────────────────
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_mae  = float(mean_absolute_error(y_test, y_pred_lr))
    lr_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_lr)))
    logger.info("LR  →  MAE: %.1f  RMSE: %.1f", lr_mae, lr_rmse)

    # ── Quantile Regression ────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), index=X_train.index, columns=feature_cols
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), index=X_test.index, columns=feature_cols
    )

    qr_low  = QuantileRegressor(quantile=0.1, alpha=0.1)
    qr_med  = QuantileRegressor(quantile=0.5, alpha=0.1)
    qr_high = QuantileRegressor(quantile=0.9, alpha=0.1)

    for name, model in [("qr_low", qr_low), ("qr_med", qr_med), ("qr_high", qr_high)]:
        logger.info("Fitting %s …", name)
        model.fit(X_train_s, y_train)

    y_low  = qr_low.predict(X_test_s)
    y_med  = qr_med.predict(X_test_s)
    y_high = qr_high.predict(X_test_s)

    qr_mae  = float(mean_absolute_error(y_test, y_med))
    qr_rmse = float(np.sqrt(mean_squared_error(y_test, y_med)))
    coverage = float(np.mean((y_test.values >= y_low) & (y_test.values <= y_high)))
    logger.info("QR  →  MAE: %.1f  RMSE: %.1f  Coverage: %.3f", qr_mae, qr_rmse, coverage)

    metrics = {
        "linear_regression_mae": lr_mae,
        "linear_regression_rmse": lr_rmse,
        "quantile_median_mae": qr_mae,
        "quantile_median_rmse": qr_rmse,
        "empirical_coverage_80pct": coverage,
    }

    bundle = ModelBundle(
        lr=lr, qr_low=qr_low, qr_med=qr_med, qr_high=qr_high,
        scaler=scaler, feature_columns=feature_cols, metrics=metrics,
    )
    save_models(bundle)
    return bundle


# ─────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────
def _build_inference_frame(
    recent_demand: List[float],
    recent_price: List[float],
    horizon: int,
) -> pd.DataFrame:
    """
    Construct a synthetic DataFrame long enough to build all features,
    then return only the last `horizon` rows (the forecast window).
    """
    n = len(recent_demand)
    end   = pd.Timestamp.now().floor("h")
    start = end - pd.Timedelta(hours=n - 1)
    idx   = pd.date_range(start, periods=n, freq="h")

    df = pd.DataFrame(
        {"demand_kwh": recent_demand, "price_cents": recent_price},
        index=idx,
    )
    df = build_features(df)
    df = df.dropna()

    return df.tail(horizon)


def predict(
    bundle: ModelBundle,
    recent_demand: List[float],
    recent_price: List[float],
    forecast_horizon: int = 24,
    n_simulations: int = 1000,
    run_monte_carlo: bool = True,
) -> Tuple[List[dict], Optional[dict]]:
    """
    Returns:
        forecast  — list of per-hour dicts {hour, lower_80, median, upper_80}
        mc_summary — dict of Monte Carlo statistics (or None)
    """
    df_feat = _build_inference_frame(recent_demand, recent_price, forecast_horizon)

    if len(df_feat) == 0:
        raise ValueError(
            "Not enough history to compute features. "
            "Provide at least 168 + forecast_horizon data points."
        )

    X = df_feat[bundle.feature_columns]
    X_scaled = pd.DataFrame(
        bundle.scaler.transform(X), index=X.index, columns=bundle.feature_columns
    )

    y_low  = np.clip(bundle.qr_low.predict(X_scaled),  0, None)
    y_med  = np.clip(bundle.qr_med.predict(X_scaled),  0, None)
    y_high = np.clip(bundle.qr_high.predict(X_scaled), 0, None)

    forecast = [
        {
            "hour": i,
            "lower_80": round(float(y_low[i]),  2),
            "median":   round(float(y_med[i]),  2),
            "upper_80": round(float(y_high[i]), 2),
        }
        for i in range(len(y_med))
    ]

    mc_summary = None
    if run_monte_carlo:
        sigma = np.maximum((y_high - y_low) / 2.56, 1.0)
        np.random.seed(42)
        scenarios = np.random.normal(
            loc=y_med[:, None], scale=sigma[:, None],
            size=(len(y_med), n_simulations),
        )
        scenarios = np.clip(scenarios, 0, None)
        peak_demands = scenarios.max(axis=0)

        mc_summary = {
            "mean_peak":             round(float(peak_demands.mean()), 2),
            "std_peak":              round(float(peak_demands.std()),  2),
            "peak_95th_percentile":  round(float(np.percentile(peak_demands, 95)), 2),
            "peak_99th_percentile":  round(float(np.percentile(peak_demands, 99)), 2),
            "p_exceed_18000":        round(float(np.mean(peak_demands > 18_000_000)), 4),
            "p_exceed_19000":        round(float(np.mean(peak_demands > 19_000_000)), 4),
            "p_exceed_20000":        round(float(np.mean(peak_demands > 20_000_000)), 4),
            "p_exceed_21000":        round(float(np.mean(peak_demands > 21_000_000)), 4),
            "p_exceed_22000":        round(float(np.mean(peak_demands > 22_000_000)), 4),
        }

    return forecast, mc_summary


# ─────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="Train Ontario energy demand models.")
    parser.add_argument("--data", required=True, help="Path to ontario_electricity_demand.csv")
    args = parser.parse_args()
    train(args.data)