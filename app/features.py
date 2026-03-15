import pandas as pd
import numpy as np
from typing import List


# ─────────────────────────────────────────────
# Holiday calendar (Ontario statutory holidays)
# ─────────────────────────────────────────────
_EASTER_DATES = {
    2020: "04-10", 2021: "04-02", 2022: "04-15",
    2023: "04-07", 2024: "03-29", 2025: "04-18",
    2026: "04-03", 2027: "03-26", 2028: "04-14",
}


def build_ontario_holidays(start_year: int, end_year: int) -> List[pd.Timestamp]:
    holidays = []
    for year in range(start_year, end_year + 1):
        # Fixed-date holidays
        holidays.extend([
            pd.Timestamp(f"{year}-01-01"),  # New Year's Day
            pd.Timestamp(f"{year}-07-01"),  # Canada Day
            pd.Timestamp(f"{year}-12-25"),  # Christmas
            pd.Timestamp(f"{year}-12-26"),  # Boxing Day
        ])

        # Family Day — 3rd Monday in February
        feb_mondays = pd.date_range(f"{year}-02-01", f"{year}-02-28", freq="W-MON")
        if len(feb_mondays) >= 3:
            holidays.append(feb_mondays[2])

        # Good Friday
        if year in _EASTER_DATES:
            holidays.append(pd.Timestamp(f"{year}-{_EASTER_DATES[year]}"))

        # Victoria Day — Monday before May 25
        may_24 = pd.Timestamp(f"{year}-05-24")
        holidays.append(may_24 - pd.Timedelta(days=(may_24.dayofweek + 1) % 7))

        # Civic Holiday — 1st Monday in August
        aug = pd.date_range(f"{year}-08-01", f"{year}-08-07", freq="W-MON")
        if len(aug) >= 1:
            holidays.append(aug[0])

        # Labour Day — 1st Monday in September
        sep = pd.date_range(f"{year}-09-01", f"{year}-09-07", freq="W-MON")
        if len(sep) >= 1:
            holidays.append(sep[0])

        # Thanksgiving — 2nd Monday in October
        octo = pd.date_range(f"{year}-10-01", f"{year}-10-31", freq="W-MON")
        if len(octo) >= 2:
            holidays.append(octo[1])

    return list(set(h.normalize() for h in holidays))


# ─────────────────────────────────────────────
# Core feature builder
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects a DataFrame with a DatetimeIndex and columns:
        demand_kwh, price_cents
    Returns a feature-engineered DataFrame ready for model input.
    Drops rows where lag/rolling features produce NaN (first 168 rows).
    """
    df = df.copy()

    # ── Temporal features ──────────────────────────────────────────────────
    df["hour"]       = df.index.hour
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek"]  = df.index.dayofweek
    df["month"]      = df.index.month
    df["dayofyear"]  = df.index.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # ── Holiday flag ───────────────────────────────────────────────────────
    start_year = df.index.min().year
    end_year   = df.index.max().year
    holidays   = build_ontario_holidays(start_year, end_year)
    df["is_holiday"] = df.index.normalize().isin(holidays).astype(int)

    # ── Demand lags & rolling stats ────────────────────────────────────────
    for lag in [1, 24, 168]:
        df[f"demand_lag_{lag}"] = df["demand_kwh"].shift(lag)

    df["rolling_mean_24"]  = df["demand_kwh"].rolling(24).mean()
    df["rolling_std_24"]   = df["demand_kwh"].rolling(24).std()
    df["rolling_mean_168"] = df["demand_kwh"].rolling(168).mean()
    df["rolling_std_168"]  = df["demand_kwh"].rolling(168).std()

    # ── Price lags & rolling stats ─────────────────────────────────────────
    df["price_lag_24"]           = df["price_cents"].shift(24)
    df["price_lag_168"]          = df["price_cents"].shift(168)
    df["price_rolling_mean_168"] = df["price_cents"].rolling(168).mean()
    df["price_rolling_std_168"]  = df["price_cents"].rolling(168).std()

    return df


def get_feature_columns() -> List[str]:
    """Returns the ordered list of feature columns expected by the model."""
    return [
        "hour_sin", "hour_cos",
        "dayofweek", "month", "dayofyear",
        "is_weekend", "is_holiday",
        "demand_lag_1", "demand_lag_24", "demand_lag_168",
        "rolling_mean_24", "rolling_std_24",
        "rolling_mean_168", "rolling_std_168",
        "price_lag_24", "price_lag_168",
        "price_rolling_mean_168", "price_rolling_std_168",
    ]