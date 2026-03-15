"""
scripts/fetch_ieso.py

Downloads the current and prior year hourly Ontario demand data from IESO's
public reports server, normalises it to match the training data format, and
writes it to data/ontario_electricity_demand.csv.

Run manually:
    python scripts/fetch_ieso.py

Run automatically:
    Called by .github/workflows/update_data.yml every day at 8am UTC.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BASE_URL  = "https://reports-public.ieso.ca/public/Demand/PUB_Demand_{year}.csv"
OUT_PATH  = Path(__file__).parent.parent / "data" / "ontario_electricity_demand.csv"
MIN_ROWS  = 168   # minimum hours needed downstream


def fetch_year(year: int) -> pd.DataFrame | None:
    url = BASE_URL.format(year=year)
    logger.info("Fetching %s", url)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Could not fetch %d: %s", year, exc)
        return None

    try:
        # IESO demand CSV has 3 header rows then: Date, Hour, Ontario Demand, Market Demand
        df = pd.read_csv(
            StringIO(r.text),
            skiprows=3,
            header=None,
            names=["date", "hour", "ontario_demand", "market_demand"],
        )
        df = df.dropna(subset=["date", "hour", "ontario_demand"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int64")
        df["ontario_demand"] = pd.to_numeric(df["ontario_demand"], errors="coerce")
        df = df.dropna(subset=["date", "hour", "ontario_demand"])
        logger.info("  → %d rows for %d", len(df), year)
        return df
    except Exception as exc:
        logger.warning("Could not parse %d: %s", year, exc)
        return None


def build_output(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenates yearly frames and normalises to the training data schema:
        date, hour, hourly_demand, hourly_average_price

    IESO reports demand in MW — multiply by 1000 to match the kWh scale
    used during model training.

    Price is not available in the demand report; we fill with a neutral
    placeholder of 8.0 cents/kWh so the price-based lag features still
    compute without errors.
    """
    df = pd.concat(frames, ignore_index=True)

    # Deduplicate (overlapping years may have duplicate rows)
    df = df.drop_duplicates(subset=["date", "hour"])
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)

    out = pd.DataFrame({
        "date":                  df["date"].dt.strftime("%Y-%m-%d"),
        "hour":                  df["hour"],
        "hourly_demand":         (df["ontario_demand"] * 1000).round(0),
        "hourly_average_price":  8.0,
    })

    return out


def main() -> None:
    today = datetime.utcnow()
    years = sorted({today.year - 1, today.year})

    frames = []
    for year in years:
        df = fetch_year(year)
        if df is not None:
            frames.append(df)

    if not frames:
        logger.error("No data fetched from IESO. Exiting without writing.")
        sys.exit(1)

    out = build_output(frames)

    if len(out) < MIN_ROWS:
        logger.error(
            "Only %d rows fetched — need at least %d. Exiting without writing.",
            len(out), MIN_ROWS,
        )
        sys.exit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    logger.info(
        "Written %d rows (%s → %s) to %s",
        len(out),
        out["date"].iloc[0],
        out["date"].iloc[-1],
        OUT_PATH,
    )


if __name__ == "__main__":
    main()