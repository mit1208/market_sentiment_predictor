"""Macro feature helpers for bronze raw macro history and live inference."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

try:
    from google.cloud import bigquery
except ImportError:  # pragma: no cover - exercised only in lean local test envs
    bigquery = None

from backend.ml.shared.feature_engineering import DEFAULT_MACRO_FEATURES, MACRO_FEATURE_NAMES

try:
    from fredapi import Fred
except ImportError:  # pragma: no cover - exercised only in lean local test envs
    Fred = None

log = logging.getLogger(__name__)

FRED_SERIES = {
    "yield_curve_spread": "T10Y2Y",
    "fed_funds_rate": "FEDFUNDS",
    "credit_spread": "BAMLH0A0HYM2",
    "dxy_level": "DTWEXBGS",
}
RAW_MACRO_SERIES = {series_id: feature_name for feature_name, series_id in FRED_SERIES.items()}
MACRO_CACHE_COLUMNS = [
    "date",
    "yield_curve_spread",
    "fed_funds_rate",
    "fed_funds_change_3m",
    "credit_spread",
    "dxy_change_5d",
]


def _fred(api_key: str) -> Fred:
    if Fred is None:
        raise RuntimeError("fredapi is required for live macro fetches.")
    return Fred(api_key=api_key)


def fetch_macro_history(
    *,
    fred_api_key: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    fred = _fred(fred_api_key)
    frames: list[pd.DataFrame] = []
    for feature_name, series_id in FRED_SERIES.items():
        series = fred.get_series(series_id, observation_start=start_date.date(), observation_end=end_date.date())
        frame = pd.DataFrame({"date": pd.to_datetime(series.index), feature_name: pd.to_numeric(series.values, errors="coerce")})
        frames.append(frame)

    macro = frames[0]
    for frame in frames[1:]:
        macro = macro.merge(frame, on="date", how="outer")
    macro = macro.sort_values("date").reset_index(drop=True)
    macro["fed_funds_change_3m"] = macro["fed_funds_rate"] - macro["fed_funds_rate"].shift(63)
    macro["dxy_change_5d"] = macro["dxy_level"].pct_change(5, fill_method=None)
    return macro[MACRO_CACHE_COLUMNS]


def build_macro_history_from_raw(raw_macro_df: pd.DataFrame) -> pd.DataFrame:
    if raw_macro_df.empty:
        return _empty_macro_history()

    working = raw_macro_df.copy()
    date_column = "observation_date" if "observation_date" in working.columns else "date"
    working["date"] = (
        pd.to_datetime(working[date_column], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    working["series_id"] = working["series_id"].astype(str).str.strip()
    working["value"] = pd.to_numeric(working["value"], errors="coerce")
    working = working.dropna(subset=["date", "series_id", "value"])
    working = working[working["series_id"].isin(RAW_MACRO_SERIES)]
    if working.empty:
        return _empty_macro_history()

    pivoted = (
        working.pivot_table(
            index="date",
            columns="series_id",
            values="value",
            aggfunc="last",
        )
        .rename(columns=RAW_MACRO_SERIES)
        .reset_index()
        .sort_values("date")
    )
    pivoted["date"] = pd.to_datetime(pivoted["date"], errors="coerce").astype("datetime64[ns]")
    for column in RAW_MACRO_SERIES.values():
        if column not in pivoted.columns:
            pivoted[column] = np.nan

    pivoted = pivoted.sort_values("date").reset_index(drop=True)
    daily_index = pd.date_range(pivoted["date"].min(), pivoted["date"].max(), freq="D")
    pivoted = (
        pivoted.set_index("date")
        .reindex(daily_index)
        .sort_index()
        .ffill()
        .rename_axis("date")
        .reset_index()
    )
    pivoted["date"] = pd.to_datetime(pivoted["date"], errors="coerce").astype("datetime64[ns]")
    pivoted["fed_funds_change_3m"] = pivoted["fed_funds_rate"] - pivoted["fed_funds_rate"].shift(90)
    pivoted["dxy_change_5d"] = pivoted["dxy_level"].pct_change(5, fill_method=None)
    return pivoted[MACRO_CACHE_COLUMNS]


def _empty_macro_history() -> pd.DataFrame:
    return pd.DataFrame(columns=MACRO_CACHE_COLUMNS)


def _query_macro_cache(
    client: bigquery.Client,
    *,
    table_name: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    query = f"""
        WITH latest_rows AS (
          SELECT
            snapshot_date AS date,
            yield_curve_spread,
            fed_funds_rate,
            fed_funds_change_3m,
            credit_spread,
            dxy_change_5d,
            ROW_NUMBER() OVER (
              PARTITION BY snapshot_date
              ORDER BY fetched_at DESC
            ) AS row_num
          FROM `{table_name}`
          WHERE snapshot_date BETWEEN @start_date AND @end_date
        )
        SELECT
          date,
          yield_curve_spread,
          fed_funds_rate,
          fed_funds_change_3m,
          credit_spread,
          dxy_change_5d
        FROM latest_rows
        WHERE row_num = 1
        ORDER BY date
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date.date()),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date.date()),
        ]
    )
    frame = client.query(query, job_config=job_config).to_dataframe()
    if frame.empty:
        return _empty_macro_history()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame[MACRO_CACHE_COLUMNS].sort_values("date").reset_index(drop=True)


def _replace_macro_cache_rows(
    client: bigquery.Client,
    *,
    table_name: str,
    frame: pd.DataFrame,
) -> None:
    if frame.empty:
        return

    working = frame.copy()
    working["snapshot_date"] = pd.to_datetime(working["date"]).dt.date
    working["fetched_at"] = datetime.now(timezone.utc)
    rows = [
        {
            "snapshot_date": row.snapshot_date.isoformat(),
            "yield_curve_spread": None if pd.isna(row.yield_curve_spread) else float(row.yield_curve_spread),
            "fed_funds_rate": None if pd.isna(row.fed_funds_rate) else float(row.fed_funds_rate),
            "fed_funds_change_3m": None if pd.isna(row.fed_funds_change_3m) else float(row.fed_funds_change_3m),
            "credit_spread": None if pd.isna(row.credit_spread) else float(row.credit_spread),
            "dxy_change_5d": None if pd.isna(row.dxy_change_5d) else float(row.dxy_change_5d),
            "fetched_at": row.fetched_at.isoformat(),
        }
        for row in working.itertuples(index=False)
    ]

    for start in range(0, len(rows), 500):
        batch = rows[start : start + 500]
        errors = client.insert_rows_json(table_name, batch)
        if errors:
            raise RuntimeError(f"Failed to write macro cache rows: {errors[:3]}")


def _ensure_macro_cache(
    *,
    client: bigquery.Client,
    table_name: str,
    fred_api_key: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    cached = _query_macro_cache(
        client,
        table_name=table_name,
        start_date=start_date,
        end_date=end_date,
    )
    missing_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if cached.empty:
        missing_ranges.append((start_date, end_date))
    else:
        cached_min = pd.Timestamp(cached["date"].min())
        cached_max = pd.Timestamp(cached["date"].max())
        if cached_min > start_date:
            missing_ranges.append((start_date, cached_min - pd.Timedelta(days=1)))
        if cached_max < end_date:
            missing_ranges.append((cached_max + pd.Timedelta(days=1), end_date))

    for missing_start, missing_end in missing_ranges:
        if missing_start > missing_end:
            continue
        fetched = fetch_macro_history(
            fred_api_key=fred_api_key,
            start_date=missing_start,
            end_date=missing_end,
        )
        if not fetched.empty:
            _replace_macro_cache_rows(client, table_name=table_name, frame=fetched)

    return _query_macro_cache(
        client,
        table_name=table_name,
        start_date=start_date,
        end_date=end_date,
    )


def compute_universe_breadth_history(
    market_df: pd.DataFrame,
    *,
    active_tickers: list[str] | None = None,
) -> pd.DataFrame:
    if market_df.empty:
        return pd.DataFrame(columns=["date", "universe_breadth_50d"])

    working = market_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    if active_tickers:
        working = working[working["ticker"].isin(active_tickers)].copy()
    if working.empty:
        return pd.DataFrame(columns=["date", "universe_breadth_50d"])

    frames: list[pd.DataFrame] = []
    for _ticker, group in working.groupby("ticker", sort=False):
        ticker_frame = group.sort_values("date").copy()
        close = ticker_frame["close"].astype(float)
        sma_50 = close.rolling(50, min_periods=50).mean()
        ticker_frame["above_50d"] = (close > sma_50).astype(float)
        frames.append(ticker_frame[["date", "above_50d"]])

    merged = pd.concat(frames, ignore_index=True)
    breadth = merged.groupby("date")["above_50d"].mean().reset_index()
    breadth = breadth.rename(columns={"above_50d": "universe_breadth_50d"})
    return breadth


def build_training_macro_feature_frame(
    dates: pd.Series | list[pd.Timestamp],
    *,
    market_df: pd.DataFrame,
    raw_macro_df: pd.DataFrame,
    active_tickers: list[str] | None = None,
) -> pd.DataFrame:
    date_series = pd.to_datetime(pd.Series(dates), errors="coerce").dt.normalize()
    date_series = date_series.astype("datetime64[ns]")
    if date_series.empty:
        return pd.DataFrame(columns=MACRO_FEATURE_NAMES)

    macro_history = build_macro_history_from_raw(raw_macro_df)
    macro_history["date"] = pd.to_datetime(macro_history["date"], errors="coerce").astype("datetime64[ns]")
    breadth_history = compute_universe_breadth_history(market_df, active_tickers=active_tickers)
    if not breadth_history.empty:
        breadth_history["date"] = pd.to_datetime(breadth_history["date"], errors="coerce").astype("datetime64[ns]")
    macro_history = macro_history.merge(breadth_history, on="date", how="outer").sort_values("date").ffill()

    frame = pd.DataFrame({"date": date_series, "_row_id": np.arange(len(date_series))})
    valid = frame[frame["date"].notna()].sort_values("date")
    merged = pd.merge_asof(valid, macro_history.sort_values("date"), on="date", direction="backward")
    result = frame.merge(merged[["_row_id", *MACRO_FEATURE_NAMES]], on="_row_id", how="left").sort_values("_row_id")
    for feature_name, default_value in DEFAULT_MACRO_FEATURES.items():
        result[feature_name] = result[feature_name].fillna(default_value)
    return result[MACRO_FEATURE_NAMES].reset_index(drop=True)


def fetch_live_macro_features(
    *,
    fred_api_key: str,
    universe_breadth_50d: float,
    cache_client: bigquery.Client | None = None,
    cache_table_name: str | None = None,
    as_of_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    target_date = pd.Timestamp(as_of_date or pd.Timestamp.utcnow()).tz_localize(None).normalize()
    start_date = target_date - pd.Timedelta(days=120)
    end_date = target_date + pd.Timedelta(days=1)

    if cache_client is not None and cache_table_name:
        history = _ensure_macro_cache(
            client=cache_client,
            table_name=cache_table_name,
            fred_api_key=fred_api_key,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        history = fetch_macro_history(
            fred_api_key=fred_api_key,
            start_date=start_date,
            end_date=end_date,
        )

    latest_frame = history[history["date"] <= target_date].sort_values("date")
    latest = latest_frame.iloc[-1] if not latest_frame.empty else None
    if latest is None:
        values = dict(DEFAULT_MACRO_FEATURES)
    else:
        values = {
            "yield_curve_spread": float(latest["yield_curve_spread"]),
            "fed_funds_rate": float(latest["fed_funds_rate"]),
            "fed_funds_change_3m": float(latest["fed_funds_change_3m"]),
            "credit_spread": float(latest["credit_spread"]),
            "dxy_change_5d": float(latest["dxy_change_5d"]),
            "universe_breadth_50d": float(universe_breadth_50d),
        }
    values["_sources"] = {
        "yield_curve_spread": "bigquery_cache" if cache_client is not None and cache_table_name else "fred",
        "fed_funds_rate": "bigquery_cache" if cache_client is not None and cache_table_name else "fred",
        "fed_funds_change_3m": "bigquery_cache" if cache_client is not None and cache_table_name else "fred",
        "credit_spread": "bigquery_cache" if cache_client is not None and cache_table_name else "fred",
        "dxy_change_5d": "bigquery_cache" if cache_client is not None and cache_table_name else "fred",
        "universe_breadth_50d": "bigquery",
    }
    return values


def load_latest_universe_breadth(
    client: bigquery.Client,
    *,
    table_name: str,
    as_of_date: pd.Timestamp,
) -> float:
    query = f"""
        SELECT universe_breadth_50d
        FROM `{table_name}`
        WHERE snapshot_date <= @snapshot_date
          AND universe_breadth_50d IS NOT NULL
        ORDER BY snapshot_date DESC
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("snapshot_date", "DATE", as_of_date.date())]
    )
    rows = list(client.query(query, job_config=job_config).result())
    if not rows:
        return DEFAULT_MACRO_FEATURES["universe_breadth_50d"]
    return float(rows[0].universe_breadth_50d)
