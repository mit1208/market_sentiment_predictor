"""Validation helpers for historical market context coverage."""

from __future__ import annotations

import pandas as pd

CONTEXT_HISTORY_START_TOLERANCE_DAYS = 7


def validate_context_history_coverage(
    training_market_df: pd.DataFrame,
    *,
    spy_history: pd.DataFrame,
    vix_history: pd.DataFrame,
) -> None:
    required_start = pd.Timestamp(training_market_df["date"].min()).normalize()
    required_end = pd.Timestamp(training_market_df["date"].max()).normalize()
    tolerated_start = required_start + pd.Timedelta(days=CONTEXT_HISTORY_START_TOLERANCE_DAYS)
    tolerated_end = required_end - pd.Timedelta(days=CONTEXT_HISTORY_START_TOLERANCE_DAYS)

    for ticker, history in {"SPY": spy_history, "^VIX": vix_history}.items():
        history_dates = (
            pd.to_datetime(history["date"], errors="coerce")
            .dropna()
            .dt.normalize()
            .drop_duplicates()
            .sort_values()
        )
        if history_dates.empty:
            raise RuntimeError(
                f"Bronze raw_market_data has no historical rows for required context ticker {ticker}. "
                "Run ingestion to backfill full SPY/^VIX market history before feature ranking."
            )

        min_date = pd.Timestamp(history_dates.min()).normalize()
        max_date = pd.Timestamp(history_dates.max()).normalize()
        if min_date > tolerated_start or max_date < tolerated_end:
            raise RuntimeError(
                "Bronze raw_market_data context coverage is incomplete for "
                f"{ticker}. Found {len(history_dates)} context dates spanning {min_date.date()} to {max_date.date()}, "
                f"but training history spans {required_start.date()} to {required_end.date()}. "
                "Run ingestion to backfill full SPY/^VIX market history before feature ranking."
            )
