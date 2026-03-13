"""Helpers for persisting live feature snapshots to BigQuery."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from shared.feature_engineering import BASE_FEATURE_NAMES

SNAPSHOT_SOURCE_HISTORICAL = "historical_backfill"
SNAPSHOT_SOURCE_ENDPOINT = "endpoint_live_api"
SNAPSHOT_SOURCE_BATCH = "batch_live_api"


def build_snapshot_id(prefix: str, ticker: str, snapshot_timestamp: Any) -> str:
    timestamp = pd.Timestamp(snapshot_timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return f"{prefix}:{ticker}:{timestamp.strftime('%Y%m%dT%H%M%SZ')}"


def _serialize_feature_snapshot(feature_snapshot: dict | None) -> str | None:
    if not feature_snapshot:
        return None
    return json.dumps(feature_snapshot, separators=(",", ":"), default=str)


def _coerce_json_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def build_live_feature_snapshot_row(
    *,
    snapshot_id: str,
    ticker: str,
    snapshot_timestamp: Any,
    headlines_text: str,
    base_feature_values: dict[str, Any],
    snapshot_source: str,
    request_id: str | None = None,
    batch_id: str | None = None,
    target: int | None = None,
    feature_snapshot: dict | None = None,
    quote_currency: str | None = None,
    model_artifact_uri: str | None = None,
    created_at: Any | None = None,
) -> dict:
    snapshot_ts = pd.Timestamp(snapshot_timestamp)
    if snapshot_ts.tzinfo is None:
        snapshot_ts = snapshot_ts.tz_localize("UTC")
    else:
        snapshot_ts = snapshot_ts.tz_convert("UTC")

    created_ts = pd.Timestamp(created_at or datetime.now(timezone.utc))
    if created_ts.tzinfo is None:
        created_ts = created_ts.tz_localize("UTC")
    else:
        created_ts = created_ts.tz_convert("UTC")

    normalized_quote_currency = quote_currency.strip().upper() if isinstance(quote_currency, str) else None

    row = {
        "snapshot_id": snapshot_id,
        "request_id": request_id,
        "batch_id": batch_id,
        "ticker": ticker,
        "quote_currency": normalized_quote_currency,
        "snapshot_date": snapshot_ts.date().isoformat(),
        "snapshot_timestamp": snapshot_ts.isoformat(),
        "headlines_text": headlines_text,
        "target": target,
        "feature_snapshot": _serialize_feature_snapshot(feature_snapshot),
        "snapshot_source": snapshot_source,
        "model_artifact_uri": model_artifact_uri,
        "created_at": created_ts.isoformat(),
    }
    for feature_name in BASE_FEATURE_NAMES:
        row[feature_name] = _coerce_json_scalar(base_feature_values.get(feature_name))
    return row
