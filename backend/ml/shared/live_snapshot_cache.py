"""BigQuery-backed live inference cache helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from google.cloud import bigquery

from backend.ml.shared.feature_engineering import CANDIDATE_FEATURE_NAMES, align_model_features

CACHE_SOURCE_MODE_LIVE_API = "live_api"
CACHE_SOURCE_MODE_BIGQUERY = "bigquery_cache"

CACHE_REASON_CACHE_MISS = "cache_miss"
CACHE_REASON_CACHE_FEATURE_MISMATCH = "cache_feature_mismatch"
CACHE_REASON_FRESH_CACHE = "fresh_cache"
CACHE_REASON_LIVE_REFRESH = "live_refresh"
CACHE_REASON_DAILY_BUDGET_GUARD = "daily_budget_guard"
CACHE_REASON_DAILY_BUDGET_EXHAUSTED = "daily_budget_exhausted"

LIVE_API_SNAPSHOT_SOURCES = ("endpoint_live_api", "batch_live_api")
REUSABLE_SNAPSHOT_SOURCES = (
    "endpoint_live_api",
    "batch_live_api",
    "endpoint_bigquery_cache",
    "batch_bigquery_cache",
)


def normalize_cache_timestamp(value: Any) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp).tz_convert("UTC")


def evaluate_cache_reuse(
    snapshot_timestamp: Any,
    *,
    now: Any,
    fresh_ttl_seconds: int,
    max_stale_ttl_seconds: int,
    daily_live_fetch_limit: int,
    daily_live_fetch_count: int,
) -> dict:
    normalized_now = normalize_cache_timestamp(now) or normalize_cache_timestamp(datetime.now(timezone.utc))
    normalized_snapshot_timestamp = normalize_cache_timestamp(snapshot_timestamp)
    if normalized_snapshot_timestamp is None:
        return {
            "should_use_cache": False,
            "stale_served": False,
            "quota_exhausted": daily_live_fetch_count >= daily_live_fetch_limit,
            "reason": CACHE_REASON_CACHE_MISS,
            "snapshot_age_seconds": None,
        }

    snapshot_age_seconds = max(
        0,
        int((normalized_now - normalized_snapshot_timestamp).total_seconds()),
    )
    quota_exhausted = daily_live_fetch_count >= daily_live_fetch_limit
    if snapshot_age_seconds <= max(0, fresh_ttl_seconds):
        return {
            "should_use_cache": True,
            "stale_served": False,
            "quota_exhausted": quota_exhausted,
            "reason": CACHE_REASON_FRESH_CACHE,
            "snapshot_age_seconds": snapshot_age_seconds,
        }

    if quota_exhausted and snapshot_age_seconds <= max(0, max_stale_ttl_seconds):
        return {
            "should_use_cache": True,
            "stale_served": True,
            "quota_exhausted": True,
            "reason": CACHE_REASON_DAILY_BUDGET_GUARD,
            "snapshot_age_seconds": snapshot_age_seconds,
        }

    return {
        "should_use_cache": False,
        "stale_served": False,
        "quota_exhausted": quota_exhausted,
        "reason": CACHE_REASON_DAILY_BUDGET_EXHAUSTED if quota_exhausted else CACHE_REASON_LIVE_REFRESH,
        "snapshot_age_seconds": snapshot_age_seconds,
    }


def build_cache_status(
    *,
    source_mode: str,
    reason: str,
    snapshot_timestamp: Any,
    snapshot_age_seconds: int | None,
    fresh_ttl_seconds: int,
    max_stale_ttl_seconds: int,
    daily_live_fetch_limit: int,
    daily_live_fetch_count: int,
    cache_hit: bool,
    stale_served: bool,
    quota_exhausted: bool,
    origin_snapshot_id: str | None = None,
    origin_request_id: str | None = None,
) -> dict:
    normalized_snapshot_timestamp = normalize_cache_timestamp(snapshot_timestamp)
    consumed_live_fetches = 1 if source_mode == CACHE_SOURCE_MODE_LIVE_API else 0
    used_live_fetch_count = daily_live_fetch_count + consumed_live_fetches
    return {
        "source_mode": source_mode,
        "reason": reason,
        "cache_hit": bool(cache_hit),
        "stale_served": bool(stale_served),
        "quota_exhausted": bool(quota_exhausted),
        "snapshot_timestamp": None if normalized_snapshot_timestamp is None else normalized_snapshot_timestamp.isoformat(),
        "snapshot_age_seconds": snapshot_age_seconds,
        "snapshot_age_minutes": None if snapshot_age_seconds is None else round(snapshot_age_seconds / 60.0, 1),
        "fresh_ttl_seconds": fresh_ttl_seconds,
        "fresh_ttl_minutes": round(fresh_ttl_seconds / 60.0, 1),
        "max_stale_ttl_seconds": max_stale_ttl_seconds,
        "max_stale_ttl_minutes": round(max_stale_ttl_seconds / 60.0, 1),
        "daily_live_fetch_limit": daily_live_fetch_limit,
        "daily_live_fetch_count": used_live_fetch_count,
        "daily_live_fetch_remaining": max(daily_live_fetch_limit - used_live_fetch_count, 0),
        "origin_snapshot_id": origin_snapshot_id,
        "origin_request_id": origin_request_id,
    }


def parse_feature_snapshot_payload(value: Any) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def load_latest_cached_snapshot(
    client: bigquery.Client,
    *,
    table_name: str,
    ticker: str,
) -> dict | None:
    select_columns = ", ".join([f"`{column}`" for column in CANDIDATE_FEATURE_NAMES])
    query = f"""
        SELECT
          snapshot_id,
          request_id,
          batch_id,
          ticker,
          quote_currency,
          snapshot_source,
          snapshot_timestamp,
          created_at,
          feature_snapshot,
          {select_columns}
        FROM `{table_name}`
        WHERE ticker = @ticker
          AND snapshot_source IN UNNEST(@snapshot_sources)
        ORDER BY snapshot_timestamp DESC, created_at DESC
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
            bigquery.ArrayQueryParameter("snapshot_sources", "STRING", list(REUSABLE_SNAPSHOT_SOURCES)),
        ]
    )
    rows = list(client.query(query, job_config=job_config).result())
    if not rows:
        return None

    row = rows[0]
    feature_snapshot = parse_feature_snapshot_payload(getattr(row, "feature_snapshot", None))
    base_feature_values = {feature_name: getattr(row, feature_name, None) for feature_name in CANDIDATE_FEATURE_NAMES}
    snapshot_timestamp = normalize_cache_timestamp(getattr(row, "snapshot_timestamp", None))
    created_at = normalize_cache_timestamp(getattr(row, "created_at", None))
    if snapshot_timestamp and not feature_snapshot.get("as_of"):
        feature_snapshot["as_of"] = snapshot_timestamp.isoformat()
    quote_currency = getattr(row, "quote_currency", None)
    if quote_currency and not feature_snapshot.get("quote_currency"):
        feature_snapshot["quote_currency"] = str(quote_currency).strip().upper()

    return {
        "snapshot_id": getattr(row, "snapshot_id", None),
        "request_id": getattr(row, "request_id", None),
        "batch_id": getattr(row, "batch_id", None),
        "ticker": getattr(row, "ticker", None),
        "quote_currency": quote_currency,
        "snapshot_source": getattr(row, "snapshot_source", None),
        "snapshot_timestamp": snapshot_timestamp,
        "created_at": created_at,
        "feature_snapshot": feature_snapshot,
        "base_feature_values": base_feature_values,
    }


def count_daily_live_fetches(
    client: bigquery.Client,
    *,
    table_name: str,
    ticker: str,
    snapshot_date: Any,
) -> int:
    normalized_snapshot_timestamp = normalize_cache_timestamp(snapshot_date)
    if normalized_snapshot_timestamp is None:
        normalized_snapshot_timestamp = normalize_cache_timestamp(datetime.now(timezone.utc))

    query = f"""
        SELECT COUNT(*) AS live_fetch_count
        FROM `{table_name}`
        WHERE ticker = @ticker
          AND snapshot_date = @snapshot_date
          AND snapshot_source IN UNNEST(@snapshot_sources)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
            bigquery.ScalarQueryParameter("snapshot_date", "DATE", normalized_snapshot_timestamp.date()),
            bigquery.ArrayQueryParameter("snapshot_sources", "STRING", list(LIVE_API_SNAPSHOT_SOURCES)),
        ]
    )
    rows = list(client.query(query, job_config=job_config).result())
    if not rows:
        return 0
    return int(getattr(rows[0], "live_fetch_count", 0) or 0)


def materialize_cached_snapshot_frames(
    cached_snapshot: dict,
    *,
    feature_names: list[str],
    cache_status: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    candidate_row = pd.DataFrame(
        [{feature_name: cached_snapshot["base_feature_values"].get(feature_name) for feature_name in CANDIDATE_FEATURE_NAMES}],
        columns=CANDIDATE_FEATURE_NAMES,
    )
    selected_row = align_model_features(candidate_row, feature_names)
    feature_snapshot = {
        **dict(cached_snapshot.get("feature_snapshot") or {}),
        "cache_status": cache_status,
    }
    if "selected_feature_count" not in feature_snapshot:
        feature_snapshot["selected_feature_count"] = len(selected_row.columns)
    if "feature_count" not in feature_snapshot:
        feature_snapshot["feature_count"] = len(CANDIDATE_FEATURE_NAMES)
    return candidate_row, selected_row, feature_snapshot
