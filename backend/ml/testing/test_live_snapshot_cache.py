import sys
import types

try:
    import google as google_module  # type: ignore
except ImportError:
    google_module = types.ModuleType("google")
    sys.modules["google"] = google_module

cloud_module = sys.modules.setdefault("google.cloud", types.ModuleType("google.cloud"))
bigquery_module = sys.modules.setdefault("google.cloud.bigquery", types.ModuleType("google.cloud.bigquery"))
setattr(bigquery_module, "Client", object)
setattr(google_module, "cloud", cloud_module)
setattr(cloud_module, "bigquery", bigquery_module)

from backend.ml.shared.live_snapshot_cache import (
    CACHE_REASON_DAILY_BUDGET_EXHAUSTED,
    CACHE_REASON_DAILY_BUDGET_GUARD,
    CACHE_REASON_FRESH_CACHE,
    CACHE_SOURCE_MODE_BIGQUERY,
    CACHE_SOURCE_MODE_LIVE_API,
    build_cache_status,
    evaluate_cache_reuse,
)


def test_evaluate_cache_reuse_prefers_fresh_cache():
    decision = evaluate_cache_reuse(
        "2026-03-14T15:00:00Z",
        now="2026-03-14T15:20:00Z",
        fresh_ttl_seconds=3600,
        max_stale_ttl_seconds=21600,
        daily_live_fetch_limit=12,
        daily_live_fetch_count=4,
    )

    assert decision["should_use_cache"] is True
    assert decision["stale_served"] is False
    assert decision["reason"] == CACHE_REASON_FRESH_CACHE


def test_evaluate_cache_reuse_allows_stale_cache_when_budget_is_exhausted():
    decision = evaluate_cache_reuse(
        "2026-03-14T10:30:00Z",
        now="2026-03-14T15:00:00Z",
        fresh_ttl_seconds=3600,
        max_stale_ttl_seconds=21600,
        daily_live_fetch_limit=3,
        daily_live_fetch_count=3,
    )

    assert decision["should_use_cache"] is True
    assert decision["stale_served"] is True
    assert decision["quota_exhausted"] is True
    assert decision["reason"] == CACHE_REASON_DAILY_BUDGET_GUARD


def test_evaluate_cache_reuse_blocks_refresh_when_budget_is_exhausted_and_snapshot_too_old():
    decision = evaluate_cache_reuse(
        "2026-03-14T05:00:00Z",
        now="2026-03-14T15:00:00Z",
        fresh_ttl_seconds=3600,
        max_stale_ttl_seconds=21600,
        daily_live_fetch_limit=3,
        daily_live_fetch_count=3,
    )

    assert decision["should_use_cache"] is False
    assert decision["quota_exhausted"] is True
    assert decision["reason"] == CACHE_REASON_DAILY_BUDGET_EXHAUSTED


def test_build_cache_status_increments_live_fetch_usage_for_live_requests():
    status = build_cache_status(
        source_mode=CACHE_SOURCE_MODE_LIVE_API,
        reason="live_refresh",
        snapshot_timestamp="2026-03-14T15:00:00Z",
        snapshot_age_seconds=0,
        fresh_ttl_seconds=3600,
        max_stale_ttl_seconds=21600,
        daily_live_fetch_limit=12,
        daily_live_fetch_count=4,
        cache_hit=False,
        stale_served=False,
        quota_exhausted=False,
    )

    assert status["daily_live_fetch_count"] == 5
    assert status["daily_live_fetch_remaining"] == 7


def test_build_cache_status_does_not_increment_live_fetch_usage_for_cache_hits():
    status = build_cache_status(
        source_mode=CACHE_SOURCE_MODE_BIGQUERY,
        reason="fresh_cache",
        snapshot_timestamp="2026-03-14T15:00:00Z",
        snapshot_age_seconds=120,
        fresh_ttl_seconds=3600,
        max_stale_ttl_seconds=21600,
        daily_live_fetch_limit=12,
        daily_live_fetch_count=4,
        cache_hit=True,
        stale_served=False,
        quota_exhausted=False,
    )

    assert status["daily_live_fetch_count"] == 4
    assert status["daily_live_fetch_remaining"] == 8
