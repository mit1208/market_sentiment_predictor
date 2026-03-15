from types import SimpleNamespace

import pytest

from backend.ml.inference import batch_job


class DummyModel:
    def predict_proba(self, rows):
        return [[0.4, 0.6] for _ in range(len(rows))]


def test_score_request_requires_live_news(monkeypatch):
    monkeypatch.setattr(batch_job, "_live_feature_snapshot_table", lambda: "proj.ds.live_feature_snapshots")
    monkeypatch.setattr(batch_job, "count_daily_live_fetches", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        batch_job,
        "_load_cached_prediction_inputs",
        lambda **kwargs: (None, 0, None),
    )
    monkeypatch.setattr(
        batch_job,
        "fetch_live_snapshot",
        lambda ticker, **kwargs: {"as_of": "2026-03-14T15:30:00Z", "news": []},
    )

    request_row = SimpleNamespace(
        request_id="req-1",
        batch_id="batch-1",
        ticker="AAPL",
        headlines="",
        as_of_date=None,
    )

    with pytest.raises(ValueError) as exc_info:
        batch_job._score_request(
            request_row,
            DummyModel(),
            DummyModel(),
            {"feature_names": ["gap_pct", "gemini_sentiment_avg"], "trained_at": "2026-03-14T15:00:00Z"},
            "gs://bucket/model",
            object(),
        )

    assert str(exc_info.value) == (
        "Live news is required for inference for AAPL. "
        "No live news items were available from Finnhub or Yahoo."
    )


def test_load_cached_prediction_inputs_skips_incompatible_cached_snapshot(monkeypatch):
    monkeypatch.setattr(batch_job, "_live_feature_snapshot_table", lambda: "proj.ds.live_feature_snapshots")
    monkeypatch.setattr(batch_job, "count_daily_live_fetches", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        batch_job,
        "load_latest_cached_snapshot",
        lambda *args, **kwargs: {
            "snapshot_id": "snap-1",
            "request_id": "req-1",
            "snapshot_timestamp": "2026-03-14T15:00:00Z",
            "base_feature_values": {},
            "feature_snapshot": {},
        },
    )
    monkeypatch.setattr(
        batch_job,
        "evaluate_cache_reuse",
        lambda *args, **kwargs: {
            "should_use_cache": True,
            "stale_served": False,
            "quota_exhausted": False,
            "reason": "fresh_cache",
            "snapshot_age_seconds": 30,
        },
    )
    monkeypatch.setattr(
        batch_job,
        "materialize_cached_snapshot_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Required features contain null values: ['credit_spread']")
        ),
    )

    prediction_inputs, daily_live_fetch_count, cache_context = batch_job._load_cached_prediction_inputs(
        ticker="AAPL",
        metadata={"feature_names": ["credit_spread"]},
        client=object(),
    )

    assert prediction_inputs is None
    assert daily_live_fetch_count == 0
    assert cache_context["cached_snapshot"]["snapshot_id"] == "snap-1"
    assert cache_context["reuse_decision"]["reason"] == "cache_feature_mismatch"
