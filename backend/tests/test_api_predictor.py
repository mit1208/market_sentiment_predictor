import json
import shutil

import joblib
import pytest

import backend.api.predictor as predictor_module
from backend.api.predictor import Predictor


class DummyModel:
    def predict_proba(self, rows):
        return [[0.4, 0.6] for _ in range(len(rows))]


class FakeBlob:
    def __init__(self, *, source_path=None, text=None, generation="1"):
        self.source_path = source_path
        self.text = text
        self.generation = generation

    def download_as_text(self):
        return self.text

    def download_to_filename(self, filename):
        shutil.copy(self.source_path, filename)

    def reload(self):
        return None


class FakeBucket:
    def __init__(self, blobs):
        self._blobs = blobs

    def blob(self, object_name):
        return self._blobs[object_name]


class FakeStorageClient:
    def __init__(self, buckets):
        self._buckets = buckets

    def bucket(self, bucket_name):
        return self._buckets[bucket_name]


def test_predictor_load_exposes_feature_selection_metadata(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "run-1"
    artifact_dir.mkdir(parents=True)
    joblib.dump(DummyModel(), artifact_dir / "lgb_model.joblib")
    joblib.dump(DummyModel(), artifact_dir / "xgb_model.joblib")
    metadata = {
        "feature_names": ["gap_pct", "gemini_sentiment_avg"],
        "feature_importance": [{"feature": "gap_pct", "importance": 0.9}],
        "cv_acc_mean": 0.58,
        "cv_auc_mean": 0.61,
        "test_metrics": {"accuracy": 0.56, "auc": 0.6},
        "candidate_feature_count": 54,
        "feature_selection_source": "temporary_seeded_random_with_all_gemini",
        "feature_selection_mode": "seeded_random_subset",
        "feature_selection_is_temporary": True,
        "trained_at": "2026-03-14T15:00:00+00:00",
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata))

    manifest_blob = FakeBlob(
        text=json.dumps({"artifact_uri": "gs://unit-test-bucket/artifacts/run-1"}),
        generation="42",
    )
    artifact_blobs = {
        "models/market_sentiment_lgbm/latest.json": manifest_blob,
        "artifacts/run-1/lgb_model.joblib": FakeBlob(source_path=artifact_dir / "lgb_model.joblib"),
        "artifacts/run-1/xgb_model.joblib": FakeBlob(source_path=artifact_dir / "xgb_model.joblib"),
        "artifacts/run-1/metadata.json": FakeBlob(source_path=artifact_dir / "metadata.json"),
    }

    predictor = Predictor()
    predictor._storage = FakeStorageClient({"unit-test-bucket": FakeBucket(artifact_blobs)})

    predictor.load("gs://unit-test-bucket/models/market_sentiment_lgbm/latest.json")

    assert predictor.feature_names == ["gap_pct", "gemini_sentiment_avg"]
    assert predictor.model_quality["feature_selection_source"] == "temporary_seeded_random_with_all_gemini"
    assert predictor.model_quality["feature_selection_mode"] == "seeded_random_subset"
    assert predictor.model_quality["feature_selection_is_temporary"] is True


def test_predictor_predict_requires_live_news(monkeypatch):
    predictor = Predictor()
    predictor.lgb_model = DummyModel()
    predictor.xgb_model = DummyModel()
    predictor.feature_names = ["gap_pct", "gemini_sentiment_avg"]

    monkeypatch.setattr(
        predictor,
        "_load_cached_prediction_inputs",
        lambda ticker: (None, 0, None),
    )
    monkeypatch.setattr(
        predictor,
        "_get_live_snapshot",
        lambda ticker: {"as_of": "2026-03-14T15:30:00Z", "news": []},
    )

    with pytest.raises(ValueError) as exc_info:
        predictor.predict("AAPL")

    assert str(exc_info.value) == (
        "Live news is required for inference for AAPL. "
        "No live news items were available from Finnhub or Yahoo."
    )


def test_predictor_skips_cached_snapshot_when_current_model_features_are_missing(monkeypatch):
    predictor = Predictor()
    predictor.feature_names = ["credit_spread"]
    predictor._bq = object()
    monkeypatch.setattr(predictor, "_live_feature_snapshot_table", lambda: "proj.ds.live_feature_snapshots")

    monkeypatch.setattr(predictor_module, "count_daily_live_fetches", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        predictor_module,
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
        predictor_module,
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
        predictor_module,
        "materialize_cached_snapshot_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Required features contain null values: ['credit_spread']")
        ),
    )

    prediction_inputs, daily_live_fetch_count, cache_context = predictor._load_cached_prediction_inputs("AAPL")

    assert prediction_inputs is None
    assert daily_live_fetch_count == 0
    assert cache_context["cached_snapshot"]["snapshot_id"] == "snap-1"
    assert cache_context["reuse_decision"]["reason"] == "cache_feature_mismatch"
