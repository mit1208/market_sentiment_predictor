"""Stateful predictor for Cloud Run with live API feature fetching and snapshot persistence."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
from google.cloud import bigquery, storage

from backend.config import get_env, get_int_env, qualify_table_name
from backend.ml.shared.feature_engineering import (
    build_live_feature_bundle,
    require_live_news_items,
)
from backend.ml.shared.gemini_sentiment import score_live_news_items
from backend.ml.shared.live_data import fetch_live_snapshot
from backend.ml.shared.live_snapshot_cache import (
    CACHE_REASON_CACHE_FEATURE_MISMATCH,
    CACHE_REASON_DAILY_BUDGET_EXHAUSTED,
    CACHE_REASON_LIVE_REFRESH,
    CACHE_SOURCE_MODE_BIGQUERY,
    CACHE_SOURCE_MODE_LIVE_API,
    build_cache_status,
    count_daily_live_fetches,
    evaluate_cache_reuse,
    load_latest_cached_snapshot,
    materialize_cached_snapshot_frames,
)
from backend.ml.shared.model_scoring import blend_positive_class_scores, classify_probability
from backend.ml.shared.macro_features import fetch_live_macro_features, load_latest_universe_breadth
from backend.ml.shared.snapshot_storage import (
    SNAPSHOT_SOURCE_ENDPOINT,
    SNAPSHOT_SOURCE_ENDPOINT_CACHE,
    build_live_feature_snapshot_row,
    build_snapshot_id,
)

PROJECT_ID = get_env("GCP_PROJECT_ID")
REGION = get_env("GCP_REGION")
DATASET_ID = get_env("BQ_DATASET", "BQ_DATASET_ID")
MODEL_MANIFEST_URI = get_env("MODEL_MANIFEST_URI")
MODEL_REFRESH_INTERVAL_SECONDS = get_int_env("MODEL_REFRESH_INTERVAL_SECONDS", default=30) or 30
SNAPSHOT_CACHE_TTL_SECONDS = get_int_env("SNAPSHOT_CACHE_TTL_SECONDS", default=300) or 300
SENTIMENT_CACHE_TTL_SECONDS = get_int_env("SENTIMENT_CACHE_TTL_SECONDS", default=1800) or 1800
MACRO_CACHE_TTL_SECONDS = get_int_env("MACRO_CACHE_TTL_SECONDS", default=86400) or 86400
LIVE_SNAPSHOT_FRESH_TTL_SECONDS = get_int_env("LIVE_SNAPSHOT_FRESH_TTL_SECONDS", default=3600) or 3600
LIVE_SNAPSHOT_MAX_STALE_TTL_SECONDS = get_int_env("LIVE_SNAPSHOT_MAX_STALE_TTL_SECONDS", default=21600) or 21600
LIVE_SNAPSHOT_DAILY_LIVE_FETCH_LIMIT = get_int_env("LIVE_SNAPSHOT_DAILY_LIVE_FETCH_LIMIT", default=12) or 12
FINNHUB_KEY = get_env("FINNHUB_KEY", "FINNHUB_API_KEY")
FRED_API_KEY = get_env("FRED_API_KEY")
LIVE_FEATURE_SNAPSHOT_TABLE = get_env("LIVE_FEATURE_SNAPSHOT_TABLE")


class Predictor:
    def __init__(self) -> None:
        self.lgb_model: object = None
        self.xgb_model: object = None
        self.feature_names: list[str] = []
        self.model_trained_at: str | None = None
        self.model_quality: dict[str, object] = {}
        self._metadata: dict[str, object] = {}
        self._bq: bigquery.Client | None = None
        self._storage: storage.Client | None = None
        self._manifest_uri: str | None = None
        self._manifest_generation: str | None = None
        self._artifact_uri: str | None = None
        self._next_manifest_check_at: float = 0.0
        self._reload_lock = threading.Lock()
        self._snapshot_cache: dict[str, tuple[float, dict]] = {}
        self._sentiment_cache: dict[str, tuple[float, dict]] = {}
        self._macro_cache: tuple[float, dict] | None = None

    @property
    def storage_client(self) -> storage.Client:
        if self._storage is None:
            self._storage = storage.Client(project=PROJECT_ID or None)
        return self._storage

    @property
    def bq(self) -> bigquery.Client:
        if self._bq is None:
            self._bq = bigquery.Client(project=PROJECT_ID or None)
        return self._bq

    def _live_feature_snapshot_table(self) -> str:
        return LIVE_FEATURE_SNAPSHOT_TABLE or qualify_table_name(
            "live_feature_snapshots",
            project_id=PROJECT_ID or None,
            dataset_id=DATASET_ID or None,
        )

    def _training_snapshot_table(self) -> str:
        return qualify_table_name(
            "training_feature_snapshots",
            project_id=PROJECT_ID or None,
            dataset_id=DATASET_ID or None,
        )

    def _gemini_cache_table(self) -> str:
        return qualify_table_name(
            "gemini_sentiment_cache",
            project_id=PROJECT_ID or None,
            dataset_id=DATASET_ID or None,
        )

    def _macro_cache_table(self) -> str:
        return qualify_table_name(
            "macro_feature_cache",
            project_id=PROJECT_ID or None,
            dataset_id=DATASET_ID or None,
        )

    def _split_gcs_uri(self, uri: str) -> tuple[str, str]:
        without_scheme = uri.removeprefix("gs://")
        bucket_name, _, object_name = without_scheme.partition("/")
        if not bucket_name or not object_name:
            raise ValueError(f"Invalid GCS URI: {uri}")
        return bucket_name, object_name

    def _resolve_artifact_uri(self, uri: str) -> str:
        if uri.endswith(".json"):
            bucket_name, object_name = self._split_gcs_uri(uri)
            blob = self.storage_client.bucket(bucket_name).blob(object_name)
            manifest = json.loads(blob.download_as_text())
            artifact_uri = manifest["artifact_uri"].rstrip("/")
            self._manifest_uri = uri
            self._manifest_generation = str(blob.generation)
            self._next_manifest_check_at = time.monotonic() + MODEL_REFRESH_INTERVAL_SECONDS
            return artifact_uri

        self._manifest_uri = None
        self._manifest_generation = None
        self._next_manifest_check_at = 0.0
        return uri.rstrip("/")

    def load(self, gcs_uri: str | None = None) -> None:
        uri = (gcs_uri or MODEL_MANIFEST_URI).rstrip("/")
        if not uri:
            raise RuntimeError("No GCS URI found in MODEL_MANIFEST_URI.")

        artifact_uri = self._resolve_artifact_uri(uri)
        self._artifact_uri = artifact_uri
        bucket_name, prefix = self._split_gcs_uri(artifact_uri)
        bucket = self.storage_client.bucket(bucket_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            for filename in ("lgb_model.joblib", "xgb_model.joblib", "metadata.json"):
                bucket.blob(f"{prefix}/{filename}").download_to_filename(os.path.join(tmpdir, filename))

            self.lgb_model = joblib.load(os.path.join(tmpdir, "lgb_model.joblib"))
            self.xgb_model = joblib.load(os.path.join(tmpdir, "xgb_model.joblib"))
            with open(os.path.join(tmpdir, "metadata.json")) as handle:
                metadata = json.load(handle)

        self.feature_names = metadata["feature_names"]
        self.model_trained_at = metadata.get("trained_at")
        self._metadata = metadata
        feature_importance = metadata.get("feature_importance") or []
        self.model_quality = {
            "score_semantics": "prob_up is the ensemble positive-class score for this live snapshot.",
            "cv_accuracy_mean": metadata.get("cv_acc_mean"),
            "cv_auc_mean": metadata.get("cv_auc_mean"),
            "holdout_accuracy": (metadata.get("test_metrics") or {}).get("accuracy"),
            "holdout_auc": (metadata.get("test_metrics") or {}).get("auc"),
            "used_feature_count": len(self.feature_names),
            "candidate_feature_count": metadata.get("candidate_feature_count"),
            "feature_selection_source": metadata.get("feature_selection_source"),
            "feature_selection_mode": metadata.get("feature_selection_mode"),
            "feature_selection_is_temporary": bool(metadata.get("feature_selection_is_temporary")),
            "prediction_strategy": metadata.get("prediction_strategy"),
            "classification_threshold": metadata.get("classification_threshold"),
            "top_features": feature_importance[:8],
        }

    def maybe_reload(self) -> None:
        if not self._manifest_uri:
            return

        now = time.monotonic()
        if now < self._next_manifest_check_at:
            return

        with self._reload_lock:
            now = time.monotonic()
            if now < self._next_manifest_check_at:
                return

            bucket_name, object_name = self._split_gcs_uri(self._manifest_uri)
            blob = self.storage_client.bucket(bucket_name).blob(object_name)
            blob.reload()
            self._next_manifest_check_at = now + MODEL_REFRESH_INTERVAL_SECONDS

            if str(blob.generation) != self._manifest_generation:
                self.load(self._manifest_uri)

    def _get_live_snapshot(self, ticker: str) -> dict:
        now = time.monotonic()
        cached = self._snapshot_cache.get(ticker)
        if cached and now < cached[0]:
            return cached[1]

        snapshot = fetch_live_snapshot(ticker, finnhub_key=FINNHUB_KEY)
        self._snapshot_cache[ticker] = (now + SNAPSHOT_CACHE_TTL_SECONDS, snapshot)
        return snapshot

    def _sentiment_cache_key(self, ticker: str, news_items: list[dict]) -> str:
        material = "|".join(
            f"{ticker}:{item.get('headline','')}:{item.get('published_at','')}"
            for item in news_items
        )
        return material

    def _get_sentiment_features(self, ticker: str, news_items: list[dict]) -> dict:
        cache_key = self._sentiment_cache_key(ticker, news_items)
        now = time.monotonic()
        cached = self._sentiment_cache.get(cache_key)
        if cached and now < cached[0]:
            return cached[1]

        features = score_live_news_items(
            ticker=ticker,
            news_items=news_items,
            project_id=PROJECT_ID,
            region=REGION,
            cache_client=self.bq,
            cache_table_name=self._gemini_cache_table(),
        )
        self._sentiment_cache[cache_key] = (now + SENTIMENT_CACHE_TTL_SECONDS, features)
        return features

    def _get_macro_features(self, as_of: pd.Timestamp) -> dict:
        now = time.monotonic()
        if self._macro_cache and now < self._macro_cache[0]:
            return self._macro_cache[1]
        if not FRED_API_KEY:
            raise RuntimeError("FRED_API_KEY is required for live macro features.")

        breadth = load_latest_universe_breadth(
            self.bq,
            table_name=self._training_snapshot_table(),
            as_of_date=as_of,
        )
        features = fetch_live_macro_features(
            fred_api_key=FRED_API_KEY,
            universe_breadth_50d=breadth,
            cache_client=self.bq,
            cache_table_name=self._macro_cache_table(),
            as_of_date=as_of,
        )
        self._macro_cache = (now + MACRO_CACHE_TTL_SECONDS, features)
        return features

    def _write_snapshot_row(self, row: dict) -> None:
        try:
            errors = self.bq.insert_rows_json(self._live_feature_snapshot_table(), [row])
        except Exception:
            return
        if errors:
            raise RuntimeError(f"BigQuery snapshot insert failed: {errors[:3]}")

    def _load_cached_prediction_inputs(self, ticker: str) -> tuple[dict | None, int, dict | None]:
        now = datetime.now(timezone.utc)
        daily_live_fetch_count = count_daily_live_fetches(
            self.bq,
            table_name=self._live_feature_snapshot_table(),
            ticker=ticker,
            snapshot_date=now,
        )
        cached_snapshot = load_latest_cached_snapshot(
            self.bq,
            table_name=self._live_feature_snapshot_table(),
            ticker=ticker,
        )
        if not cached_snapshot:
            return None, daily_live_fetch_count, None

        reuse_decision = evaluate_cache_reuse(
            cached_snapshot.get("snapshot_timestamp"),
            now=now,
            fresh_ttl_seconds=LIVE_SNAPSHOT_FRESH_TTL_SECONDS,
            max_stale_ttl_seconds=LIVE_SNAPSHOT_MAX_STALE_TTL_SECONDS,
            daily_live_fetch_limit=LIVE_SNAPSHOT_DAILY_LIVE_FETCH_LIMIT,
            daily_live_fetch_count=daily_live_fetch_count,
        )
        if not reuse_decision["should_use_cache"]:
            return None, daily_live_fetch_count, {
                "cached_snapshot": cached_snapshot,
                "reuse_decision": reuse_decision,
            }

        cache_status = build_cache_status(
            source_mode=CACHE_SOURCE_MODE_BIGQUERY,
            reason=reuse_decision["reason"],
            snapshot_timestamp=cached_snapshot.get("snapshot_timestamp"),
            snapshot_age_seconds=reuse_decision["snapshot_age_seconds"],
            fresh_ttl_seconds=LIVE_SNAPSHOT_FRESH_TTL_SECONDS,
            max_stale_ttl_seconds=LIVE_SNAPSHOT_MAX_STALE_TTL_SECONDS,
            daily_live_fetch_limit=LIVE_SNAPSHOT_DAILY_LIVE_FETCH_LIMIT,
            daily_live_fetch_count=daily_live_fetch_count,
            cache_hit=True,
            stale_served=bool(reuse_decision["stale_served"]),
            quota_exhausted=bool(reuse_decision["quota_exhausted"]),
            origin_snapshot_id=cached_snapshot.get("snapshot_id"),
            origin_request_id=cached_snapshot.get("request_id"),
        )
        try:
            candidate_row, selected_row, feature_snapshot = materialize_cached_snapshot_frames(
                cached_snapshot,
                feature_names=self.feature_names,
                cache_status=cache_status,
            )
        except AssertionError:
            return None, daily_live_fetch_count, {
                "cached_snapshot": cached_snapshot,
                "reuse_decision": {
                    **reuse_decision,
                    "reason": CACHE_REASON_CACHE_FEATURE_MISMATCH,
                },
            }
        return (
            {
                "candidate_row": candidate_row,
                "selected_row": selected_row,
                "feature_snapshot": feature_snapshot,
                "snapshot_timestamp": cached_snapshot.get("snapshot_timestamp"),
                "snapshot_source": SNAPSHOT_SOURCE_ENDPOINT_CACHE,
            },
            daily_live_fetch_count,
            {
                "cached_snapshot": cached_snapshot,
                "reuse_decision": reuse_decision,
            },
        )

    def _build_live_prediction_inputs(self, ticker: str, daily_live_fetch_count: int) -> dict:
        if daily_live_fetch_count >= LIVE_SNAPSHOT_DAILY_LIVE_FETCH_LIMIT:
            raise RuntimeError(
                f"Live data budget exhausted for {ticker}. "
                f"The daily live fetch limit is {LIVE_SNAPSHOT_DAILY_LIVE_FETCH_LIMIT} and no reusable cached snapshot is available."
            )

        snapshot = self._get_live_snapshot(ticker)
        normalized_news = require_live_news_items(snapshot.get("news"), ticker=ticker)
        sentiment_features = self._get_sentiment_features(ticker, normalized_news)
        macro_features = self._get_macro_features(pd.Timestamp(snapshot["as_of"]))
        candidate_row, selected_row, feature_snapshot = build_live_feature_bundle(
            snapshot,
            sentiment_features=sentiment_features,
            macro_features=macro_features,
            feature_names=self.feature_names,
        )
        feature_snapshot["cache_status"] = build_cache_status(
            source_mode=CACHE_SOURCE_MODE_LIVE_API,
            reason=CACHE_REASON_LIVE_REFRESH,
            snapshot_timestamp=snapshot["as_of"],
            snapshot_age_seconds=0,
            fresh_ttl_seconds=LIVE_SNAPSHOT_FRESH_TTL_SECONDS,
            max_stale_ttl_seconds=LIVE_SNAPSHOT_MAX_STALE_TTL_SECONDS,
            daily_live_fetch_limit=LIVE_SNAPSHOT_DAILY_LIVE_FETCH_LIMIT,
            daily_live_fetch_count=daily_live_fetch_count,
            cache_hit=False,
            stale_served=False,
            quota_exhausted=False,
        )
        return {
            "candidate_row": candidate_row,
            "selected_row": selected_row,
            "feature_snapshot": feature_snapshot,
            "snapshot_timestamp": pd.Timestamp(snapshot["as_of"]),
            "snapshot_source": SNAPSHOT_SOURCE_ENDPOINT,
        }

    def _resolve_prediction_inputs(self, ticker: str) -> dict:
        cached_inputs, daily_live_fetch_count, cache_context = self._load_cached_prediction_inputs(ticker)
        if cached_inputs is not None:
            return cached_inputs

        reuse_decision = (cache_context or {}).get("reuse_decision") or {}
        if reuse_decision.get("reason") == CACHE_REASON_DAILY_BUDGET_EXHAUSTED:
            cached_snapshot = (cache_context or {}).get("cached_snapshot") or {}
            snapshot_age_seconds = reuse_decision.get("snapshot_age_seconds")
            snapshot_age_hours = (
                "unknown"
                if snapshot_age_seconds is None
                else f"{round(float(snapshot_age_seconds) / 3600.0, 1)}h"
            )
            origin_snapshot_id = cached_snapshot.get("snapshot_id")
            raise RuntimeError(
                f"Live data budget exhausted for {ticker}. "
                f"The latest cached snapshot is {snapshot_age_hours} old and exceeds the {round(LIVE_SNAPSHOT_MAX_STALE_TTL_SECONDS / 3600.0, 1)}h maximum stale window."
                + (f" Latest snapshot: {origin_snapshot_id}." if origin_snapshot_id else "")
            )

        return self._build_live_prediction_inputs(ticker, daily_live_fetch_count)

    def predict(self, ticker: str) -> dict:
        if self.lgb_model is None or self.xgb_model is None:
            raise RuntimeError("Predictor not loaded.")

        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            raise ValueError("Ticker is required.")

        self.maybe_reload()
        prediction_inputs = self._resolve_prediction_inputs(normalized_ticker)
        candidate_row = prediction_inputs["candidate_row"]
        selected_row = prediction_inputs["selected_row"]
        feature_snapshot = prediction_inputs["feature_snapshot"]
        snapshot_timestamp = pd.Timestamp(prediction_inputs["snapshot_timestamp"])
        snapshot_source = prediction_inputs["snapshot_source"]
        lgb_proba = float(self.lgb_model.predict_proba(selected_row)[0, 1])
        xgb_proba = float(self.xgb_model.predict_proba(selected_row)[0, 1])
        probability = blend_positive_class_scores(lgb_proba, xgb_proba, metadata=self._metadata)
        label, confidence, _threshold = classify_probability(probability, metadata=self._metadata)

        predicted_at = datetime.now(timezone.utc).isoformat()
        request_id = build_snapshot_id("req", normalized_ticker, predicted_at)
        snapshot_id = build_snapshot_id(
            "onlinecache" if snapshot_source == SNAPSHOT_SOURCE_ENDPOINT_CACHE else "online",
            normalized_ticker,
            predicted_at if snapshot_source == SNAPSHOT_SOURCE_ENDPOINT_CACHE else snapshot_timestamp,
        )
        headlines_text = feature_snapshot.get("headlines_text") or " ".join(
            item.get("headline", "") for item in feature_snapshot.get("news_items", [])
        ).strip()

        snapshot_row = build_live_feature_snapshot_row(
            snapshot_id=snapshot_id,
            request_id=request_id,
            ticker=normalized_ticker,
            snapshot_timestamp=snapshot_timestamp,
            headlines_text=headlines_text,
            base_feature_values=candidate_row.iloc[0].to_dict(),
            snapshot_source=snapshot_source,
            feature_snapshot=feature_snapshot,
            quote_currency=feature_snapshot.get("quote_currency"),
            model_artifact_uri=self._artifact_uri,
        )
        self._write_snapshot_row(snapshot_row)

        return {
            "request_id": request_id,
            "ticker": normalized_ticker,
            "prediction": label,
            "confidence": round(confidence, 4),
            "prob_up": round(probability, 4),
            "as_of": snapshot_timestamp.isoformat(),
            "predicted_at": predicted_at,
            "model_artifact_uri": self._artifact_uri,
            "model_trained_at": self.model_trained_at,
            "model_quality": self.model_quality,
            "feature_snapshot": feature_snapshot,
        }
