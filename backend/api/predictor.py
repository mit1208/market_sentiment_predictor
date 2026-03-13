"""Stateful predictor for Cloud Run with live API feature fetching and snapshot persistence."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timezone

import joblib
from google.cloud import bigquery, storage

from backend.config import get_env, get_int_env, qualify_table_name
from backend.ml.shared.feature_engineering import (
    BASE_FEATURE_NAMES,
    EXPECTED_FEATURE_COUNT,
    build_live_feature_bundle,
)
from backend.ml.shared.live_data import fetch_live_snapshot
from backend.ml.shared.snapshot_storage import (
    SNAPSHOT_SOURCE_ENDPOINT,
    build_live_feature_snapshot_row,
    build_snapshot_id,
)

log = logging.getLogger(__name__)

PROJECT_ID = get_env("GCP_PROJECT_ID")
REGION = get_env("GCP_REGION")
DATASET_ID = get_env("BQ_DATASET", "BQ_DATASET_ID")
MODEL_MANIFEST_URI = get_env("MODEL_MANIFEST_URI")
MODEL_REFRESH_INTERVAL_SECONDS = get_int_env("MODEL_REFRESH_INTERVAL_SECONDS", default=30) or 30
SNAPSHOT_CACHE_TTL_SECONDS = get_int_env("SNAPSHOT_CACHE_TTL_SECONDS", default=300) or 300
FINNHUB_KEY = get_env("FINNHUB_KEY", "FINNHUB_API_KEY")
LIVE_FEATURE_SNAPSHOT_TABLE = get_env("LIVE_FEATURE_SNAPSHOT_TABLE")


class Predictor:
    def __init__(self) -> None:
        self.model: object = None
        self.tfidf: object = None
        self.feature_names: list[str] = []
        self.model_trained_at: str | None = None
        self.model_quality: dict[str, object] = {}
        self._bq: bigquery.Client | None = None
        self._storage: storage.Client | None = None
        self._manifest_uri: str | None = None
        self._manifest_generation: str | None = None
        self._artifact_uri: str | None = None
        self._next_manifest_check_at: float = 0.0
        self._reload_lock = threading.Lock()
        self._snapshot_cache: dict[str, tuple[float, dict]] = {}

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

        log.info("Downloading model artifact from %s ...", artifact_uri)
        bucket_name, prefix = self._split_gcs_uri(artifact_uri)
        bucket = self.storage_client.bucket(bucket_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            for filename in ("model.joblib", "tfidf_vectorizer.joblib", "metadata.json"):
                local_path = os.path.join(tmpdir, filename)
                bucket.blob(f"{prefix}/{filename}").download_to_filename(local_path)

            self.model = joblib.load(os.path.join(tmpdir, "model.joblib"))
            self.tfidf = joblib.load(os.path.join(tmpdir, "tfidf_vectorizer.joblib"))
            with open(os.path.join(tmpdir, "metadata.json")) as handle:
                metadata = json.load(handle)

        self.feature_names = metadata["feature_names"]
        self.model_trained_at = metadata.get("trained_at")
        feature_importance = metadata.get("feature_importance") or []
        active_feature_importance = [
            row for row in feature_importance if float(row.get("importance") or 0.0) > 0
        ]
        self.model_quality = {
            "score_semantics": (
                "prob_up is the model's raw positive-class score for this snapshot. "
                "It is not calibrated historical hit rate."
            ),
            "cv_accuracy_mean": metadata.get("cv_acc_mean"),
            "cv_auc_mean": metadata.get("cv_auc_mean"),
            "holdout_accuracy": (metadata.get("test_metrics") or {}).get("accuracy"),
            "holdout_auc": (metadata.get("test_metrics") or {}).get("auc"),
            "used_feature_count": len(active_feature_importance),
            "zero_importance_feature_count": max(len(feature_importance) - len(active_feature_importance), 0),
            "top_features": active_feature_importance[:8],
        }
        if len(self.feature_names) != EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                f"Loaded model has {len(self.feature_names)} features; expected {EXPECTED_FEATURE_COUNT}."
            )
        log.info("Model loaded with %d features.", len(self.feature_names))

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

            if str(blob.generation) == self._manifest_generation:
                return

            log.info(
                "Detected updated model manifest generation %s (was %s). Reloading model.",
                blob.generation,
                self._manifest_generation,
            )
            self.load(self._manifest_uri)

    def _get_live_snapshot(self, ticker: str) -> dict:
        now = time.monotonic()
        cached = self._snapshot_cache.get(ticker)
        if cached and now < cached[0]:
            return cached[1]

        snapshot = fetch_live_snapshot(ticker, finnhub_key=FINNHUB_KEY)
        self._snapshot_cache[ticker] = (now + SNAPSHOT_CACHE_TTL_SECONDS, snapshot)
        return snapshot

    def _write_snapshot_row(self, row: dict) -> None:
        try:
            errors = self.bq.insert_rows_json(self._live_feature_snapshot_table(), [row])
        except Exception as exc:
            log.warning("Failed to write live feature snapshot for %s: %s", row.get("ticker"), exc)
            return
        if errors:
            log.warning("BigQuery reported snapshot insert errors: %s", errors)

    def predict(self, ticker: str) -> dict:
        if self.model is None:
            raise RuntimeError("Predictor not loaded.")

        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            raise ValueError("Ticker is required.")

        try:
            self.maybe_reload()
        except Exception as exc:
            log.exception("Model refresh failed, continuing with cached model: %s", exc)

        snapshot = self._get_live_snapshot(normalized_ticker)
        row, feature_snapshot = build_live_feature_bundle(snapshot, self.tfidf, self.feature_names)

        probability = float(self.model.predict_proba(row)[0, 1])
        label = "Up" if probability >= 0.5 else "Down"
        confidence = probability if probability >= 0.5 else 1.0 - probability

        predicted_at = datetime.now(timezone.utc).isoformat()
        request_id = build_snapshot_id("req", normalized_ticker, predicted_at)
        snapshot_id = build_snapshot_id("online", normalized_ticker, snapshot["as_of"])
        headlines_text = feature_snapshot.get("headlines_text") or " ".join(
            item.get("headline", "") for item in feature_snapshot.get("news_items", [])
        ).strip()
        base_feature_values = row.iloc[0][BASE_FEATURE_NAMES].to_dict()

        snapshot_row = build_live_feature_snapshot_row(
            snapshot_id=snapshot_id,
            request_id=request_id,
            ticker=normalized_ticker,
            snapshot_timestamp=snapshot["as_of"],
            headlines_text=headlines_text,
            base_feature_values=base_feature_values,
            snapshot_source=SNAPSHOT_SOURCE_ENDPOINT,
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
            "as_of": snapshot["as_of"],
            "predicted_at": predicted_at,
            "model_artifact_uri": self._artifact_uri,
            "model_trained_at": self.model_trained_at,
            "model_quality": self.model_quality,
            "feature_snapshot": feature_snapshot,
        }
