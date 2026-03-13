"""Cloud Run Job entrypoint for live batch inference."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone

import joblib
import pandas as pd
from google.cloud import bigquery, storage

from shared.feature_engineering import (
    BASE_FEATURE_NAMES,
    EXPECTED_FEATURE_COUNT,
    build_live_feature_bundle,
    normalize_headlines,
)
from shared.live_data import fetch_live_snapshot
from shared.snapshot_storage import (
    SNAPSHOT_SOURCE_BATCH,
    build_live_feature_snapshot_row,
    build_snapshot_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "llm-testing-489113")
DATASET_ID = os.getenv("BQ_DATASET", "market_sentiment_predictor")
MODEL_MANIFEST_URI = os.getenv("MODEL_MANIFEST_URI", "")
REQUEST_TABLE = os.getenv(
    "BATCH_REQUEST_TABLE",
    f"{PROJECT_ID}.{DATASET_ID}.batch_inference_requests",
)
PREDICTION_TABLE = os.getenv(
    "BATCH_PREDICTION_TABLE",
    f"{PROJECT_ID}.{DATASET_ID}.batch_inference_predictions",
)
LIVE_FEATURE_SNAPSHOT_TABLE = os.getenv(
    "LIVE_FEATURE_SNAPSHOT_TABLE",
    f"{PROJECT_ID}.{DATASET_ID}.live_feature_snapshots",
)
FINNHUB_KEY = os.getenv("FINNHUB_KEY") or os.getenv("FINNHUB_API_KEY")

REQUEST_STATUSES = {
    "pending": "PENDING",
    "processing": "PROCESSING",
    "complete": "COMPLETE",
    "failed": "FAILED",
}


def _split_gcs_uri(uri: str) -> tuple[str, str]:
    without_scheme = uri.removeprefix("gs://")
    bucket_name, _, object_name = without_scheme.partition("/")
    if not bucket_name or not object_name:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return bucket_name, object_name


def _load_model_artifacts(manifest_uri: str) -> tuple[object, object, dict, str]:
    if not manifest_uri:
        raise RuntimeError("MODEL_MANIFEST_URI is not configured for batch inference.")

    storage_client = storage.Client(project=PROJECT_ID)
    bucket_name, object_name = _split_gcs_uri(manifest_uri)
    manifest_blob = storage_client.bucket(bucket_name).blob(object_name)
    manifest = json.loads(manifest_blob.download_as_text())
    artifact_uri = manifest["artifact_uri"].rstrip("/")

    artifact_bucket, artifact_prefix = _split_gcs_uri(artifact_uri)
    bucket = storage_client.bucket(artifact_bucket)

    with tempfile.TemporaryDirectory() as tmpdir:
        for filename in ("model.joblib", "tfidf_vectorizer.joblib", "metadata.json"):
            bucket.blob(f"{artifact_prefix}/{filename}").download_to_filename(
                os.path.join(tmpdir, filename)
            )

        model = joblib.load(os.path.join(tmpdir, "model.joblib"))
        tfidf = joblib.load(os.path.join(tmpdir, "tfidf_vectorizer.joblib"))
        with open(os.path.join(tmpdir, "metadata.json")) as handle:
            metadata = json.load(handle)

    return model, tfidf, metadata, artifact_uri


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT_ID)


def _pending_batch_ids(client: bigquery.Client) -> list[str]:
    query = f"""
        SELECT batch_id
        FROM `{REQUEST_TABLE}`
        WHERE status = @status
        GROUP BY batch_id
        ORDER BY MIN(created_at)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("status", "STRING", REQUEST_STATUSES["pending"])
        ]
    )
    rows = client.query(query, job_config=job_config).result()
    return [row.batch_id for row in rows]


def _mark_batch_processing(client: bigquery.Client, batch_id: str) -> None:
    query = f"""
        UPDATE `{REQUEST_TABLE}`
        SET status = @processing_status,
            error_message = NULL
        WHERE batch_id = @batch_id
          AND status = @pending_status
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("processing_status", "STRING", REQUEST_STATUSES["processing"]),
            bigquery.ScalarQueryParameter("batch_id", "STRING", batch_id),
            bigquery.ScalarQueryParameter("pending_status", "STRING", REQUEST_STATUSES["pending"]),
        ]
    )
    client.query(query, job_config=job_config).result()


def _load_batch_requests(client: bigquery.Client, batch_id: str) -> pd.DataFrame:
    query = f"""
        SELECT request_id, batch_id, ticker, headlines, as_of_date, created_at
        FROM `{REQUEST_TABLE}`
        WHERE batch_id = @batch_id
          AND status = @status
        ORDER BY created_at, request_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("batch_id", "STRING", batch_id),
            bigquery.ScalarQueryParameter("status", "STRING", REQUEST_STATUSES["processing"]),
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    if df.empty:
        return df

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce").dt.normalize()
    df["headlines"] = df["headlines"].fillna("").astype(str)
    df["ticker"] = df["ticker"].fillna("").astype(str).str.upper()
    return df


def _delete_existing_predictions(client: bigquery.Client, request_ids: list[str]) -> None:
    if not request_ids:
        return

    query = f"""
        DELETE FROM `{PREDICTION_TABLE}`
        WHERE request_id IN UNNEST(@request_ids)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("request_ids", "STRING", request_ids)]
    )
    client.query(query, job_config=job_config).result()


def _append_predictions(client: bigquery.Client, predictions_df: pd.DataFrame) -> None:
    if predictions_df.empty:
        return

    _delete_existing_predictions(client, predictions_df["request_id"].tolist())
    load_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    client.load_table_from_dataframe(
        predictions_df,
        PREDICTION_TABLE,
        job_config=load_config,
    ).result()


def _append_live_feature_snapshots(client: bigquery.Client, rows: list[dict]) -> None:
    if not rows:
        return
    for start in range(0, len(rows), 200):
        batch = rows[start:start + 200]
        errors = client.insert_rows_json(LIVE_FEATURE_SNAPSHOT_TABLE, batch)
        if errors:
            raise RuntimeError(f"Failed to insert live feature snapshots: {errors[:3]}")


def _update_request_statuses(
    client: bigquery.Client,
    request_ids: list[str],
    status: str,
    error_message: str | None = None,
) -> None:
    if not request_ids:
        return

    query = f"""
        UPDATE `{REQUEST_TABLE}`
        SET status = @status,
            processed_at = @processed_at,
            error_message = @error_message
        WHERE request_id IN UNNEST(@request_ids)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("status", "STRING", status),
            bigquery.ScalarQueryParameter("processed_at", "TIMESTAMP", datetime.now(timezone.utc)),
            bigquery.ScalarQueryParameter("error_message", "STRING", error_message),
            bigquery.ArrayQueryParameter("request_ids", "STRING", request_ids),
        ]
    )
    client.query(query, job_config=job_config).result()


def _mark_batch_failed(client: bigquery.Client, batch_id: str, error_message: str) -> None:
    query = f"""
        UPDATE `{REQUEST_TABLE}`
        SET status = @status,
            processed_at = @processed_at,
            error_message = @error_message
        WHERE batch_id = @batch_id
          AND status = @processing_status
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("status", "STRING", REQUEST_STATUSES["failed"]),
            bigquery.ScalarQueryParameter("processed_at", "TIMESTAMP", datetime.now(timezone.utc)),
            bigquery.ScalarQueryParameter("error_message", "STRING", error_message),
            bigquery.ScalarQueryParameter("batch_id", "STRING", batch_id),
            bigquery.ScalarQueryParameter("processing_status", "STRING", REQUEST_STATUSES["processing"]),
        ]
    )
    client.query(query, job_config=job_config).result()


def _validate_live_request_date(as_of_date: pd.Timestamp | pd.NaT) -> str | None:
    if pd.isna(as_of_date):
        return None
    current_date = pd.Timestamp.utcnow()
    if current_date.tzinfo is not None:
        current_date = current_date.tz_convert(None)
    current_date = current_date.normalize()
    if pd.Timestamp(as_of_date).normalize() != current_date:
        return "Live batch inference only supports a null or current as_of_date."
    return None


def _score_request(
    request_row,
    model,
    tfidf,
    metadata: dict,
    artifact_uri: str,
) -> tuple[dict | None, dict | None, str | None]:
    invalid_date_message = _validate_live_request_date(request_row.as_of_date)
    if invalid_date_message:
        return None, None, invalid_date_message

    extra_headlines = normalize_headlines(request_row.headlines)
    snapshot = fetch_live_snapshot(
        request_row.ticker,
        finnhub_key=FINNHUB_KEY,
        extra_headlines=extra_headlines,
    )
    feature_row, feature_snapshot = build_live_feature_bundle(snapshot, tfidf, metadata.get("feature_names"))

    probability = float(model.predict_proba(feature_row)[0, 1])
    label = "Up" if probability >= 0.5 else "Down"
    confidence = probability if probability >= 0.5 else 1.0 - probability

    predicted_at = datetime.now(timezone.utc)
    as_of_ts = pd.Timestamp(snapshot["as_of"])
    as_of_date = as_of_ts.date()
    base_feature_values = feature_row.iloc[0][BASE_FEATURE_NAMES].to_dict()
    headlines_text = feature_snapshot.get("headlines_text") or " ".join(
        item.get("headline", "") for item in feature_snapshot.get("news_items", [])
    ).strip()

    prediction_row = {
        "request_id": request_row.request_id,
        "batch_id": request_row.batch_id,
        "ticker": request_row.ticker,
        "as_of_date": as_of_date,
        "market_feature_date": as_of_date,
        "prediction": label,
        "confidence": round(float(confidence), 4),
        "prob_up": round(float(probability), 4),
        "model_artifact_uri": artifact_uri,
        "model_trained_at": pd.to_datetime(metadata["trained_at"], utc=True),
        "predicted_at": predicted_at,
    }

    snapshot_row = build_live_feature_snapshot_row(
        snapshot_id=build_snapshot_id("batch", request_row.ticker, snapshot["as_of"]),
        request_id=request_row.request_id,
        batch_id=request_row.batch_id,
        ticker=request_row.ticker,
        snapshot_timestamp=snapshot["as_of"],
        headlines_text=headlines_text,
        base_feature_values=base_feature_values,
        snapshot_source=SNAPSHOT_SOURCE_BATCH,
        feature_snapshot=feature_snapshot,
        quote_currency=feature_snapshot.get("quote_currency"),
        model_artifact_uri=artifact_uri,
    )
    return prediction_row, snapshot_row, None


def _process_batch(
    client: bigquery.Client,
    batch_id: str,
    model,
    tfidf,
    metadata: dict,
    artifact_uri: str,
) -> None:
    logger.info("Processing batch %s", batch_id)
    _mark_batch_processing(client, batch_id)
    requests_df = _load_batch_requests(client, batch_id)
    if requests_df.empty:
        logger.info("No pending requests found for batch %s", batch_id)
        return

    predictions: list[dict] = []
    snapshot_rows: list[dict] = []
    failures: dict[str, str] = {}

    for request_row in requests_df.itertuples(index=False):
        try:
            prediction_row, snapshot_row, error_message = _score_request(
                request_row,
                model,
                tfidf,
                metadata,
                artifact_uri,
            )
        except Exception as exc:
            logger.exception("Live batch inference failed for %s: %s", request_row.ticker, exc)
            prediction_row, snapshot_row, error_message = None, None, str(exc)

        if error_message:
            failures[request_row.request_id] = error_message
            continue

        predictions.append(prediction_row)
        snapshot_rows.append(snapshot_row)

    predictions_df = pd.DataFrame(predictions)
    if not predictions_df.empty:
        _append_predictions(client, predictions_df)
        _append_live_feature_snapshots(client, snapshot_rows)
        _update_request_statuses(
            client,
            predictions_df["request_id"].tolist(),
            REQUEST_STATUSES["complete"],
        )

    for request_id, error_message in failures.items():
        _update_request_statuses(
            client,
            [request_id],
            REQUEST_STATUSES["failed"],
            error_message,
        )

    logger.info(
        "Batch %s finished: %d complete, %d failed",
        batch_id,
        len(predictions),
        len(failures),
    )


def main() -> None:
    client = _client()
    model, tfidf, metadata, artifact_uri = _load_model_artifacts(MODEL_MANIFEST_URI)

    feature_names = metadata.get("feature_names", [])
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"Expected a {EXPECTED_FEATURE_COUNT}-feature model artifact, found {len(feature_names)} features."
        )

    pending_batches = _pending_batch_ids(client)
    if not pending_batches:
        logger.info("No pending batch inference requests found.")
        return

    for batch_id in pending_batches:
        try:
            _process_batch(client, batch_id, model, tfidf, metadata, artifact_uri)
        except Exception as exc:
            logger.exception("Batch %s failed: %s", batch_id, exc)
            _mark_batch_failed(client, batch_id, str(exc))


if __name__ == "__main__":
    main()
