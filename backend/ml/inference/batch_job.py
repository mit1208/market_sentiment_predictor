"""Cloud Run Job entrypoint for live batch inference."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone

import joblib
import pandas as pd
from google.cloud import bigquery, storage

from backend.config import get_env, get_int_env, qualify_table_name
from backend.ml.shared.feature_engineering import (
    build_live_feature_bundle,
    normalize_headlines,
    require_live_news_items,
)
from backend.ml.shared.gemini_sentiment import score_live_news_items
from backend.ml.shared.live_data import fetch_live_snapshot
from backend.ml.shared.live_snapshot_cache import (
    CACHE_REASON_CACHE_FEATURE_MISMATCH,
    CACHE_REASON_DAILY_BUDGET_EXHAUSTED,
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
    SNAPSHOT_SOURCE_BATCH,
    SNAPSHOT_SOURCE_BATCH_CACHE,
    build_live_feature_snapshot_row,
    build_snapshot_id,
)

PROJECT_ID = get_env("GCP_PROJECT_ID")
DATASET_ID = get_env("BQ_DATASET", "BQ_DATASET_ID")
MODEL_MANIFEST_URI = get_env("MODEL_MANIFEST_URI")
REQUEST_TABLE = get_env("BATCH_REQUEST_TABLE")
PREDICTION_TABLE = get_env("BATCH_PREDICTION_TABLE")
LIVE_FEATURE_SNAPSHOT_TABLE = get_env("LIVE_FEATURE_SNAPSHOT_TABLE")
FINNHUB_KEY = get_env("FINNHUB_KEY", "FINNHUB_API_KEY")
FRED_API_KEY = get_env("FRED_API_KEY")
LIVE_SNAPSHOT_FRESH_TTL_SECONDS = get_int_env("LIVE_SNAPSHOT_FRESH_TTL_SECONDS", default=3600) or 3600
LIVE_SNAPSHOT_MAX_STALE_TTL_SECONDS = get_int_env("LIVE_SNAPSHOT_MAX_STALE_TTL_SECONDS", default=21600) or 21600
LIVE_SNAPSHOT_DAILY_LIVE_FETCH_LIMIT = get_int_env("LIVE_SNAPSHOT_DAILY_LIVE_FETCH_LIMIT", default=12) or 12

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

    storage_client = storage.Client(project=PROJECT_ID or None)
    bucket_name, object_name = _split_gcs_uri(manifest_uri)
    manifest_blob = storage_client.bucket(bucket_name).blob(object_name)
    manifest = json.loads(manifest_blob.download_as_text())
    artifact_uri = manifest["artifact_uri"].rstrip("/")

    artifact_bucket, artifact_prefix = _split_gcs_uri(artifact_uri)
    bucket = storage_client.bucket(artifact_bucket)

    with tempfile.TemporaryDirectory() as tmpdir:
        for filename in ("lgb_model.joblib", "xgb_model.joblib", "metadata.json"):
            bucket.blob(f"{artifact_prefix}/{filename}").download_to_filename(os.path.join(tmpdir, filename))

        lgb_model = joblib.load(os.path.join(tmpdir, "lgb_model.joblib"))
        xgb_model = joblib.load(os.path.join(tmpdir, "xgb_model.joblib"))
        with open(os.path.join(tmpdir, "metadata.json")) as handle:
            metadata = json.load(handle)

    return lgb_model, xgb_model, metadata, artifact_uri


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT_ID or None)


def _request_table() -> str:
    return REQUEST_TABLE or qualify_table_name(
        "batch_inference_requests",
        project_id=PROJECT_ID or None,
        dataset_id=DATASET_ID or None,
    )


def _prediction_table() -> str:
    return PREDICTION_TABLE or qualify_table_name(
        "batch_inference_predictions",
        project_id=PROJECT_ID or None,
        dataset_id=DATASET_ID or None,
    )


def _live_feature_snapshot_table() -> str:
    return LIVE_FEATURE_SNAPSHOT_TABLE or qualify_table_name(
        "live_feature_snapshots",
        project_id=PROJECT_ID or None,
        dataset_id=DATASET_ID or None,
    )


def _training_snapshot_table() -> str:
    return qualify_table_name(
        "training_feature_snapshots",
        project_id=PROJECT_ID or None,
        dataset_id=DATASET_ID or None,
    )


def _gemini_cache_table() -> str:
    return qualify_table_name(
        "gemini_sentiment_cache",
        project_id=PROJECT_ID or None,
        dataset_id=DATASET_ID or None,
    )


def _macro_cache_table() -> str:
    return qualify_table_name(
        "macro_feature_cache",
        project_id=PROJECT_ID or None,
        dataset_id=DATASET_ID or None,
    )


def _pending_batch_ids(client: bigquery.Client) -> list[str]:
    query = f"""
        SELECT batch_id
        FROM `{_request_table()}`
        WHERE status = @status
        GROUP BY batch_id
        ORDER BY MIN(created_at)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("status", "STRING", REQUEST_STATUSES["pending"])]
    )
    return [row.batch_id for row in client.query(query, job_config=job_config).result()]


def _mark_batch_processing(client: bigquery.Client, batch_id: str) -> None:
    query = f"""
        UPDATE `{_request_table()}`
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
        FROM `{_request_table()}`
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
    query = f"DELETE FROM `{_prediction_table()}` WHERE request_id IN UNNEST(@request_ids)"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("request_ids", "STRING", request_ids)]
    )
    client.query(query, job_config=job_config).result()


def _append_predictions(client: bigquery.Client, predictions_df: pd.DataFrame) -> None:
    if predictions_df.empty:
        return
    _delete_existing_predictions(client, predictions_df["request_id"].tolist())
    client.load_table_from_dataframe(
        predictions_df,
        _prediction_table(),
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
    ).result()


def _append_live_feature_snapshots(client: bigquery.Client, rows: list[dict]) -> None:
    if not rows:
        return
    for start in range(0, len(rows), 200):
        batch = rows[start:start + 200]
        errors = client.insert_rows_json(_live_feature_snapshot_table(), batch)
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
        UPDATE `{_request_table()}`
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
        UPDATE `{_request_table()}`
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


def _load_cached_prediction_inputs(
    *,
    ticker: str,
    metadata: dict,
    client: bigquery.Client,
) -> tuple[dict | None, int, dict | None]:
    now = datetime.now(timezone.utc)
    daily_live_fetch_count = count_daily_live_fetches(
        client,
        table_name=_live_feature_snapshot_table(),
        ticker=ticker,
        snapshot_date=now,
    )
    cached_snapshot = load_latest_cached_snapshot(
        client,
        table_name=_live_feature_snapshot_table(),
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
            feature_names=list(metadata.get("feature_names") or []),
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
            "snapshot_source": SNAPSHOT_SOURCE_BATCH_CACHE,
        },
        daily_live_fetch_count,
        {
            "cached_snapshot": cached_snapshot,
            "reuse_decision": reuse_decision,
        },
    )


def _score_request(
    request_row,
    lgb_model,
    xgb_model,
    metadata: dict,
    artifact_uri: str,
    client: bigquery.Client,
) -> tuple[dict | None, dict | None, str | None]:
    invalid_date_message = _validate_live_request_date(request_row.as_of_date)
    if invalid_date_message:
        return None, None, invalid_date_message

    extra_headlines = normalize_headlines(request_row.headlines)
    prediction_inputs = None
    daily_live_fetch_count = count_daily_live_fetches(
        client,
        table_name=_live_feature_snapshot_table(),
        ticker=request_row.ticker,
        snapshot_date=datetime.now(timezone.utc),
    )
    cache_context = None
    if not extra_headlines:
        prediction_inputs, daily_live_fetch_count, cache_context = _load_cached_prediction_inputs(
            ticker=request_row.ticker,
            metadata=metadata,
            client=client,
        )

    if prediction_inputs is None:
        reuse_decision = (cache_context or {}).get("reuse_decision") or {}
        if daily_live_fetch_count >= LIVE_SNAPSHOT_DAILY_LIVE_FETCH_LIMIT:
            if extra_headlines:
                return (
                    None,
                    None,
                    (
                        f"Live data budget exhausted for {request_row.ticker}. "
                        "This batch request includes request-specific headlines, so a cached snapshot cannot be reused."
                    ),
                )
            if reuse_decision.get("reason") == CACHE_REASON_DAILY_BUDGET_EXHAUSTED:
                cached_snapshot = (cache_context or {}).get("cached_snapshot") or {}
                snapshot_age_seconds = reuse_decision.get("snapshot_age_seconds")
                snapshot_age_hours = (
                    "unknown"
                    if snapshot_age_seconds is None
                    else f"{round(float(snapshot_age_seconds) / 3600.0, 1)}h"
                )
                return (
                    None,
                    None,
                    (
                        f"Live data budget exhausted for {request_row.ticker}. "
                        f"The latest cached snapshot is {snapshot_age_hours} old and exceeds the {round(LIVE_SNAPSHOT_MAX_STALE_TTL_SECONDS / 3600.0, 1)}h stale limit."
                        + (
                            f" Latest snapshot: {cached_snapshot.get('snapshot_id')}."
                            if cached_snapshot.get("snapshot_id")
                            else ""
                        )
                    ),
                )
            return (
                None,
                None,
                (
                    f"Live data budget exhausted for {request_row.ticker}. "
                    f"The daily live fetch limit is {LIVE_SNAPSHOT_DAILY_LIVE_FETCH_LIMIT} and no reusable cached snapshot is available."
                ),
            )

        snapshot = fetch_live_snapshot(request_row.ticker, finnhub_key=FINNHUB_KEY, extra_headlines=extra_headlines)
        normalized_news = require_live_news_items(snapshot.get("news"), ticker=request_row.ticker)
        sentiment_features = score_live_news_items(
            ticker=request_row.ticker,
            news_items=normalized_news,
            project_id=PROJECT_ID,
            region=get_env("GCP_REGION"),
            cache_client=client,
            cache_table_name=_gemini_cache_table(),
        )
        breadth = load_latest_universe_breadth(
            client,
            table_name=_training_snapshot_table(),
            as_of_date=pd.Timestamp(snapshot["as_of"]),
        )
        macro_features = fetch_live_macro_features(
            fred_api_key=FRED_API_KEY,
            universe_breadth_50d=breadth,
            cache_client=client,
            cache_table_name=_macro_cache_table(),
            as_of_date=pd.Timestamp(snapshot["as_of"]),
        )
        candidate_row, selected_row, feature_snapshot = build_live_feature_bundle(
            snapshot,
            sentiment_features=sentiment_features,
            macro_features=macro_features,
            feature_names=metadata.get("feature_names"),
        )
        feature_snapshot["cache_status"] = build_cache_status(
            source_mode=CACHE_SOURCE_MODE_LIVE_API,
            reason="live_refresh",
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
        prediction_inputs = {
            "candidate_row": candidate_row,
            "selected_row": selected_row,
            "feature_snapshot": feature_snapshot,
            "snapshot_timestamp": pd.Timestamp(snapshot["as_of"]),
            "snapshot_source": SNAPSHOT_SOURCE_BATCH,
        }

    candidate_row = prediction_inputs["candidate_row"]
    selected_row = prediction_inputs["selected_row"]
    feature_snapshot = prediction_inputs["feature_snapshot"]
    snapshot_timestamp = pd.Timestamp(prediction_inputs["snapshot_timestamp"])
    snapshot_source = prediction_inputs["snapshot_source"]

    probability = blend_positive_class_scores(
        lgb_model.predict_proba(selected_row)[0, 1],
        xgb_model.predict_proba(selected_row)[0, 1],
        metadata=metadata,
    )
    label, confidence, _threshold = classify_probability(probability, metadata=metadata)

    predicted_at = datetime.now(timezone.utc)
    as_of_ts = snapshot_timestamp
    as_of_date = as_of_ts.date()
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

    # Embed prediction results into the feature_snapshot metadata for better caching/history
    feature_snapshot["prediction"] = label
    feature_snapshot["confidence"] = round(float(confidence), 4)
    feature_snapshot["prob_up"] = round(float(probability), 4)

    snapshot_row = build_live_feature_snapshot_row(
        snapshot_id=build_snapshot_id(
            "batchcache" if snapshot_source == SNAPSHOT_SOURCE_BATCH_CACHE else "batch",
            request_row.ticker,
            predicted_at if snapshot_source == SNAPSHOT_SOURCE_BATCH_CACHE else snapshot_timestamp,
        ),
        request_id=request_row.request_id,
        batch_id=request_row.batch_id,
        ticker=request_row.ticker,
        snapshot_timestamp=snapshot_timestamp,
        headlines_text=headlines_text,
        base_feature_values=candidate_row.iloc[0].to_dict(),
        snapshot_source=snapshot_source,
        feature_snapshot=feature_snapshot,
        quote_currency=feature_snapshot.get("quote_currency"),
        model_artifact_uri=artifact_uri,
    )
    return prediction_row, snapshot_row, None


def _process_batch(
    client: bigquery.Client,
    batch_id: str,
    lgb_model,
    xgb_model,
    metadata: dict,
    artifact_uri: str,
) -> None:
    _mark_batch_processing(client, batch_id)
    requests_df = _load_batch_requests(client, batch_id)
    if requests_df.empty:
        return

    predictions: list[dict] = []
    snapshot_rows: list[dict] = []
    failures: dict[str, str] = {}

    for request_row in requests_df.itertuples(index=False):
        try:
            prediction_row, snapshot_row, error_message = _score_request(
                request_row,
                lgb_model,
                xgb_model,
                metadata,
                artifact_uri,
                client,
            )
        except Exception as exc:
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
        _update_request_statuses(client, predictions_df["request_id"].tolist(), REQUEST_STATUSES["complete"])

    for request_id, error_message in failures.items():
        _update_request_statuses(client, [request_id], REQUEST_STATUSES["failed"], error_message)


def main() -> None:
    client = _client()
    lgb_model, xgb_model, metadata, artifact_uri = _load_model_artifacts(MODEL_MANIFEST_URI)
    pending_batches = _pending_batch_ids(client)
    if not pending_batches:
        return

    for batch_id in pending_batches:
        try:
            _process_batch(client, batch_id, lgb_model, xgb_model, metadata, artifact_uri)
        except Exception as exc:
            _mark_batch_failed(client, batch_id, str(exc))


if __name__ == "__main__":
    main()
