"""
Cloud Run Job entrypoint for event-driven training.

Reads market/news data from BigQuery, trains the model, stores artifacts in GCS,
and promotes a new live model only when it clears the deployment threshold.
"""

import json
import logging
import os

from google.cloud import bigquery, storage

from backend.config import get_env, get_float_env, qualify_table_name
from backend.ml.training.components.train import (
    MODEL_NAME,
    prepare_training_dataset,
    save_artifacts,
    train_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = get_env("GCP_PROJECT_ID")
REGION = get_env("GCP_REGION")
DATASET_ID = get_env("BQ_DATASET", "BQ_DATASET_ID")
GCS_BUCKET = get_env("GCS_BUCKET")
MIN_ACCURACY = get_float_env("MIN_ACCURACY", default=0.50) or 0.50


def _model_metadata_table() -> str:
    return qualify_table_name(
        "model_metadata",
        project_id=PROJECT_ID or None,
        dataset_id=DATASET_ID or None,
    )


def _json_cell(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def _upload_json(bucket_name: str, object_name: str, payload: dict) -> str:
    storage_client = storage.Client(project=PROJECT_ID or None)
    blob = storage_client.bucket(bucket_name).blob(object_name)
    blob.upload_from_string(json.dumps(payload, indent=2), content_type="application/json")
    return f"gs://{bucket_name}/{object_name}"


def _write_model_metadata(metrics: dict, artifact_uri: str, deployment_status: str) -> None:
    performance_metrics = {
        **metrics["performance_metrics"],
        "deployment_status": deployment_status,
    }
    data_lineage = {
        "artifact_uri": artifact_uri,
        "market_table": qualify_table_name(
            "raw_market_data",
            project_id=PROJECT_ID or None,
            dataset_id=DATASET_ID or None,
        ),
        "news_table": qualify_table_name(
            "raw_news_data",
            project_id=PROJECT_ID or None,
            dataset_id=DATASET_ID or None,
        ),
        "snapshot_table": metrics["snapshot_table"],
        "tickers": metrics["tickers"],
        "ticker_count": metrics["ticker_count"],
    }
    row = {
        "model_id": MODEL_NAME,
        "version": metrics["trained_at"].replace(":", "-"),
        "training_date": metrics["trained_at"],
        "training_start_date": metrics["training_start_date"],
        "training_end_date": metrics["training_end_date"],
        "feature_list": metrics["feature_names"],
        "feature_count": metrics["n_features"],
        "model_type": metrics["model_type"],
        "training_samples": metrics["n_train_rows"],
        "hyperparameters": _json_cell(metrics["hyperparameters"]),
        "performance_metrics": _json_cell(performance_metrics),
        "validation_metrics": _json_cell(metrics["validation_metrics"]),
        "test_metrics": _json_cell(metrics["test_metrics"]),
        "class_weights": _json_cell(metrics["class_weights"]),
        "feature_importance": metrics["feature_importance"],
        "data_lineage": _json_cell(data_lineage),
        "notes": metrics["evaluation_notes"],
        "created_by": "backend/ml/training/job_runner.py",
    }

    model_metadata_table = _model_metadata_table()
    client = bigquery.Client(project=PROJECT_ID or None)
    errors = client.insert_rows_json(model_metadata_table, [row])
    if errors:
        logger.error("Failed to write model metadata: %s", errors)
    else:
        logger.info("Wrote model evaluation metadata to %s", model_metadata_table)


def main() -> None:
    if not GCS_BUCKET:
        raise RuntimeError("GCS_BUCKET is not configured for the training job.")

    logger.info("Starting event-driven training job")
    gold, dataset_metadata = prepare_training_dataset()
    model, tfidf_vectorizer, metrics = train_model(gold, dataset_metadata)
    artifact_uri = save_artifacts(model, tfidf_vectorizer, metrics)
    promoted = float(metrics["cv_acc_mean"]) > MIN_ACCURACY

    manifest = {
        "artifact_uri": artifact_uri,
        "accuracy": metrics["cv_acc_mean"],
        "auc": metrics["cv_auc_mean"],
        "holdout_accuracy": metrics["test_metrics"]["accuracy"],
        "holdout_auc": metrics["test_metrics"]["auc"],
        "project_id": PROJECT_ID,
        "region": REGION,
        "dataset_id": DATASET_ID,
        "trained_at": metrics["trained_at"],
    }

    _write_model_metadata(
        metrics,
        artifact_uri,
        "PROMOTED" if promoted else "NOT_PROMOTED",
    )

    if not promoted:
        logger.info(
            "Accuracy %.4f did not clear threshold %.4f. Keeping the current live model unchanged.",
            metrics["cv_acc_mean"],
            MIN_ACCURACY,
        )
        return

    latest_uri = _upload_json(
        GCS_BUCKET,
        f"models/{MODEL_NAME}/latest.json",
        manifest,
    )
    logger.info("Promoted trained model to live manifest %s", latest_uri)


if __name__ == "__main__":
    main()
