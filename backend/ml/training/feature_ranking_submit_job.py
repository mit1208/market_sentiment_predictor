"""Cloud Run Job entrypoint for async AutoML feature ranking submission."""

from __future__ import annotations

import logging
import os
import sys

from backend.config import get_env, get_int_env
from backend.ml.shared.feature_engineering import LABEL_NOISE_THRESHOLD
from backend.ml.training.automl_selection import submit_or_reuse_automl_feature_ranking
from backend.ml.training.components.train import MODEL_NAME, prepare_candidate_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = get_env("GCP_PROJECT_ID")
REGION = get_env("GCP_REGION")
DATASET_ID = get_env("BQ_DATASET", "BQ_DATASET_ID")
GCS_BUCKET = get_env("GCS_BUCKET")
AUTO_ML_REFRESH = get_env("AUTO_ML_REFRESH").lower() in {"1", "true", "yes"}
AUTO_ML_MAX_SELECTED_FEATURES = get_int_env("AUTO_ML_MAX_SELECTED_FEATURES", default=30) or 30
AUTO_ML_BUDGET_MILLI_NODE_HOURS = get_int_env("AUTO_ML_BUDGET_MILLI_NODE_HOURS", default=1000) or 1000
AUTO_ML_MIN_ROWS = get_int_env("AUTO_ML_MIN_ROWS", default=1000) or 1000
AUTO_ML_SAMPLE_ROWS = get_int_env("AUTO_ML_SAMPLE_ROWS", default=0) or 0


def _terminate_process(exit_code: int) -> None:
    # Vertex/BigQuery client transports can keep the interpreter alive after the submit step is done.
    logging.shutdown()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(exit_code)


def _apply_automl_sample_limit(dataset, *, sample_rows: int, min_rows: int):
    if sample_rows <= 0:
        return dataset
    if sample_rows < min_rows:
        raise RuntimeError(
            f"AUTO_ML_SAMPLE_ROWS must be at least {min_rows} for Vertex AutoML tabular, got {sample_rows}."
        )

    sorted_dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)
    sampled = sorted_dataset.tail(sample_rows).reset_index(drop=True)
    logger.info(
        "Applying AutoML sample limit: requested=%s actual=%s date_range=%s..%s",
        sample_rows,
        len(sampled),
        sampled["date"].min(),
        sampled["date"].max(),
    )
    return sampled


def main() -> None:
    if not GCS_BUCKET:
        raise RuntimeError("GCS_BUCKET is not configured for the feature-ranking submit job.")

    logger.info("Starting feature-ranking submit job")
    silver, dataset_metadata = prepare_candidate_dataset(rebuild_snapshots=True)
    logger.info("Silver dataset rows before label-noise filter: %s", len(silver))
    filtered = silver[silver["next_day_return"].abs() >= LABEL_NOISE_THRESHOLD].copy()
    if filtered.empty:
        raise RuntimeError("Label-noise filtering removed all rows from the feature-ranking dataset.")
    logger.info("Silver dataset rows after label-noise filter: %s", len(filtered))
    logger.info("AutoML minimum rows: %s", AUTO_ML_MIN_ROWS)
    if AUTO_ML_SAMPLE_ROWS:
        filtered = _apply_automl_sample_limit(
            filtered,
            sample_rows=AUTO_ML_SAMPLE_ROWS,
            min_rows=AUTO_ML_MIN_ROWS,
        )
        logger.info("Silver dataset rows after optional sample limit: %s", len(filtered))
    if len(filtered) < AUTO_ML_MIN_ROWS:
        raise RuntimeError(
            f"AutoML feature ranking requires at least {AUTO_ML_MIN_ROWS} rows after label-noise filtering, "
            f"but only {len(filtered)} rows are available. Refresh bronze history or context backfill before retrying."
        )

    contract = submit_or_reuse_automl_feature_ranking(
        filtered,
        project_id=PROJECT_ID,
        region=REGION,
        dataset_id=DATASET_ID,
        gcs_bucket=GCS_BUCKET,
        model_name=MODEL_NAME,
        refresh=AUTO_ML_REFRESH,
        max_selected_features=AUTO_ML_MAX_SELECTED_FEATURES,
        budget_milli_node_hours=AUTO_ML_BUDGET_MILLI_NODE_HOURS,
    )
    logger.info("Feature ranking run %s submitted with execution status %s", contract.get("run_id"), contract.get("execution_status"))
    logger.info("Vertex AutoML training job: %s", (contract.get("automl") or {}).get("training_job_resource_name"))
    logger.info("Silver dataset rows available for downstream finalize/training: %s", dataset_metadata.get("training_rows"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Feature-ranking submit job failed")
        _terminate_process(1)
    _terminate_process(0)
