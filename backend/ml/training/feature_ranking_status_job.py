"""Cloud Run Job entrypoint for polling async AutoML feature ranking status."""

from __future__ import annotations

import logging
import os
import sys

from backend.config import get_env
from backend.ml.training.automl_selection import refresh_automl_feature_ranking_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = get_env("GCP_PROJECT_ID")
REGION = get_env("GCP_REGION")
DATASET_ID = get_env("BQ_DATASET", "BQ_DATASET_ID")
GCS_BUCKET = get_env("GCS_BUCKET")
FEATURE_RANK_RUN_ID = get_env("FEATURE_RANK_RUN_ID")


def _terminate_process(exit_code: int) -> None:
    logging.shutdown()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(exit_code)


def main() -> None:
    if not GCS_BUCKET:
        raise RuntimeError("GCS_BUCKET is not configured for the feature-ranking status job.")

    logger.info("Starting feature-ranking status job")
    contract = refresh_automl_feature_ranking_status(
        project_id=PROJECT_ID,
        region=REGION,
        dataset_id=DATASET_ID,
        gcs_bucket=GCS_BUCKET,
        run_id=FEATURE_RANK_RUN_ID or None,
        write_local_mirror=True,
    )
    automl = contract.get("automl") or {}
    logger.info("Feature ranking run: %s", contract.get("run_id"))
    logger.info("Execution status: %s", contract.get("execution_status"))
    logger.info("Review status: %s", contract.get("review_status"))
    logger.info("Vertex state: %s", automl.get("vertex_training_job_state"))
    logger.info("Training job: %s", automl.get("training_job_resource_name"))
    logger.info("Dataset resource: %s", automl.get("dataset_resource_name"))
    logger.info("Dataset source: %s", automl.get("dataset_bq_source"))
    logger.info("Model resource: %s", automl.get("model_resource_name"))
    logger.info("Selected feature count: %s", contract.get("selected_feature_count"))
    logger.info("Notes: %s", contract.get("notes"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Feature-ranking status job failed")
        _terminate_process(1)
    _terminate_process(0)
