"""Cloud Run Job entrypoint for finalizing async AutoML feature ranking."""

from __future__ import annotations

import logging
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from backend.config import get_env, get_int_env
from backend.ml.shared.notifications import maybe_send_feature_ranking_webhook
from backend.ml.training.automl_selection import finalize_automl_feature_ranking
from backend.ml.training.automl_selection import persist_feature_ranking_contract
from backend.ml.training.components.train import MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = get_env("GCP_PROJECT_ID")
REGION = get_env("GCP_REGION")
DATASET_ID = get_env("BQ_DATASET", "BQ_DATASET_ID")
GCS_BUCKET = get_env("GCS_BUCKET")
FEATURE_RANK_RUN_ID = get_env("FEATURE_RANK_RUN_ID")
FEATURE_RANK_NOTIFICATION_WEBHOOK = get_env("FEATURE_RANK_NOTIFICATION_WEBHOOK")
AUTO_ML_MAX_SELECTED_FEATURES = get_int_env("AUTO_ML_MAX_SELECTED_FEATURES", default=30) or 30


def _terminate_process(exit_code: int) -> None:
    # Vertex/BigQuery client transports can keep the interpreter alive after the finalize step is done.
    logging.shutdown()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(exit_code)


def main() -> None:
    if not GCS_BUCKET:
        raise RuntimeError("GCS_BUCKET is not configured for the feature-ranking finalize job.")

    logger.info("Starting feature-ranking finalize job")
    contract = finalize_automl_feature_ranking(
        project_id=PROJECT_ID,
        region=REGION,
        dataset_id=DATASET_ID,
        gcs_bucket=GCS_BUCKET,
        model_name=MODEL_NAME,
        run_id=FEATURE_RANK_RUN_ID or None,
        max_selected_features=AUTO_ML_MAX_SELECTED_FEATURES,
    )
    logger.info("Feature ranking run %s now has execution status %s", contract.get("run_id"), contract.get("execution_status"))
    logger.info("Review status: %s", contract.get("review_status"))
    logger.info("Selected features: %s", contract.get("selected_feature_count"))
    logger.info("Review report: %s", contract.get("report_uri"))
    notified, updated_contract = maybe_send_feature_ranking_webhook(
        webhook_url=FEATURE_RANK_NOTIFICATION_WEBHOOK,
        contract=contract,
    )
    if updated_contract is not contract:
        contract = updated_contract
    if contract.get("notification") is not None:
        persist_feature_ranking_contract(
            project_id=PROJECT_ID,
            dataset_id=DATASET_ID,
            contract=contract,
            write_local_mirror=True,
        )
    if notified:
        logger.info("Feature-ranking completion notification sent for run %s", contract.get("run_id"))
    elif FEATURE_RANK_NOTIFICATION_WEBHOOK:
        logger.info(
            "Feature-ranking completion notification not sent for run %s; current notification status=%s",
            contract.get("run_id"),
            (contract.get("notification") or {}).get("status"),
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Feature-ranking finalize job failed")
        _terminate_process(1)
    _terminate_process(0)
