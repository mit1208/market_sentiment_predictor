"""Cloud Run job entrypoint for silver-to-gold training ETL."""

from __future__ import annotations

import logging

from backend.ml.training.components.train import run_gold_training_etl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting silver-to-gold training ETL job")
    _gold, dataset_metadata = run_gold_training_etl(rebuild_gold=True)
    logger.info(
        "Prepared gold training data: rows=%s selected_features=%s source=%s temporary_fallback=%s date_range=%s..%s",
        dataset_metadata.get("training_rows"),
        dataset_metadata.get("selected_feature_count"),
        dataset_metadata.get("feature_selection_source"),
        dataset_metadata.get("feature_selection_is_temporary"),
        dataset_metadata.get("training_start_date"),
        dataset_metadata.get("training_end_date"),
    )


if __name__ == "__main__":
    main()
