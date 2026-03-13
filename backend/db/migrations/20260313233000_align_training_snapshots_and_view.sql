-- migrate:up

ALTER TABLE `training_feature_snapshots`
ADD COLUMN IF NOT EXISTS quote_currency STRING;

CREATE OR REPLACE VIEW `vw_model_performance`
AS SELECT
  model_id,
  version,
  training_date,
  SAFE_CAST(JSON_EXTRACT_SCALAR(test_metrics, '$.accuracy') AS FLOAT64) AS test_accuracy,
  SAFE_CAST(JSON_EXTRACT_SCALAR(test_metrics, '$.f1') AS FLOAT64) AS test_f1,
  SAFE_CAST(JSON_EXTRACT_SCALAR(test_metrics, '$.precision') AS FLOAT64) AS test_precision,
  SAFE_CAST(JSON_EXTRACT_SCALAR(test_metrics, '$.recall') AS FLOAT64) AS test_recall,
  feature_count,
  training_samples,
  JSON_EXTRACT_SCALAR(performance_metrics, '$.deployment_status') AS deployment_status
FROM `model_metadata`
ORDER BY training_date DESC;

-- migrate:down

DROP VIEW IF EXISTS `vw_model_performance`;

ALTER TABLE `training_feature_snapshots`
DROP COLUMN quote_currency;
