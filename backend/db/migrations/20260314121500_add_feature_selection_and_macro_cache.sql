-- migrate:up

CREATE TABLE IF NOT EXISTS `macro_feature_cache` (
  snapshot_date DATE NOT NULL,
  yield_curve_spread FLOAT64,
  fed_funds_rate FLOAT64,
  fed_funds_change_3m FLOAT64,
  credit_spread FLOAT64,
  dxy_change_5d FLOAT64,
  fetched_at TIMESTAMP NOT NULL
)
PARTITION BY snapshot_date
CLUSTER BY snapshot_date;

CREATE TABLE IF NOT EXISTS `feature_selection_runs` (
  run_id STRING NOT NULL,
  candidate_feature_version STRING NOT NULL,
  candidate_feature_count INT64 NOT NULL,
  selected_feature_names ARRAY<STRING>,
  selected_feature_count INT64 NOT NULL,
  selection_source STRING NOT NULL,
  review_status STRING NOT NULL,
  report_uri STRING,
  contract_uri STRING,
  generated_at TIMESTAMP NOT NULL,
  reviewed_at TIMESTAMP,
  reviewed_by STRING,
  notes STRING,
  contract_json STRING NOT NULL
)
PARTITION BY DATE(generated_at)
CLUSTER BY candidate_feature_version, review_status;

-- migrate:down

DROP TABLE IF EXISTS `feature_selection_runs`;
DROP TABLE IF EXISTS `macro_feature_cache`;
