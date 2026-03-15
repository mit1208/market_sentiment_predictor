-- migrate:up

CREATE TABLE IF NOT EXISTS `raw_macro_data` (
  observation_date DATE NOT NULL,
  series_id STRING NOT NULL,
  value FLOAT64,
  source STRING,
  fetched_at TIMESTAMP NOT NULL
)
PARTITION BY observation_date
CLUSTER BY series_id;

CREATE TABLE IF NOT EXISTS `gold_training_features` (
  materialization_id STRING NOT NULL,
  source_snapshot_id STRING NOT NULL,
  feature_selection_run_id STRING NOT NULL,
  candidate_feature_version STRING NOT NULL,
  selected_feature_count INT64 NOT NULL,
  selected_feature_names ARRAY<STRING>,
  ticker STRING NOT NULL,
  snapshot_date DATE NOT NULL,
  target INT64 NOT NULL,
  next_day_return FLOAT64,
  feature_values STRING NOT NULL,
  created_at TIMESTAMP NOT NULL
)
PARTITION BY snapshot_date
CLUSTER BY feature_selection_run_id, ticker;

-- migrate:down

DROP TABLE IF EXISTS `gold_training_features`;
DROP TABLE IF EXISTS `raw_macro_data`;
