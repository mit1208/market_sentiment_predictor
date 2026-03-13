-- migrate:up

CREATE TABLE IF NOT EXISTS `live_feature_snapshots` (
  snapshot_id STRING NOT NULL,
  request_id STRING,
  batch_id STRING,
  ticker STRING NOT NULL,
  snapshot_date DATE NOT NULL,
  snapshot_timestamp TIMESTAMP NOT NULL,
  headlines_text STRING,
  current_price FLOAT64,
  previous_close FLOAT64,
  day_open FLOAT64,
  day_high FLOAT64,
  day_low FLOAT64,
  day_change FLOAT64,
  day_change_pct FLOAT64,
  gap_pct FLOAT64,
  intraday_range_pct FLOAT64,
  close_vs_open_pct FLOAT64,
  day_range_position FLOAT64,
  volume FLOAT64,
  avg_volume_10d FLOAT64,
  avg_volume_3m FLOAT64,
  volume_vs_avg_10d FLOAT64,
  volume_vs_avg_3m FLOAT64,
  fifty_day_avg FLOAT64,
  two_hundred_day_avg FLOAT64,
  price_vs_50d_avg FLOAT64,
  price_vs_200d_avg FLOAT64,
  fifty_two_week_high FLOAT64,
  fifty_two_week_low FLOAT64,
  price_vs_52w_high FLOAT64,
  price_vs_52w_low FLOAT64,
  position_52w_range FLOAT64,
  spy_change_pct FLOAT64,
  spy_gap_pct FLOAT64,
  vix_level FLOAT64,
  vix_change_pct FLOAT64,
  sentiment_score FLOAT64,
  headline_count INT64,
  has_news INT64,
  avg_news_age_hours FLOAT64,
  news_source_count INT64,
  target INT64,
  feature_snapshot STRING,
  snapshot_source STRING NOT NULL,
  model_artifact_uri STRING,
  created_at TIMESTAMP NOT NULL
)
PARTITION BY snapshot_date
CLUSTER BY snapshot_source, ticker;

-- migrate:down

DROP TABLE IF EXISTS `live_feature_snapshots`;
