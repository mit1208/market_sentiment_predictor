
--
-- Database schema
--

CREATE TABLE `mkt-sent-friend-260313-225751.market_sentiment_predictor.engineered_features`
(
  date DATE NOT NULL,
  ticker STRING NOT NULL,
  return_1d FLOAT64,
  return_3d FLOAT64,
  return_5d FLOAT64,
  volatility_5d FLOAT64,
  volatility_10d FLOAT64,
  rsi_14 FLOAT64,
  macd FLOAT64,
  macd_signal FLOAT64,
  macd_hist FLOAT64,
  bb_position FLOAT64,
  volume_ratio_5d FLOAT64,
  volume_ratio_10d FLOAT64,
  intraday_range FLOAT64,
  gap FLOAT64,
  price_vs_sma5 FLOAT64,
  price_vs_sma10 FLOAT64,
  obv_slope_5d FLOAT64,
  sentiment_mean FLOAT64,
  sentiment_max FLOAT64,
  sentiment_min FLOAT64,
  sentiment_std FLOAT64,
  sentiment_positive_ratio FLOAT64,
  sentiment_negative_ratio FLOAT64,
  news_count INT64,
  sentiment_momentum FLOAT64,
  news_volume_ratio FLOAT64,
  target INT64
)
PARTITION BY date;
CREATE TABLE `mkt-sent-friend-260313-225751.market_sentiment_predictor.raw_news_data`
(
  date DATETIME,
  timestamp TIMESTAMP,
  ticker STRING,
  headline STRING,
  description STRING,
  source STRING,
  url STRING
)
PARTITION BY DATE(date);
CREATE TABLE `mkt-sent-friend-260313-225751.market_sentiment_predictor.raw_market_data`
(
  date DATETIME,
  ticker STRING,
  open FLOAT64,
  high FLOAT64,
  low FLOAT64,
  close FLOAT64,
  volume INT64
)
PARTITION BY DATE(date);
CREATE VIEW `mkt-sent-friend-260313-225751.market_sentiment_predictor.vw_technical_summary`
AS SELECT
  date,
  ticker,
  ROUND(return_1d, 4) as daily_return,
  ROUND(volatility_5d, 4) as vol_5d,
  ROUND(rsi_14, 2) as rsi,
  ROUND(bb_position, 3) as bollinger_position,
  ROUND(price_vs_sma5, 4) as vs_sma5,
  ROUND(price_vs_sma10, 4) as vs_sma10,
  ROUND(volume_ratio_5d, 2) as vol_ratio
FROM `mkt-sent-friend-260313-225751.market_sentiment_predictor.engineered_features`
WHERE date IS NOT NULL;
CREATE TABLE `mkt-sent-friend-260313-225751.market_sentiment_predictor.model_metadata`
(
  model_id STRING NOT NULL,
  version STRING NOT NULL,
  training_date TIMESTAMP NOT NULL,
  training_start_date DATE,
  training_end_date DATE,
  feature_list ARRAY<STRING>,
  feature_count INT64,
  model_type STRING,
  training_samples INT64,
  hyperparameters JSON,
  performance_metrics JSON,
  validation_metrics JSON,
  test_metrics JSON,
  class_weights JSON,
  feature_importance ARRAY<STRUCT<feature STRING, importance FLOAT64>>,
  data_lineage JSON,
  notes STRING,
  created_by STRING
);
CREATE VIEW `mkt-sent-friend-260313-225751.market_sentiment_predictor.vw_model_performance`
AS SELECT
  model_id,
  version,
  training_date,
  SAFE_CAST(JSON_EXTRACT_SCALAR(test_metrics, '$.accuracy') AS FLOAT64) as test_accuracy,
  SAFE_CAST(JSON_EXTRACT_SCALAR(test_metrics, '$.f1') AS FLOAT64) as test_f1,
  SAFE_CAST(JSON_EXTRACT_SCALAR(test_metrics, '$.precision') AS FLOAT64) as test_precision,
  SAFE_CAST(JSON_EXTRACT_SCALAR(test_metrics, '$.recall') AS FLOAT64) as test_recall,
  feature_count,
  training_samples,
  JSON_EXTRACT_SCALAR(performance_metrics, '$.deployment_status') as deployment_status
FROM `mkt-sent-friend-260313-225751.market_sentiment_predictor.model_metadata`
ORDER BY training_date DESC;
CREATE TABLE `mkt-sent-friend-260313-225751.market_sentiment_predictor.live_feature_snapshots`
(
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
  created_at TIMESTAMP NOT NULL,
  quote_currency STRING
)
PARTITION BY snapshot_date
CLUSTER BY snapshot_source, ticker;
CREATE TABLE `mkt-sent-friend-260313-225751.market_sentiment_predictor.batch_inference_predictions`
(
  request_id STRING,
  batch_id STRING,
  ticker STRING,
  as_of_date DATE,
  predicted_class INT64,
  probability FLOAT64,
  model_version STRING,
  processed_at TIMESTAMP
);
CREATE TABLE `mkt-sent-friend-260313-225751.market_sentiment_predictor.schema_migrations`
(
  version STRING
);
CREATE TABLE `mkt-sent-friend-260313-225751.market_sentiment_predictor.training_feature_snapshots`
(
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
  created_at TIMESTAMP NOT NULL,
  quote_currency STRING
)
PARTITION BY snapshot_date
CLUSTER BY snapshot_source, ticker;
CREATE VIEW `mkt-sent-friend-260313-225751.market_sentiment_predictor.vw_sentiment_daily_summary`
AS SELECT
  date,
  ticker,
  ROUND(sentiment_mean, 4) as avg_sentiment,
  ROUND(sentiment_max, 4) as max_sentiment,
  ROUND(sentiment_min, 4) as min_sentiment,
  ROUND(sentiment_std, 4) as sentiment_volatility,
  ROUND(sentiment_positive_ratio, 3) as pct_positive,
  ROUND(sentiment_negative_ratio, 3) as pct_negative,
  news_count,
  ROUND(news_volume_ratio, 2) as relative_coverage
FROM `mkt-sent-friend-260313-225751.market_sentiment_predictor.engineered_features`
WHERE date IS NOT NULL
  AND sentiment_mean IS NOT NULL;
CREATE TABLE `mkt-sent-friend-260313-225751.market_sentiment_predictor.batch_inference_requests`
(
  request_id STRING,
  batch_id STRING,
  ticker STRING,
  headlines STRING,
  as_of_date DATE,
  status STRING,
  created_at TIMESTAMP,
  processed_at TIMESTAMP,
  error_message STRING
);

--
-- Dbmate schema migrations
--

INSERT INTO schema_migrations (version) VALUES
    ('20260313151204'),
    ('20260313203000'),
    ('20260313214500'),
    ('20260313223000'),
    ('20260313233000');
