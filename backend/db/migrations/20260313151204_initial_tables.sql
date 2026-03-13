-- migrate:up

-- 1. Raw Market Data
CREATE TABLE IF NOT EXISTS `raw_market_data` (
  date DATETIME,
  ticker STRING,
  open FLOAT64,
  high FLOAT64,
  low FLOAT64,
  close FLOAT64,
  volume INT64
) PARTITION BY DATE(date);

-- 2. Raw News Data
CREATE TABLE IF NOT EXISTS `raw_news_data` (
  date DATETIME,
  timestamp TIMESTAMP,
  ticker STRING,
  headline STRING,
  description STRING,
  source STRING,
  url STRING
) PARTITION BY DATE(date);

-- 3. Model Metadata
CREATE TABLE IF NOT EXISTS `model_metadata` (
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

-- 4. Batch Inference Requests
CREATE TABLE IF NOT EXISTS `batch_inference_requests` (
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

-- 5. Batch Inference Predictions
CREATE TABLE IF NOT EXISTS `batch_inference_predictions` (
  request_id STRING,
  batch_id STRING,
  ticker STRING,
  as_of_date DATE,
  predicted_class INT64,
  probability FLOAT64,
  model_version STRING,
  processed_at TIMESTAMP
);

-- 6. Engineered Features (Gold Table)
CREATE TABLE IF NOT EXISTS `engineered_features` (
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
) PARTITION BY date;

-- migrate:down

DROP TABLE IF EXISTS `raw_market_data`;
DROP TABLE IF EXISTS `raw_news_data`;
DROP TABLE IF EXISTS `model_metadata`;
DROP TABLE IF EXISTS `batch_inference_requests`;
DROP TABLE IF EXISTS `batch_inference_predictions`;
DROP TABLE IF EXISTS `engineered_features`;
