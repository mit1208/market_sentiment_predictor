
--
-- Database schema
--

CREATE VIEW `llm-testing-489113.market_sentiment_predictor.vw_technical_summary`
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
FROM `llm-testing-489113.market_sentiment_predictor.engineered_features`
WHERE date IS NOT NULL;
CREATE VIEW `llm-testing-489113.market_sentiment_predictor.vw_model_performance`
AS SELECT
  model_id,
  version,
  training_date,
  JSON_EXTRACT_SCALAR(test_metrics, '$.accuracy') as test_accuracy,
  JSON_EXTRACT_SCALAR(test_metrics, '$.f1') as test_f1,
  JSON_EXTRACT_SCALAR(test_metrics, '$.precision') as test_precision,
  JSON_EXTRACT_SCALAR(test_metrics, '$.recall') as test_recall,
  feature_count,
  training_samples,
  production_ready
FROM `llm-testing-489113.market_sentiment_predictor.model_metadata`
ORDER BY training_date DESC;
CREATE TABLE `llm-testing-489113.market_sentiment_predictor.schema_migrations`
(
  version STRING
);
CREATE VIEW `llm-testing-489113.market_sentiment_predictor.vw_sentiment_daily_summary`
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
FROM `llm-testing-489113.market_sentiment_predictor.engineered_features`
WHERE date IS NOT NULL
  AND sentiment_mean IS NOT NULL;

--
-- Dbmate schema migrations
--

