-- migrate:up

CREATE TABLE IF NOT EXISTS `user_prediction_log` (
  user_email   STRING    NOT NULL,
  request_id   STRING    NOT NULL,
  ticker       STRING    NOT NULL,
  predicted_at TIMESTAMP NOT NULL
)
PARTITION BY DATE(predicted_at)
CLUSTER BY user_email, ticker;

-- migrate:down

DROP TABLE IF EXISTS `user_prediction_log`;
