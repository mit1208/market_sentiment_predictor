-- migrate:up

ALTER TABLE `live_feature_snapshots`
ADD COLUMN IF NOT EXISTS quote_currency STRING;

-- migrate:down

ALTER TABLE `live_feature_snapshots`
DROP COLUMN quote_currency;
