-- migrate:up

ALTER TABLE `feature_selection_runs`
ADD COLUMN IF NOT EXISTS execution_status STRING,
ADD COLUMN IF NOT EXISTS started_at TIMESTAMP,
ADD COLUMN IF NOT EXISTS completed_at TIMESTAMP,
ADD COLUMN IF NOT EXISTS error_message STRING;

UPDATE `feature_selection_runs`
SET
  execution_status = IFNULL(execution_status, 'COMPLETED'),
  started_at = IFNULL(started_at, generated_at),
  completed_at = IFNULL(completed_at, generated_at)
WHERE TRUE;

-- migrate:down

ALTER TABLE `feature_selection_runs`
DROP COLUMN error_message,
DROP COLUMN completed_at,
DROP COLUMN started_at,
DROP COLUMN execution_status;
