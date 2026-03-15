# MLOps Runbook

This runbook covers the data and model lifecycle after infrastructure is deployed.

## Scope

MLOps owns:

- bronze data refresh execution
- silver feature generation
- Gemini feature creation
- AutoML feature ranking
- selected-feature review and approval
- gold table materialization
- training and evaluation
- model artifact review

MLOps does not own:

- infrastructure creation
- IAM provisioning
- Cloud Run or Cloud Function deployment mechanics

## End-To-End Flow

The production training path is:

1. bronze ingestion from APIs into BigQuery
2. silver feature build from bronze tables only
3. async AutoML feature-ranking submission
4. feature-ranking finalization after Vertex completes
5. manual review of selected features
6. gold materialization from the approved contract
7. model training from gold only

## Layer Definitions

Bronze tables:

- `raw_market_data`
- `raw_news_data`
- `raw_macro_data`

Silver table:

- `training_feature_snapshots`

Review table:

- `feature_selection_runs`

Gold table:

- `gold_training_features`

Audit/model tables:

- `live_feature_snapshots`
- `model_metadata`

## Job Responsibilities

`market-data-ingestion`

- fetches raw market/news/macro inputs from APIs
- writes bronze tables only

`market-sentiment-feature-rank-submit`

- reads bronze tables only
- scores text with Gemini
- rebuilds silver candidate features
- submits Vertex AutoML ranking
- exits without waiting for AutoML to finish

`market-sentiment-feature-rank-status`

- polls the submitted Vertex AutoML run
- writes the latest Vertex state into `feature_selection_runs`
- logs whether the run is still active, failed, or ready for finalize

`market-sentiment-feature-rank-finalize`

- checks the submitted Vertex AutoML run
- finalizes selected features after Vertex completes
- writes a pending-review contract to `feature_selection_runs`
- sends an optional completion notification if `FEATURE_RANK_NOTIFICATION_WEBHOOK` is configured

`market-sentiment-train`

- requires an approved feature-selection contract
- materializes gold rows into `gold_training_features`
- trains LightGBM + XGBoost only on approved selected features

## Standard Operating Sequence

From repo root:

```bash
make ingest
make submit-feature-ranking
# use make check-feature-ranking while Vertex is running
# wait for Vertex AutoML completion or the finalize webhook
make finalize-feature-ranking
```

Smoke test:

```bash
make submit-feature-ranking-smoke
```

Use at least `1000` sampled rows. Vertex AutoML tabular rejects smaller datasets.

If a webhook is configured, completion notification happens only after the finalize job sees `execution_status = COMPLETED` and writes `review_status = PENDING_REVIEW`.

After `make finalize-feature-ranking` finishes:

1. inspect the latest row in `feature_selection_runs`
2. review selected features and report URI
3. approve the contract
4. run training

Training command:

```bash
make train
```

## Review Queries

Latest feature-selection runs:

```sql
SELECT
  run_id,
  generated_at,
  execution_status,
  selected_feature_count,
  selected_feature_names,
  review_status,
  report_uri,
  contract_uri
FROM `mkt-sent-friend-260313-225751.market_sentiment_predictor.feature_selection_runs`
ORDER BY generated_at DESC
LIMIT 10;
```

Check silver row count:

```sql
SELECT COUNT(*) AS silver_rows
FROM `mkt-sent-friend-260313-225751.market_sentiment_predictor.training_feature_snapshots`;
```

Check gold row count:

```sql
SELECT COUNT(*) AS gold_rows
FROM `mkt-sent-friend-260313-225751.market_sentiment_predictor.gold_training_features`;
```

## Approval Query

Approve a reviewed feature-selection run:

```sql
UPDATE `mkt-sent-friend-260313-225751.market_sentiment_predictor.feature_selection_runs`
SET
  review_status = 'APPROVED',
  reviewed_at = CURRENT_TIMESTAMP(),
  reviewed_by = 'mlops'
WHERE run_id = 'REPLACE_WITH_RUN_ID';
```

## Training Expectations

Training will:

- refuse to run without an approved contract
- read only approved selected features
- materialize those features into gold
- train on `gold_training_features`, not on bronze or silver directly

Promotion gate:

- `cv_accuracy_mean > 0.52`

Label noise filter:

- rows with `|next_day_return| < 0.003` are dropped before training

## Important Guardrails

- No post-ingestion API calls are allowed for feature ranking or training
- Inference is the only allowed live API path after bronze ingestion
- Gemini is used for feature creation, not for free-form training-time enrichment outside the silver build
- AutoML selection happens before gold materialization
- Gold must always reflect the approved feature contract

## What To Review Before Approving

Review:

- selected feature names
- selected feature count
- candidate feature version
- report URI in GCS
- whether selected features are stable and explainable enough for training

Do not approve if:

- selected features look dominated by unstable or empty columns
- candidate feature version is not the expected one
- report and contract artifacts are missing

## Post-Training Checks

Inspect latest model metadata:

```sql
SELECT
  model_id,
  version,
  training_date,
  feature_count,
  JSON_VALUE(performance_metrics, '$.deployment_status') AS deployment_status
FROM `mkt-sent-friend-260313-225751.market_sentiment_predictor.model_metadata`
ORDER BY training_date DESC
LIMIT 5;
```

Check live manifest in GCS:

```bash
gsutil cat gs://$GCS_BUCKET/models/market_sentiment_lgbm/latest.json
```
