# DevOps Runbook

This runbook covers infrastructure, deployment, service configuration, and operational visibility for the market sentiment pipeline.

## Scope

DevOps owns:

- BigQuery migrations
- Cloud Run Job deployment
- Cloud Function deployment
- Cloud Run service deployment
- IAM and service-account permissions
- Artifact/image rollout
- Operational logging and runtime checks

DevOps does not own:

- feature approval
- selected-feature review
- model promotion decisions beyond platform health

## Pipeline Topology

Runtime components:

- Cloud Function: `market-data-ingestion`
- Cloud Run Job: `market-sentiment-feature-rank-submit`
- Cloud Run Job: `market-sentiment-feature-rank-status`
- Cloud Run Job: `market-sentiment-feature-rank-finalize`
- Cloud Run Job: `market-sentiment-train`
- Cloud Run Job: `market-sentiment-batch-infer`
- Cloud Run Service: `market-sentiment-predict`

Data layers:

- Bronze: `raw_market_data`, `raw_news_data`, `raw_macro_data`
- Silver: `training_feature_snapshots`
- Review: `feature_selection_runs`
- Gold: `gold_training_features`

## Deploy Sequence

Run from repo root:

```bash
make up
make deploy-pipeline
make deploy-function
make deploy-endpoint
```

What each command does:

- `make up`: applies BigQuery migrations
- `make deploy-pipeline`: builds and deploys the Cloud Run Job image used by feature ranking, training, and batch inference
- `make deploy-function`: deploys bronze ingestion
- `make deploy-endpoint`: deploys the live prediction service

## Required Permissions

The pipeline service account must have at least:

- `roles/bigquery.dataEditor`
- `roles/bigquery.jobUser`
- `roles/bigquery.user`
- `roles/storage.objectAdmin`
- `roles/logging.logWriter`
- `roles/run.developer`
- `roles/run.invoker`
- `roles/aiplatform.user`
- `roles/serviceusage.serviceUsageConsumer`
- `roles/cloudscheduler.jobRunner`

These are required so the cloud jobs can:

- read and write BigQuery tables
- call Gemini on Vertex AI
- run AutoML
- upload artifacts to GCS
- execute Cloud Run jobs and services

## Environment Contract

Infra/runtime must provide:

```bash
GCP_PROJECT_ID
GCP_REGION
BQ_DATASET
GCS_BUCKET
FINNHUB_KEY
FRED_API_KEY
FEATURE_RANK_NOTIFICATION_WEBHOOK
```

Notes:

- `FINNHUB_KEY` and `FRED_API_KEY` are required for bronze ingestion and live inference
- `FEATURE_RANK_NOTIFICATION_WEBHOOK` is optional and only used by the finalize job
- training and feature ranking do not call external market/news/macro APIs after bronze ingestion
- inference is the only runtime path allowed to call live APIs

## Operational Checks

Check Cloud Run jobs:

```bash
gcloud run jobs list --region=us-central1
```

Check recent executions:

```bash
gcloud run jobs executions list --job=market-sentiment-feature-rank-submit --region=us-central1
gcloud run jobs executions list --job=market-sentiment-feature-rank-finalize --region=us-central1
gcloud run jobs executions list --job=market-sentiment-train --region=us-central1
```

Check ingestion function:

```bash
gcloud functions describe market-data-ingestion --gen2 --region=us-central1
```

Check endpoint health:

```bash
make verify-endpoint
```

Read logs:

```bash
gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="market-sentiment-feature-rank-submit"' --limit=50
gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="market-sentiment-feature-rank-finalize"' --limit=50
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="market-data-ingestion"' --limit=50
```

## Failure Triage

If ingestion fails:

- check `market-data-ingestion` logs first
- verify BigQuery write schema matches migration state
- verify `FINNHUB_KEY` and `FRED_API_KEY` are present

If feature ranking fails:

- verify `roles/aiplatform.user` and `roles/serviceusage.serviceUsageConsumer`
- inspect Gemini and AutoML permissions
- confirm bronze tables contain fresh rows

If training fails:

- confirm there is an `APPROVED` row in `feature_selection_runs`
- confirm `gold_training_features` can be materialized
- inspect artifact bucket permissions and job logs

## Handoff To MLOps

After deploy is healthy, the MLOps sequence is:

```bash
make ingest
make submit-feature-ranking
# use make check-feature-ranking while Vertex is running
# wait for Vertex AutoML completion or the finalize webhook
make finalize-feature-ranking
```

Then MLOps reviews `feature_selection_runs`, approves a contract, and runs:

```bash
make train
```
