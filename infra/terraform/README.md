# Terraform Infrastructure for Market Sentiment Predictor

This directory contains the Terraform configuration for the event-driven market sentiment workflow.
In the monorepo, application code lives under `backend/` and `frontend/`, while this folder owns the
GCP infrastructure and packaging glue between them.

## Resources Managed

1.  **BigQuery Dataset**: `market_sentiment_predictor`
2.  **BigQuery Tables**:
    *   `raw_market_data` (Partitioned by date, Clustered by ticker)
    *   `raw_news_data` (Partitioned by date, Clustered by ticker/timestamp)
    *   `engineered_features` (Partitioned by date, Clustered by ticker)
    *   `model_metadata`
    *   `batch_inference_requests`
    *   `batch_inference_predictions`
3.  **Cloud Storage**:
    *   Function source bucket
    *   Model artifact bucket
4.  **Cloud Run Job**: `market-sentiment-train`
5.  **Cloud Run Job**: `market-sentiment-batch-infer`
6.  **Cloud Run Service**: `market-sentiment-predict`
7.  **Cloud Function (Gen 2)**: `market-data-ingestion`
8.  **Cloud Scheduler**: `weekly-market-load-tf`

## Prerequisites

*   Terraform installed (`>= 1.0`)
*   Google Cloud SDK authenticated (`gcloud auth application-default login`)
*   A local `terraform.tfvars` created from `terraform.tfvars.example`

If you need a brand new GCP project, bootstrap it first from [`bootstrap/`](./bootstrap).

## Usage

1.  Initialize Terraform:
    ```bash
    cd infra/terraform
    cp terraform.tfvars.example terraform.tfvars
    terraform init
    ```

2.  Plan the changes:
    ```bash
    terraform plan
    ```

3.  Apply the configuration:
    ```bash
    terraform apply
    ```

For the very first apply in a brand-new project, set `bootstrap_with_placeholder_images = true`
so Terraform can create the Cloud Run jobs and service before your own Docker images exist in
Artifact Registry. After you push real images with the root `Makefile`, switch it back to `false`
or omit it.

After apply, the root [`Makefile`](/Users/mitpatel/Desktop/code-exam/market_sentiment_predictor/Makefile) reads `project_id`, `region`, `artifact_registry_repo`, and `model_artifacts_bucket` from Terraform outputs when available, so deploy commands target the provisioned project automatically.

Inference is served only from Cloud Run. Vertex AI serving resources are not part of the managed architecture anymore.

## Monorepo Wiring

- Cloud Function source: `../../backend/ingestion`
- Training job entrypoint: `backend.ml.training.job_runner`
- Batch inference job entrypoint: `backend.ml.inference.batch_job`
- Frontend remains a separate deployable app in `../../frontend`

## Importing Existing Resources

Since the resources were previously created manually, you may want to import them into Terraform state to avoid conflicts:

```bash
terraform import google_bigquery_dataset.main mkt-sent-friend-260313-225751:market_sentiment_predictor
terraform import google_cloudfunctions2_function.function projects/mkt-sent-friend-260313-225751/locations/us-central1/functions/market-data-ingestion
```

Keep `terraform.tfvars` local only. It is ignored by Git because it can contain secrets and
environment-specific values.
