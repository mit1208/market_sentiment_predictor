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

Inference is served only from Cloud Run. Vertex AI serving resources are not part of the managed architecture anymore.

## Monorepo Wiring

- Cloud Function source: `../../backend/ingestion`
- Training job entrypoint: `backend.ml.training.job_runner`
- Batch inference job entrypoint: `backend.ml.inference.batch_job`
- Frontend remains a separate deployable app in `../../frontend`

## Importing Existing Resources

Since the resources were previously created manually, you may want to import them into Terraform state to avoid conflicts:

```bash
terraform import google_bigquery_dataset.main llm-testing-489113:market_sentiment_predictor
terraform import google_cloudfunctions2_function.function projects/llm-testing-489113/locations/us-central1/functions/market-data-ingestion
```

Keep `terraform.tfvars` local only. It is ignored by Git because it can contain secrets and
environment-specific values.
