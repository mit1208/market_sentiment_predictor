provider "google" {
  project = var.project_id
  region  = var.region
}

# ---------------------------------------------------------------------------
# 0. Enable required GCP APIs
# ---------------------------------------------------------------------------

locals {
  required_apis = [
    "cloudfunctions.googleapis.com",
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudscheduler.googleapis.com",
    "bigquery.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "iam.googleapis.com",
  ]

  batch_job_name          = "market-sentiment-batch-infer"
  training_job_name       = "market-sentiment-train"
  prediction_service_name = "market-sentiment-predict"
  endpoint_image_uri      = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.market_sentiment.repository_id}/endpoint:latest"
  training_image_uri      = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.market_sentiment.repository_id}/pipeline:latest"
}

resource "google_project_service" "apis" {
  for_each           = toset(local.required_apis)
  service            = each.key
  disable_on_destroy = false
}

# ---------------------------------------------------------------------------
# 1. BigQuery Dataset
# ---------------------------------------------------------------------------

resource "google_bigquery_dataset" "main" {
  dataset_id                 = var.dataset_id
  friendly_name              = "Market Sentiment Predictor"
  description                = "Warehouse for ML market sentiment pipeline"
  location                   = "US"
  delete_contents_on_destroy = false

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
# 2. BigQuery Tables
# ---------------------------------------------------------------------------

resource "google_bigquery_table" "raw_market_data" {
  dataset_id = google_bigquery_dataset.main.dataset_id
  table_id   = "raw_market_data"
  schema     = file("${path.module}/raw_market_data.json")

  time_partitioning {
    type  = "DAY"
    field = "date"
  }

  clustering = ["ticker"]
}

resource "google_bigquery_table" "raw_news_data" {
  dataset_id = google_bigquery_dataset.main.dataset_id
  table_id   = "raw_news_data"
  schema     = file("${path.module}/raw_news_data.json")

  time_partitioning {
    type  = "DAY"
    field = "date"
  }

  clustering = ["ticker", "timestamp"]
}

resource "google_bigquery_table" "engineered_features" {
  dataset_id = google_bigquery_dataset.main.dataset_id
  table_id   = "engineered_features"
  schema     = file("${path.module}/engineered_features.json")

  time_partitioning {
    type  = "DAY"
    field = "date"
  }

  clustering = ["ticker"]
}

resource "google_bigquery_table" "model_metadata" {
  dataset_id = google_bigquery_dataset.main.dataset_id
  table_id   = "model_metadata"
  schema     = file("${path.module}/model_metadata.json")
}

resource "google_bigquery_table" "batch_inference_requests" {
  dataset_id = google_bigquery_dataset.main.dataset_id
  table_id   = "batch_inference_requests"
  schema     = file("${path.module}/batch_inference_requests.json")

  time_partitioning {
    type  = "DAY"
    field = "created_at"
  }

  clustering = ["status", "batch_id", "ticker"]
}

resource "google_bigquery_table" "batch_inference_predictions" {
  dataset_id = google_bigquery_dataset.main.dataset_id
  table_id   = "batch_inference_predictions"
  schema     = file("${path.module}/batch_inference_predictions.json")

  time_partitioning {
    type  = "DAY"
    field = "predicted_at"
  }

  clustering = ["batch_id", "ticker"]
}

# ---------------------------------------------------------------------------
# 3. Cloud Storage Buckets
# ---------------------------------------------------------------------------

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "google_storage_bucket" "function_source" {
  name                        = "${var.project_id}-function-source-${random_id.bucket_suffix.hex}"
  location                    = "US"
  uniform_bucket_level_access = true
  force_destroy               = true

  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket" "model_artifacts" {
  name                        = "${var.project_id}-model-artifacts-${random_id.bucket_suffix.hex}"
  location                    = "US"
  uniform_bucket_level_access = true
  force_destroy               = true

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
# 4. Artifact Registry
# ---------------------------------------------------------------------------

resource "google_artifact_registry_repository" "market_sentiment" {
  location      = var.region
  repository_id = "market-sentiment"
  description   = "Market sentiment predictor Docker images"
  format        = "DOCKER"

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
# 5. Zip and upload Cloud Function source
# ---------------------------------------------------------------------------

data "archive_file" "function_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../market_data_ingestion"
  output_path = "${path.module}/function_source.zip"
}

resource "google_storage_bucket_object" "source_zip" {
  name   = "source-${data.archive_file.function_zip.output_md5}.zip"
  bucket = google_storage_bucket.function_source.name
  source = data.archive_file.function_zip.output_path
}

# ---------------------------------------------------------------------------
# 6. IAM Service Account for the pipeline
# ---------------------------------------------------------------------------

resource "google_service_account" "ml_sa" {
  account_id   = "market-sentiment-sa"
  display_name = "Service Account for Market Sentiment ML Pipeline"

  depends_on = [google_project_service.apis]
}

resource "google_project_iam_member" "ml_sa_roles" {
  for_each = toset([
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    # Cloud Run prediction queries need project-level job creation explicitly.
    "roles/bigquery.user",
    "roles/storage.objectAdmin",
    "roles/logging.logWriter",
    "roles/run.developer",
    # Needed so Cloud Scheduler can invoke the Cloud Function (Cloud Run backed)
    "roles/run.invoker",
    # Needed so Cloud Scheduler can trigger the Cloud Run Job
    "roles/cloudscheduler.jobRunner",
  ])
  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.ml_sa.email}"
}

# ---------------------------------------------------------------------------
# 7. Cloud Run Job — model training
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_job" "training" {
  name     = local.training_job_name
  location = var.region

  template {
    template {
      service_account = google_service_account.ml_sa.email
      max_retries     = 0
      timeout         = "3600s"

      containers {
        image   = local.training_image_uri
        command = ["python", "-m", "pipeline.job_runner"]

        env {
          name  = "GCP_PROJECT_ID"
          value = var.project_id
        }

        env {
          name  = "GCP_REGION"
          value = var.region
        }

        env {
          name  = "BQ_DATASET"
          value = var.dataset_id
        }

        env {
          name  = "GCS_BUCKET"
          value = google_storage_bucket.model_artifacts.name
        }

        env {
          name  = "MIN_ACCURACY"
          value = "0.50"
        }

        env {
          name  = "MODEL_PROMOTION_MODE"
          value = "threshold_manifest"
        }

        resources {
          limits = {
            cpu    = "1"
            memory = "4Gi"
          }
        }
      }
    }
  }

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
# 8. Cloud Run Job — batch inference
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_job" "batch_inference" {
  name     = local.batch_job_name
  location = var.region

  template {
    template {
      service_account = google_service_account.ml_sa.email
      max_retries     = 0
      timeout         = "3600s"

      containers {
        image   = local.training_image_uri
        command = ["python", "-m", "pipeline.batch_inference_job"]

        env {
          name  = "GCP_PROJECT_ID"
          value = var.project_id
        }

        env {
          name  = "BQ_DATASET"
          value = var.dataset_id
        }

        env {
          name  = "MODEL_MANIFEST_URI"
          value = "gs://${google_storage_bucket.model_artifacts.name}/models/market_sentiment_lgbm/latest.json"
        }

        env {
          name  = "BATCH_REQUEST_TABLE"
          value = "${var.project_id}.${var.dataset_id}.${google_bigquery_table.batch_inference_requests.table_id}"
        }

        env {
          name  = "BATCH_PREDICTION_TABLE"
          value = "${var.project_id}.${var.dataset_id}.${google_bigquery_table.batch_inference_predictions.table_id}"
        }

        env {
          name  = "BATCH_RUNTIME_VERSION"
          value = "20260313-mergefix"
        }

        resources {
          limits = {
            cpu    = "1"
            memory = "4Gi"
          }
        }
      }
    }
  }

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
# 9. Cloud Run Service — online prediction API
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "prediction_api" {
  name        = local.prediction_service_name
  location    = var.region
  ingress     = "INGRESS_TRAFFIC_ALL"
  description = "Public prediction API backed by the latest trained market sentiment model"

  template {
    service_account                  = google_service_account.ml_sa.email
    timeout                          = "300s"
    max_instance_request_concurrency = 8

    scaling {
      min_instance_count = 1
      max_instance_count = 3
    }

    containers {
      image = local.endpoint_image_uri

      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }

      env {
        name  = "GCP_REGION"
        value = var.region
      }

      env {
        name  = "BQ_DATASET"
        value = var.dataset_id
      }

      env {
        name  = "MODEL_MANIFEST_URI"
        value = "gs://${google_storage_bucket.model_artifacts.name}/models/market_sentiment_lgbm/latest.json"
      }

      env {
        name  = "MODEL_REFRESH_INTERVAL_SECONDS"
        value = "30"
      }

      env {
        name  = "INFERENCE_BACKEND"
        value = "cloud_run"
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "2Gi"
        }
      }
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_service_iam_member" "prediction_public_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.prediction_api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ---------------------------------------------------------------------------
# 10. Cloud Function v2 — data ingestion
# ---------------------------------------------------------------------------

resource "google_cloudfunctions2_function" "function" {
  name        = "market-data-ingestion"
  location    = var.region
  description = "Automated market and news data ingestion that triggers the training job"

  build_config {
    runtime     = "python311"
    entry_point = "run_ingestion"
    source {
      storage_source {
        bucket = google_storage_bucket.function_source.name
        object = google_storage_bucket_object.source_zip.name
      }
    }
  }

  service_config {
    service_account_email = google_service_account.ml_sa.email
    max_instance_count    = 1
    available_memory      = "1Gi"
    available_cpu         = "1"
    timeout_seconds       = 540
    environment_variables = {
      GCP_PROJECT_ID    = var.project_id
      GCP_REGION        = var.region
      BQ_DATASET_ID     = var.dataset_id
      NEWSAPI_KEY       = var.newsapi_key
      TRAINING_JOB_NAME = local.training_job_name
    }
    ingress_settings = "ALLOW_ALL"
  }

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 11. Cloud Scheduler — weekly ingestion (every Sunday at midnight)
# ---------------------------------------------------------------------------

resource "google_cloud_scheduler_job" "weekly_ingestion" {
  name             = "weekly-market-load-tf"
  description      = "Trigger weekly market data ingestion via Cloud Function"
  schedule         = "0 0 * * 0"
  time_zone        = "Etc/UTC"
  attempt_deadline = "600s"

  http_target {
    http_method = "POST"
    uri         = google_cloudfunctions2_function.function.service_config[0].uri

    oidc_token {
      service_account_email = google_service_account.ml_sa.email
    }
  }

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
