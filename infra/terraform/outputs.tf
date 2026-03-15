output "project_id" {
  description = "Target GCP project for the application stack"
  value       = var.project_id
}

output "region" {
  description = "Region used by the application stack"
  value       = var.region
}

output "dataset_id" {
  description = "BigQuery dataset used by the application stack"
  value       = var.dataset_id
}

output "ingestion_function_uri" {
  description = "Ingress URL for the ingestion Cloud Function"
  value       = google_cloudfunctions2_function.function.service_config[0].uri
}

output "function_uri" {
  description = "Compatibility alias for the ingestion Cloud Function URL"
  value       = google_cloudfunctions2_function.function.service_config[0].uri
}

output "function_source_bucket" {
  value = google_storage_bucket.function_source.name
}

output "model_artifacts_bucket" {
  value = google_storage_bucket.model_artifacts.name
}

output "ingestion_scheduler_job" {
  value = google_cloud_scheduler_job.weekly_ingestion.name
}

output "training_job_name" {
  value = google_cloud_run_v2_job.training.name
}

output "feature_rank_submit_job_name" {
  value = google_cloud_run_v2_job.feature_ranking_submit.name
}

output "feature_rank_finalize_job_name" {
  value = google_cloud_run_v2_job.feature_ranking_finalize.name
}

output "batch_inference_job_name" {
  value = google_cloud_run_v2_job.batch_inference.name
}

output "batch_request_table" {
  value = "${var.project_id}.${var.dataset_id}.${google_bigquery_table.batch_inference_requests.table_id}"
}

output "batch_prediction_table" {
  value = "${var.project_id}.${var.dataset_id}.${google_bigquery_table.batch_inference_predictions.table_id}"
}

output "prediction_service_name" {
  value = google_cloud_run_v2_service.prediction_api.name
}

output "prediction_service_url" {
  description = "Public Cloud Run prediction API URL"
  value       = google_cloud_run_v2_service.prediction_api.uri
}

output "artifact_registry_repo" {
  description = "Base path for Docker image tags"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.market_sentiment.repository_id}"
}

output "service_account_email" {
  value = google_service_account.ml_sa.email
}
