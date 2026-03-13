variable "project_id" {
  description = "The GCP project ID"
  type        = string
  default     = "llm-testing-489113"
}

variable "region" {
  description = "The region to deploy resources in"
  type        = string
  default     = "us-central1"
}

variable "dataset_id" {
  description = "The BigQuery dataset ID"
  type        = string
  default     = "market_sentiment_predictor"
}

variable "newsapi_key" {
  description = "API key for NewsAPI"
  type        = string
  sensitive   = true
}

