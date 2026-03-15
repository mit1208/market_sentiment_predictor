variable "project_id" {
  description = "The GCP project ID"
  type        = string
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

variable "bootstrap_with_placeholder_images" {
  description = "Use a public placeholder image for Cloud Run resources during the very first apply in a new project"
  type        = bool
  default     = false
}

variable "bootstrap_placeholder_image_uri" {
  description = "Public image used only when bootstrap_with_placeholder_images is true"
  type        = string
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
}

variable "finnhub_key" {
  description = "Finnhub API key for market/news fetches"
  type        = string
  default     = ""
  sensitive   = true
}

variable "fred_api_key" {
  description = "FRED API key for macro feature fetches"
  type        = string
  default     = ""
  sensitive   = true
}

variable "feature_rank_notification_webhook" {
  description = "Optional webhook URL notified when AutoML feature ranking is ready for review"
  type        = string
  default     = ""
  sensitive   = true
}
