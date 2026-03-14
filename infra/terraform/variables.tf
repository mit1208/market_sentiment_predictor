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
