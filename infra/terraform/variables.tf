variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The region to deploy resources in"
  type        = string
}

variable "dataset_id" {
  description = "The BigQuery dataset ID"
  type        = string
}
