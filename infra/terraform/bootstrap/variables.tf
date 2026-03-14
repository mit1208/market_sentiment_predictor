variable "project_id" {
  description = "Globally unique GCP project ID to create"
  type        = string
}

variable "project_name" {
  description = "Human-readable display name for the GCP project"
  type        = string
}

variable "billing_account_id" {
  description = "Billing account ID to attach to the new project"
  type        = string
}

variable "folder_id" {
  description = "Optional folder ID to place the project under"
  type        = string
  default     = null
  nullable    = true
}

variable "organization_id" {
  description = "Optional organization ID to place the project under when folder_id is not used"
  type        = string
  default     = null
  nullable    = true
}

variable "friend_user_email" {
  description = "Optional Google account email to grant project access immediately"
  type        = string
  default     = null
  nullable    = true
}

variable "friend_project_role" {
  description = "Project role to grant to friend_user_email when provided"
  type        = string
  default     = "roles/editor"
}
