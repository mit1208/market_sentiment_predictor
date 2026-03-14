provider "google" {}

locals {
  bootstrap_apis = [
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "serviceusage.googleapis.com",
  ]
}

resource "google_project" "project" {
  project_id          = var.project_id
  name                = var.project_name
  billing_account     = var.billing_account_id
  folder_id           = var.folder_id
  org_id              = var.organization_id
  auto_create_network = false
  deletion_policy     = "PREVENT"

  lifecycle {
    precondition {
      condition     = !(var.folder_id != null && var.organization_id != null)
      error_message = "Set only one of folder_id or organization_id."
    }
  }
}

resource "google_project_service" "bootstrap_apis" {
  for_each           = toset(local.bootstrap_apis)
  project            = google_project.project.project_id
  service            = each.value
  disable_on_destroy = false

  depends_on = [google_project.project]
}

resource "google_project_iam_member" "friend_access" {
  count   = var.friend_user_email == null ? 0 : 1
  project = google_project.project.project_id
  role    = var.friend_project_role
  member  = "user:${var.friend_user_email}"

  depends_on = [google_project_service.bootstrap_apis]
}
