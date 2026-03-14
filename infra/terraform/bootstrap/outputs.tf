output "project_id" {
  description = "Created GCP project ID"
  value       = google_project.project.project_id
}

output "project_number" {
  description = "Created GCP project number"
  value       = google_project.project.number
}

output "project_name" {
  description = "Created GCP project display name"
  value       = google_project.project.name
}

output "project_console_url" {
  description = "Google Cloud Console URL for the new project"
  value       = "https://console.cloud.google.com/home/dashboard?project=${google_project.project.project_id}"
}
