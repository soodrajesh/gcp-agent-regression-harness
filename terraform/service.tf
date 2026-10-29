resource "google_service_account" "app" {
  account_id   = "ar-app"
  display_name = "agent-regression runtime"
}
resource "google_project_iam_member" "app_vertex" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.app.email}"
}
resource "google_project_iam_member" "app_logs" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.app.email}"
}
resource "google_cloud_run_v2_service" "app" {
  count               = var.image != "" ? 1 : 0
  name                = "agent-regression"
  location            = var.region
  deletion_protection = false
  labels              = { env = "demo", owner = "platform" }
  template {
    service_account                  = google_service_account.app.email
    timeout                          = "300s"
    max_instance_request_concurrency = 20
    scaling {
      min_instance_count = 0
      max_instance_count = 1 # runs and results live in memory, so there must be one instance
    }
    containers {
      image = var.image
      resources {
        limits   = { cpu = "1", memory = "512Mi" }
        cpu_idle = false # comparison jobs keep running after the request that started them returns
      }
      startup_probe {
        http_get { path = "/healthz" }
      }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "AGENTREG_ALLOWED_PROVIDERS"
        value = "gemini,mock"
      }
      env {
        name  = "AGENTREG_MAX_RUNS_PER_HOUR"
        value = "20"
      }
    }
  }
  depends_on = [
    google_project_iam_member.app_vertex,
  ]
}

# Private: only the admin may invoke. No allUsers binding.
resource "google_cloud_run_v2_service_iam_member" "admin_invoker" {
  count    = var.image != "" ? 1 : 0
  name     = google_cloud_run_v2_service.app[0].name
  location = var.region
  role     = "roles/run.invoker"
  member   = "user:${var.admin_email}"
}
