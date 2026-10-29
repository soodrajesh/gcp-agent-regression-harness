# Runbook 01: deploy

1. `cp deploy.env.example deploy.env` (PROJECT_ID, REGION, ADMIN_EMAIL); `gcloud auth login`.
2. `./scripts/up.sh`: foundation, Cloud Build image, Cloud Run (saved plans; destroys need `ALLOW_DESTROY=1`).
3. `gcloud run services proxy agent-regression --region us-central1 --port 8081`, open http://localhost:8081.
4. Verify: unauthenticated `curl <url>/` is 403; run a comparison in the UI (a 3-run careless-prompt comparison costs about $0.01).
The service is pinned to one instance with CPU always allocated because jobs run after the request returns and results are in memory.
