# Frontend

This app calls the server-side `/api/predict` route, which proxies requests to the deployed backend
prediction API.

## Environment

Copy `frontend/.env.local.example` to `frontend/.env.local` and set `PREDICT_API_URL` to the
backend prediction URL.

From the repo root, you can fetch the deployed Cloud Run service URL from Terraform:

```bash
cd infra/terraform
terraform output -raw prediction_service_url
```

Then set:

```bash
PREDICT_API_URL=https://...a.run.app/predict
```

Do not commit `frontend/.env.local`.

## Local Development

From `frontend/`:

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Monorepo Context

- `frontend/src/app/` contains the UI routes.
- `frontend/src/app/api/predict/route.ts` is the proxy to the backend API.
- Backend services live under `../backend/`.
- Infrastructure lives under `../infra/terraform/`.
