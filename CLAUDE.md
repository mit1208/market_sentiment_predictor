# CLAUDE.md

This file provides guidance to AI agents (Codex, Claude Code, etc.) when working with code in this repository.

## Project Overview

End-to-end market sentiment system: ingest market/news data into BigQuery, rebuild a candidate
feature snapshot table, lock an AutoML-selected feature contract, train an LGB/XGB ensemble, serve
live predictions from Cloud Run, and expose the results through a Next.js frontend.

## Current Baseline Metrics

Latest promoted baseline from BigQuery `model_metadata` (`2026-03-13 20:27:24 UTC`):

- Model version: `2026-03-13T20-27-24.653025+00-00`
- Training window: `2020-01-02` to `2026-03-12`
- Training samples: `7,775`
- Feature count: `50`
- 5-fold CV: accuracy `0.5277 +/- 0.0279`, AUC `0.5036 +/- 0.0185`
- Chronological holdout (`2024-12-11` to `2026-03-11`): accuracy `0.5209`, AUC `0.5336`
- Deployment status: `PROMOTED`
- Dominant features: `day_range_position`, `gap_pct`, `volume_vs_avg_10d`, `intraday_range_pct`
- Current weakness: market-context, sentiment, and TF-IDF features all had zero importance in the promoted baseline

## Repository Layout

```text
market_sentiment_predictor/
├── backend/
│   ├── api/              # Cloud Run FastAPI service
│   ├── ingestion/        # Cloud Function source
│   ├── ml/
│   │   ├── training/     # Training job orchestration + feature snapshot build
│   │   ├── inference/    # Online predictor + batch inference job
│   │   ├── testing/      # ML-focused tests
│   │   └── shared/       # Shared feature engineering and live data helpers
│   ├── db/               # dbmate schema + migrations
│   └── tests/            # Backend service and ingestion tests
├── frontend/             # Next.js app
├── infra/
│   └── terraform/        # GCP infrastructure as code
├── docs/
└── requirements.txt      # Local dev/test Python dependencies
```

## Architecture

```text
Cloud Scheduler
  -> Cloud Function (`backend/ingestion`)
  -> Cloud Run Job (`backend.ml.training.job_runner`)

Cloud Run Job (`backend.ml.training.job_runner`)
  -> rebuild historical candidate feature snapshots
  -> score headlines with Gemini + add FRED macro features
  -> refresh/reuse AutoML feature contract
  -> train LightGBM + XGBoost ensemble
  -> evaluate with time-based CV + holdout
  -> promote latest manifest in GCS

Cloud Run Job (`backend.ml.inference.batch_job`)
  -> score queued batch requests with live features

Cloud Run Service (`backend.api.main`)
  -> fetch live market/news features
  -> score live headlines with Gemini
  -> add live macro context
  -> load promoted ensemble from GCS
  -> return prediction + feature snapshot

Next.js frontend
  -> calls frontend API route
  -> proxies to backend prediction service
```

## Key Technology Choices

| Concern | Choice | Reason |
|---------|--------|--------|
| Workflow | Cloud Function + Cloud Run Jobs | Simple event-driven orchestration for ingest/train/batch inference |
| Text Sentiment | Gemini 2.5 Flash-Lite | Higher-quality financial headline scoring |
| Feature selection | Vertex AI AutoML Tabular | One-time feature pruning for the production contract |
| ML model | LightGBM + XGBoost ensemble | Better variance reduction on selected tabular features |
| Serving | FastAPI on Cloud Run | Pulls promoted artifacts from GCS and computes live features |
| Infra | Terraform | Reproducible, all resources in one place |
| Frontend | Next.js | Simple UI and server-side proxy for prediction requests |

## Dockerfiles

Build from the project root:

```bash
docker build -f backend/api/Dockerfile -t <REGION>-docker.pkg.dev/<PROJECT>/market-sentiment/endpoint:latest .
docker build -f backend/ml/Dockerfile -t <REGION>-docker.pkg.dev/<PROJECT>/market-sentiment/pipeline:latest .
```

## Environment Variables

### Ingestion Function (`backend/ingestion/`)

- `GCP_PROJECT_ID`
- `GCP_REGION`
- `BQ_DATASET_ID`
- `TRAINING_JOB_NAME`
- `FINNHUB_KEY`

### ML Jobs (`backend/ml/`)

- `GCP_PROJECT_ID`
- `GCP_REGION`
- `BQ_DATASET`
- `GCS_BUCKET`
- `MODEL_MANIFEST_URI`
- `FINNHUB_KEY`
- `FRED_API_KEY`

### API Service (`backend/api/`)

- `GCP_PROJECT_ID`
- `GCP_REGION`
- `BQ_DATASET`
- `MODEL_MANIFEST_URI`
- `FINNHUB_KEY`
- `FRED_API_KEY`

### Frontend

- `PREDICT_API_URL`

## Terraform

```bash
cd infra/terraform
terraform init
terraform plan
terraform apply
```

## BigQuery Schema (dataset: `market_sentiment_predictor`)

| Table | Purpose |
|-------|---------|
| `raw_market_data` | Daily OHLCV per ticker, partitioned by `date` |
| `raw_news_data` | News headlines per ticker, partitioned by `date` |
| `training_feature_snapshots` | Historical live-feature-compatible training rows |
| `live_feature_snapshots` | Endpoint and batch inference audit rows |
| `model_metadata` | Training/evaluation metadata |

## Shared Logic

Feature engineering, live snapshot construction, and snapshot persistence live in
`backend/ml/shared/` and are reused by training, batch inference, and the FastAPI service.

## Coding Conventions

- Python services: `GCP_PROJECT_ID`, `GCP_REGION` env vars with sensible defaults
- All Dockerfiles build from the project root
- Terraform resource names follow `google_<resource_type>.<short_name>`
- BigQuery MERGE in ingestion prevents duplicate rows on re-runs
- Backend tests run from the project root with `make test`
