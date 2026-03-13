# CLAUDE.md

This file provides guidance to AI agents (Codex, Claude Code, etc.) when working with code in this repository.

## Project Overview

End-to-end ML pipeline: ingest financial news + stock data into BigQuery → Vertex AI Pipeline trains LightGBM (TF-IDF for sentiment) → Evaluate → Conditionally Deploy to Vertex AI endpoint → advisor-facing Next.js frontend on Vercel lets users enter tickers + today's headlines and see tomorrow's predicted Up/Down price direction.

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

```
market_sentiment_predictor/
├── cloud_function/       # GCP Cloud Function v2 — weekly ingestion + trigger Vertex AI Pipeline
├── pipeline/             # Vertex AI Pipelines (KFP) code
│   ├── components/       # KFP components: train, evaluate, deploy
│   ├── pipeline.py       # KFP pipeline definition (DAG)
│   ├── Dockerfile        # Base image for all pipeline components
│   └── requirements.txt
├── endpoint/             # Vertex AI custom serving container — FastAPI, loads model from GCS
│   ├── main.py           # POST /predict  GET /health
│   ├── predictor.py      # Loads model + TF-IDF, predicts (inlined features)
│   ├── Dockerfile        # Build from project root: docker build -f endpoint/Dockerfile .
│   └── requirements.txt
├── frontend/             # Next.js 16 app (Vercel) — advisor UI: add tickers + headlines → predict
│   └── src/app/page.tsx  # Single page, calls Vertex AI endpoint directly
├── terraform/            # All GCP infrastructure as code
│   ├── main.tf           # BigQuery, GCS, Artifact Registry, Cloud Function, Vertex AI Endpoint, IAM
│   ├── variables.tf
│   ├── outputs.tf
│   └── *.json            # BigQuery table schemas
└── requirements.txt      # Local dev dependencies (includes kfp)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  GCP (project: llm-testing-489113, region: us-central1)             │
│                                                                     │
│  Cloud Scheduler (Sunday 00:00)                                     │
│    └─► Cloud Function v2 (market-data-ingestion)                   │
│          ├─ yfinance + NewsAPI → BigQuery                           │
│          └─ Submit Vertex AI PipelineJob                            │
│                                                                     │
│  Vertex AI Pipeline (market-sentiment-pipeline)                     │
│    ├─ Step 1: Train (BQ → technical features + TF-IDF → LightGBM)   │
│    ├─ Step 2: Evaluate (Accuracy > 0.50 check)                      │
│    └─ Step 3: Deploy (Conditional: Upload to Model Registry +       │
│                        Update Vertex AI Endpoint)                   │
│                                                                     │
│  Vertex AI Endpoint (market-sentiment-predictor)                    │
│    POST /predict  ← called directly by Vercel frontend              │
│    (FastAPI + TF-IDF vectorizer + LightGBM)                        │
└─────────────────────────────────────────────────────────────────────┘

Vercel (frontend/)
  Advisor enters tickers + today's headlines
    └─► POST to Vertex AI endpoint → Up/Down + confidence
```

## Key Technology Choices

| Concern | Choice | Reason |
|---------|--------|--------|
| Workflow | Vertex AI Pipelines (KFP) | Managed orchestration, artifact lineage, reproducibility |
| Text Sentiment | TF-IDF (Scikit-learn) | Faster, lower cost, no external API calls at inference |
| ML model | LightGBM + TimeSeriesSplit CV | Fast, handles tabular data, respects time ordering |
| Serving | Custom FastAPI container on Vertex AI | Needs live BQ history + TF-IDF vectorization |
| Infra | Terraform | Reproducible, all resources in one place |
| Frontend | Next.js on Vercel | GitHub-linked auto-deploy; calls Vertex AI endpoint directly |

## Dockerfiles — always build from project root

```bash
# Endpoint (serving)
docker build -f endpoint/Dockerfile -t <REGION>-docker.pkg.dev/<PROJECT>/market-sentiment/endpoint:latest .

# Pipeline components (all steps)
docker build -f pipeline/Dockerfile -t <REGION>-docker.pkg.dev/<PROJECT>/market-sentiment/pipeline:latest .
```

## Environment Variables

### Cloud Function (`cloud_function/`)
| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | GCP project ID |
| `GCP_REGION` | GCP region |
| `BQ_DATASET_ID` | BigQuery dataset |
| `NEWSAPI_KEY` | NewsAPI key |
| `GCS_BUCKET` | Bucket for model artifacts |
| `PIPELINE_TEMPLATE_URI` | GCS path to compiled `pipeline.json` |
| `VERTEX_ENDPOINT_ID` | Vertex AI endpoint resource name |
| `SERVING_CONTAINER_IMAGE_URI` | Endpoint image URI |

### Pipeline Components
Stored in `pipeline/components/` and configured in `pipeline.py`:
| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | GCP project ID |
| `GCP_REGION` | GCP region |
| `GCS_BUCKET` | GCS bucket for model artifacts |
| `BQ_DATASET` | BigQuery dataset |
| `VERTEX_ENDPOINT_ID` | Vertex AI endpoint resource name |
| `SERVING_CONTAINER_IMAGE_URI` | Endpoint container image URI |

### Endpoint / Serving (`endpoint/predictor.py`)
Set automatically by Vertex AI at container startup:
| Variable | Description |
|----------|-------------|
| `AIP_STORAGE_URI` | GCS path to model artifacts (`model.joblib`, `tfidf_vectorizer.joblib`, `metadata.json`) |
| `GCP_PROJECT_ID` | GCP project ID |
| `GCP_REGION` | GCP region |
| `BQ_DATASET` | BigQuery dataset |

### Frontend — Vercel environment variables
| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_VERTEX_ENDPOINT_URL` | Vertex AI prediction URL (from `terraform output vertex_endpoint_name`) |

## Terraform

```bash
cd terraform
terraform init
terraform plan  -var="newsapi_key=<KEY>"
terraform apply -var="newsapi_key=<KEY>"
```

## BigQuery Schema (dataset: `market_sentiment_predictor`)

| Table | Purpose |
|-------|---------|
| `raw_market_data` | Daily OHLCV per ticker, partitioned by `date` |
| `raw_news_data` | News headlines per ticker, partitioned by `date` |
| `model_metadata` | Pipeline run results |

## Shared Logic

Technical feature engineering (RSI, MACD, etc.) is **inlined** in both `pipeline/components/train.py` and `endpoint/predictor.py` to simplify deployment and ensure training/serving consistency without complicated shared-file Docker copies.

## Coding Conventions

- Python services: `GCP_PROJECT_ID`, `GCP_REGION` env vars with sensible defaults
- All Dockerfiles build from **project root** (not from their own directory)
- Terraform resource names follow pattern: `google_<resource_type>.<short_name>`
- BigQuery MERGE pattern in Cloud Function prevents duplicate rows on re-runs
- Vertex AI Pipeline compiles to `pipeline.json` for submission

## Deliverables Checklist

- [x] Modular, clean GitHub repository
- [x] Architecture Diagram
- [x] Performance Report populated with the 2026-03-13 promoted baseline metrics
