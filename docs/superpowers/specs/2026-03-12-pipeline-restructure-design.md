# Pipeline Restructure Design

## Goal

Restructure the market sentiment predictor to use Vertex AI Pipelines (KFP) for the train → eval → deploy workflow, simplify the folder layout, and replace Gemini-based sentiment scoring with TF-IDF text features.

## Folder Structure

```
market_sentiment_predictor/
├── cloud_function/
│   ├── main.py              # Ingest data → BQ, then trigger Vertex AI Pipeline
│   └── requirements.txt
├── pipeline/
│   ├── pipeline.py          # KFP pipeline DAG: train → evaluate → deploy
│   ├── components/
│   │   ├── train.py         # BQ → technical features + TF-IDF on headlines → LightGBM
│   │   ├── evaluate.py      # Load model, run on test set, output accuracy
│   │   └── deploy.py        # Deploy model to Vertex AI endpoint
│   ├── Dockerfile           # Base image for all pipeline components
│   └── requirements.txt
├── endpoint/
│   ├── main.py              # FastAPI: /health, /predict
│   ├── predictor.py         # Load model + TF-IDF vectorizer, predict
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                # Unchanged Next.js app
│   └── src/app/
│       ├── page.tsx
│       ├── layout.tsx
│       └── api/predict/route.ts
├── terraform/
│   ├── main.tf              # Updated for pipeline workflow
│   ├── variables.tf
│   ├── outputs.tf
│   └── *.json               # BigQuery schemas
├── docs/
├── CLAUDE.md
├── README.md
└── requirements.txt
```

### Deleted

- `ingestion/` — replaced by `cloud_function/` in production
- `warehouse/` — reference docs, no longer needed
- `data/` — local CSV data, not used in pipeline
- `ml/` — absorbed into `pipeline/components/` and `endpoint/`
- `approval_api/` — removed; auto-deploy replaces manual approval workflow
- `DATA_PIPELINE.md` — outdated
- `test_request.json` — sample file, not needed
- `architecture.svg` — will be regenerated

### Shared Code: `feature_engineering.py`

The current `ml/feature_engineering.py` contains `compute_technical_features()` used by both training and serving. With the restructure:

- **`pipeline/components/train.py`**: technical feature computation is inlined (only used here during training)
- **`endpoint/predictor.py`**: technical feature computation is inlined (only used here during inference)
- VADER sentiment functions are removed entirely (replaced by TF-IDF)

Both files implement the same technical feature logic. This is acceptable because:
- The feature list is stable and small
- No shared Dockerfile copy gymnastics needed
- If features change, both must be updated (but this is rare and a pipeline retraining + endpoint redeploy is needed anyway)

### Pipeline Compilation & Upload

`pipeline/pipeline.py` must be compiled to JSON and uploaded to GCS before the Cloud Function can trigger it:

```bash
# Compile pipeline
cd pipeline
python pipeline.py  # outputs pipeline.json

# Upload to GCS
gsutil cp pipeline.json gs://<BUCKET>/pipelines/market-sentiment-pipeline.json
```

This is a manual step during initial setup and whenever the pipeline definition changes. The `PIPELINE_TEMPLATE_URI` env var points to this GCS path.

## Data Flow

```
Cloud Scheduler (Sunday 00:00 UTC)
  └─► Cloud Function v2 (market-data-pipeline)
        ├─ yfinance → raw_market_data (BigQuery)
        ├─ NewsAPI → raw_news_data (BigQuery)
        └─ On success: trigger Vertex AI Pipeline run
              │
              ├─ STEP 1: Train
              │   ├─ Read raw_market_data + raw_news_data from BQ
              │   ├─ Compute technical features (RSI, MACD, Bollinger, etc.)
              │   ├─ TF-IDF on concatenated daily headlines per ticker
              │   ├─ Train LightGBM (TimeSeriesSplit CV)
              │   ├─ Save model.joblib + tfidf_vectorizer.joblib + metadata.json → GCS
              │   └─ Output: GCS artifact URI
              │
              ├─ STEP 2: Evaluate
              │   ├─ Load model + vectorizer from GCS
              │   ├─ Run on held-out test split
              │   ├─ Compute accuracy, AUC
              │   └─ Output: accuracy metric
              │
              └─ STEP 3: Deploy (conditional: accuracy > 0.50)
                  ├─ Upload model to Vertex AI Model Registry
                  ├─ Deploy to Vertex AI Endpoint
                  └─ Write metadata to BQ model_metadata table
```

## Training: Text Features

Instead of Gemini Flash scoring:

1. Concatenate all headlines for a (ticker, date) into a single string
2. Apply TF-IDF vectorization (max 100 features, unigrams + bigrams)
3. Combine TF-IDF features with technical indicators (RSI, MACD, Bollinger, volume ratios, etc.)
4. Train LightGBM on combined feature matrix
5. Save fitted TF-IDF vectorizer (`tfidf_vectorizer.joblib`) alongside model

**Important:** TF-IDF vectorizer is fit only on the training split (not the eval/test split) to avoid data leakage. The fitted vectorizer is saved and used to transform both the eval split and inference-time headlines.

### Technical Features (unchanged)

- Returns: 1d, 3d, 5d percent change
- Volatility: 5d, 10d rolling std dev
- Momentum: RSI(14), MACD(12,26,9), MACD signal, MACD histogram
- Bollinger: position within bands
- Volume: ratio_5d, ratio_10d, OBV slope
- Price: vs SMA(5), SMA(10)
- Intraday: range, overnight gap

### Target

Binary: 1 = next day close > today's close, 0 = down/flat

## Endpoint Changes

`predictor.py` updated to:

- Load `tfidf_vectorizer.joblib` alongside `model.joblib` at startup
- Apply TF-IDF to user-supplied headlines (replaces Gemini call)
- Keep technical feature computation (fetch 60 days OHLCV from BQ)
- Remove all Gemini/google-generativeai dependencies

## Cloud Function Changes

`cloud_function/main.py` updated to:

- Keep existing ingestion logic (yfinance + NewsAPI → BQ MERGE)
- Remove `trigger_training()` (was for Cloud Run job)
- Add `trigger_pipeline()`: submit Vertex AI Pipeline run via `aiplatform.PipelineJob`
- Call `trigger_pipeline()` at end of `run_pipeline()`

## Terraform Changes

- Remove: `trigger_training` Cloud Function, second Cloud Scheduler (`weekly-training-tf`), any Cloud Run job resources
- Remove: `GEMINI_MODEL` env var from endpoint container config
- Keep: BigQuery dataset/tables, GCS bucket, Artifact Registry, Vertex AI Endpoint, `market-data-pipeline` Cloud Function, service account, first Cloud Scheduler
- Add: IAM permissions for Cloud Function service account to submit Vertex AI Pipeline runs (`roles/aiplatform.user`)
- Add: New env vars to Cloud Function: `GCP_REGION`, `GCS_BUCKET`, `PIPELINE_TEMPLATE_URI`, `VERTEX_ENDPOINT_ID`, `SERVING_CONTAINER_IMAGE_URI`
- Update: Cloud Function source to include new pipeline trigger code
- Keep `engineered_features` BQ table (pipeline can optionally write to it for audit)
- `model_metadata` table: remove `production_ready` column since auto-deploy replaces approval workflow; accuracy threshold (0.50) is a pipeline parameter

### Deploy Threshold

The accuracy > 0.50 threshold is passed as a KFP pipeline parameter (`min_accuracy`), not hardcoded. The Cloud Function passes it when submitting the pipeline run, and it can be overridden via the `MIN_ACCURACY` env var on the Cloud Function (default: 0.50).

### Pipeline Docker Image

A single Docker image (`pipeline/Dockerfile`) is used for all three KFP components. Built from project root:

```bash
docker build -f pipeline/Dockerfile -t <REGION>-docker.pkg.dev/<PROJECT>/market-sentiment/pipeline:latest .
```

Base image: `python:3.11-slim`. Key dependencies: `kfp`, `lightgbm`, `scikit-learn`, `google-cloud-bigquery`, `google-cloud-aiplatform`, `google-cloud-storage`, `pandas`, `joblib`.

## Environment Variables

### Cloud Function

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | GCP project ID |
| `BQ_DATASET_ID` | BigQuery dataset |
| `NEWSAPI_KEY` | NewsAPI key |
| `GCP_REGION` | GCP region (for pipeline submission) |
| `GCS_BUCKET` | GCS bucket for pipeline artifacts |
| `PIPELINE_TEMPLATE_URI` | GCS path to compiled pipeline JSON |
| `VERTEX_ENDPOINT_ID` | Vertex AI endpoint resource name |
| `SERVING_CONTAINER_IMAGE_URI` | Endpoint container image URI |

### Pipeline Components

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | GCP project ID |
| `GCP_REGION` | GCP region |
| `GCS_BUCKET` | GCS bucket for model artifacts |
| `BQ_DATASET` | BigQuery dataset |
| `VERTEX_ENDPOINT_ID` | Vertex AI endpoint resource name |
| `SERVING_CONTAINER_IMAGE_URI` | Endpoint container image URI |

### Endpoint

| Variable | Description |
|----------|-------------|
| `AIP_STORAGE_URI` | GCS path to model artifacts |
| `GCP_PROJECT_ID` | GCP project ID |
| `GCP_REGION` | GCP region |
| `BQ_DATASET` | BigQuery dataset |

### Frontend (Vercel)

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_VERTEX_ENDPOINT_URL` | Vertex AI prediction URL |
