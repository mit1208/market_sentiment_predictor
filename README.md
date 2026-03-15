# Market Sentiment Predictor

End-to-end market direction prediction system built on a **bronze → silver → gold** medallion ETL architecture, with an offline/online feature store, LightGBM + XGBoost ensemble model, and a cache-aware live prediction endpoint.

## Architecture

```text
                          ┌──────────────────────────────────────────┐
                          │            BRONZE (Raw Data)             │
                          │                                          │
  Finnhub ─────────┐      │  raw_market_data   (daily OHLCV)        │
  Yahoo Finance ───┤──────│  raw_news_data     (headlines)           │
  FRED API ────────┘      │  raw_macro_data    (economic indicators) │
                          └──────────────┬───────────────────────────┘
                                         │
                          ┌──────────────▼───────────────────────────┐
                          │           SILVER (Processed)             │
                          │                                          │
  Gemini 2.5 Flash-Lite ──│  training_feature_snapshots              │
                          │  (55 candidate features + Gemini scores  │
                          │   + macro context + target labels)       │
                          │  gemini_sentiment_cache                  │
                          └──────────────┬───────────────────────────┘
                                         │
                          ┌──────────────▼───────────────────────────┐
                          │       FEATURE SELECTION (AutoML)         │
                          │                                          │
  Vertex AI AutoML ───────│  feature_selection_runs                  │
                          │  selected_features.json (contract)       │
                          └──────────────┬───────────────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │                                           │
        ┌──────────▼────────────┐               ┌──────────────▼──────────────┐
        │  OFFLINE FEATURE STORE │               │   ONLINE FEATURE STORE      │
        │  (Training)            │               │   (Endpoint / Batch)        │
        │                        │               │                             │
        │  gold_training_features│               │  Shared feature builder     │
        │  (approved features    │               │  (feature_engineering.py)   │
        │   materialized from    │               │  Live APIs → same features  │
        │   silver + contract)   │               │  + live_feature_snapshots   │
        └──────────┬─────────────┘               └──────────────┬──────────────┘
                   │                                            │
        ┌──────────▼─────────────┐               ┌──────────────▼──────────────┐
        │  MODEL TRAINING        │               │   LIVE PREDICTION           │
        │                        │               │                             │
        │  LGB + XGB ensemble    │               │  Load promoted model        │
        │  Optuna tuning         │               │  Score with ensemble        │
        │  → GCS artifacts       │               │  Cache snapshots in BQ      │
        │  → model_metadata      │               │  Serve via Cloud Run        │
        └────────────────────────┘               └─────────────────────────────┘
```

## ETL Pipeline

### 1. Bronze — Raw Data Ingestion

```bash
make ingest
```

A Cloud Function fetches raw data from external APIs and loads it into BigQuery using MERGE statements to prevent duplicates:

| Source | Table | Unique Keys |
|--------|-------|-------------|
| yfinance | `raw_market_data` | `date`, `ticker` |
| Finnhub + Yahoo | `raw_news_data` | `url`, `ticker` |
| FRED | `raw_macro_data` | `observation_date`, `series_id` |

Market data includes training tickers plus SPY and ^VIX for context features.

### 2. Silver — Feature Engineering + Gemini Scoring

```bash
make submit-feature-ranking
```

Reads bronze tables and builds the silver candidate feature table `training_feature_snapshots`:

- **25 market features** from price history (OHLCV ratios, moving averages, 52-week position)
- **10 temporal features** from returns and momentum (1d/3d/5d/10d/20d returns, volatility regime, relative strength vs SPY)
- **9 Gemini sentiment features** — each historical headline is scored by **Gemini 2.5 Flash-Lite** for sentiment, magnitude, and category. Results are cached in `gemini_sentiment_cache` to avoid re-scoring
- **4 market context features** from SPY and VIX snapshots
- **6 macro features** from FRED (yield curve, fed funds, credit spread, DXY, universe breadth)

Total: **54 candidate features** per snapshot row, plus target label and next-day return.

### 3. Feature Selection — Vertex AutoML

```bash
make submit-feature-ranking          # submit async
make check-feature-ranking           # poll status
make finalize-feature-ranking        # write contract
```

Vertex AI AutoML Tabular ranks features and writes the selected feature contract into `feature_selection_runs`. The contract must be reviewed and marked `APPROVED` before gold materialization.

**Fallback:** If AutoML is pending, a deterministic seeded-random contract selects 40 features (always retaining all 9 Gemini features).

### 4. Gold — Approved Feature Materialization

```bash
make prepare-training-data
```

Reads silver + the approved feature contract. Validates that every selected feature is servable by the shared online feature builder. Materializes only the approved subset into `gold_training_features`.

This is the explicit silver-to-gold ETL boundary. Training never rebuilds gold.

### 5. Model Training

```bash
make train
```

Reads the prepared gold table only:

- Filters label noise (`|next_day_return| < 0.003`)
- Chronological holdout (last 20% of dates)
- 5-fold date-based `TimeSeriesSplit` cross-validation
- Optuna hyperparameter tuning for both LightGBM and XGBoost
- Selects the best prediction strategy (LGB, XGB, or weighted blend)
- Learns the classification threshold from out-of-fold validation
- Writes artifacts to GCS and metadata to `model_metadata`
- Promotes model to `latest.json` manifest if CV accuracy > 0.52

### 6. Live Inference

The endpoint and batch job are the only runtime paths that call live APIs.

```bash
make deploy-endpoint
```

## Feature Store

The system uses a **shared feature registry** (`feature_engineering.py`) that defines all candidate features, their groups, defaults, and specifications. Both the offline and online paths build features from this single registry.

### Offline Store (Training)

- Pre-computed during silver ETL and materialized into gold
- Training reads `gold_training_features` — no live API calls
- Feature contract versioned by `feature_selection_run_id` and `candidate_feature_version`

### Online Store (Endpoint)

- Shared builder `build_live_feature_bundle()` computes the same features at request time
- Sources: Finnhub quote, Yahoo price history, Gemini headline scoring, FRED macro data
- `validate_online_feature_names()` ensures the model's feature contract only includes features the online builder can serve
- Every prediction writes its feature snapshot to `live_feature_snapshots` for audit and cache reuse

## Endpoint Caching

The endpoint manages a three-layer cache to serve multiple users requesting the same ticker while respecting API quotas:

### Layer 1 — In-Memory (per-instance)

| Cache | TTL | Purpose |
|-------|-----|---------|
| Snapshot | 300s | Avoid duplicate Finnhub/Yahoo calls |
| Sentiment | 1800s | Avoid duplicate Gemini scoring |
| Macro | 86400s | FRED data changes daily at most |

### Layer 2 — BigQuery (`live_feature_snapshots`)

Persisted snapshots are reused across instances and restarts:

| Condition | Action |
|-----------|--------|
| Snapshot age ≤ 1h (fresh) | Reuse cached snapshot |
| Snapshot age ≤ 6h AND daily budget exhausted | Serve stale with warning |
| Snapshot age > 6h AND budget exhausted | Reject request with error |
| Feature contract mismatch | Skip cache, fetch live |

### Layer 3 — Live API Refresh

Only triggers when no usable cache exists AND the ticker's daily live-fetch budget has remaining quota. Fresh snapshots are persisted to BigQuery for subsequent requests.

### Quota Protection

- **Daily limit**: 12 live API fetches per ticker per day (configurable)
- Each prediction response includes cache transparency: source mode, snapshot age, quota usage, and whether stale data was served
- The frontend surfaces this as `Live API`, `BQ cache`, or `BQ cache (stale)` indicators

## BigQuery Tables

| Layer | Table | Purpose |
|-------|-------|---------|
| Bronze | `raw_market_data` | Daily OHLCV per ticker |
| Bronze | `raw_news_data` | Headlines per ticker |
| Bronze | `raw_macro_data` | FRED economic time series |
| Silver | `training_feature_snapshots` | Historical candidate features (55 cols + target) |
| Silver | `gemini_sentiment_cache` | Deduplicated Gemini headline scores |
| Selection | `feature_selection_runs` | AutoML feature contracts |
| Gold | `gold_training_features` | Approved selected features for training |
| Online | `live_feature_snapshots` | Endpoint/batch audit rows + cache |
| Online | `macro_feature_cache` | Cached live FRED macro pulls |
| Metadata | `model_metadata` | Training runs, metrics, deployment status |
| Metadata | `training_tickers` | Active training universe |
| Batch | `batch_inference_requests` | Queued batch scoring requests |
| Batch | `batch_inference_predictions` | Batch scoring results |

Snapshot sources tracked in `live_feature_snapshots`:

- `endpoint_live_api` — fresh live fetch from endpoint
- `endpoint_bigquery_cache` — reused cached snapshot from endpoint
- `batch_live_api` / `batch_bigquery_cache` — same for batch job
- `historical_backfill` — silver ETL backfill rows

## Environment

Project root `.env` should include:

```bash
FINNHUB_KEY=your_finnhub_key
FRED_API_KEY=your_fred_key
FEATURE_RANK_NOTIFICATION_WEBHOOK=https://hooks.example.internal/market-sentiment  # optional
LIVE_SNAPSHOT_FRESH_TTL_SECONDS=3600
LIVE_SNAPSHOT_MAX_STALE_TTL_SECONDS=21600
LIVE_SNAPSHOT_DAILY_LIVE_FETCH_LIMIT=12
```

Frontend:

```bash
PREDICT_API_URL=https://...a.run.app/predict
```

Use `.env.example`, `frontend/.env.local.example`, and `infra/terraform/terraform.tfvars.example` as templates. Do not commit real credentials, `.env` files, `*.tfvars`, service-account JSON, or certificate/key material.

## Deployment

Infrastructure code lives under `infra/terraform/`. For a new GCP project, bootstrap once from [`infra/terraform/bootstrap`](infra/terraform/bootstrap/README.md).

```bash
make up                    # run BigQuery migrations
make deploy-pipeline       # build + deploy training/batch image
make deploy-function       # deploy ingestion Cloud Function
make deploy-endpoint       # build + deploy prediction service
```

## Workflow

Full bronze → silver → gold → train → deploy:

```bash
make ingest                        # 1. load raw data into bronze
make submit-feature-ranking        # 2. build silver + submit AutoML
make check-feature-ranking         # 3. poll AutoML status (optional)
make finalize-feature-ranking      # 4. finalize feature contract
# review and APPROVE the contract in feature_selection_runs
make prepare-training-data         # 5. materialize gold from silver
make train                         # 6. train ensemble on gold
make deploy-endpoint               # 7. deploy prediction service
make verify-endpoint               # 8. verify endpoint health
```

`make rollout` runs: migrate → deploy-pipeline → deploy-function → ingest → submit-feature-ranking → deploy-endpoint. Training stays manual so teams can review and approve the AutoML contract first.

## Testing

```bash
make test          # backend smoke tests
make test-all      # full backend + ML suite
make scan-secrets  # secret scan before pushing
```

## Frontend

```bash
cd frontend && npm install && npm run dev
```

The UI shows per prediction:

- Prediction label and confidence / `prob_up`
- Live headlines with per-headline Gemini sentiment (color-coded)
- Grouped feature snapshot with provider and fallback metadata
- Cache transparency: source mode, snapshot age, daily quota remaining

## Current Baseline

From the latest promoted `model_metadata` row (`2026-03-13 20:27:24 UTC`):

- Training window: `2020-01-02` to `2026-03-12` (7,775 samples, 50 features)
- 5-fold CV: accuracy `0.5277 ± 0.0279`, AUC `0.5036 ± 0.0185`
- Chronological holdout: accuracy `0.5209`, AUC `0.5336`
- Top features: `day_range_position`, `gap_pct`, `volume_vs_avg_10d`, `intraday_range_pct`

## Monorepo Layout

```text
├── backend/
│   ├── api/                 # FastAPI prediction service (online feature store + caching)
│   ├── ingestion/           # Cloud Function for bronze ingestion
│   ├── ml/
│   │   ├── training/        # Silver ETL, feature selection, gold ETL, model training
│   │   ├── inference/       # Batch inference job
│   │   ├── testing/         # ML contract and feature engineering tests
│   │   └── shared/          # Feature registry, live data, Gemini scoring, caching
│   ├── db/                  # BigQuery schema + dbmate migrations
│   └── tests/               # Backend service and ingestion tests
├── frontend/                # Next.js UI
├── infra/
│   └── terraform/           # GCP infrastructure
├── docs/                    # Runbooks and specs
└── Makefile
```

## Ops Runbooks

- [DevOps](docs/operations/devops-runbook.md)
- [MLOps](docs/operations/mlops-runbook.md)
- [Email Notifications](docs/operations/email-notifications.md)
