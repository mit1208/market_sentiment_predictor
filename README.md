# Market Sentiment Predictor

Market direction model with a live inference contract:

1. Historical market/news data lands in BigQuery.
2. The training job backfills a normalized `training_feature_snapshots` table in BigQuery.
3. Training reads those stored snapshot rows, fits LightGBM, evaluates with time-based CV plus a chronological holdout, and writes artifacts to GCS.
4. The endpoint and batch job fetch live features from `Finnhub + Yahoo`, score with the latest promoted model, and store the exact live feature snapshot they used back into BigQuery.
5. The frontend shows the prediction and all live features used for inference.

## Live Feature Contract

The model now uses a live snapshot contract instead of pulling stale OHLCV rows at inference time.

- Quote and intraday fields: Finnhub primary, Yahoo history fallback
- Rolling price/volume context: Yahoo live history
- Market context: SPY and VIX live snapshots
- News: Finnhub company news plus Yahoo news enrichment
- Text features: TF-IDF on the live headlines used for inference

Feature groups:

- 25 market snapshot features
- 4 market context features
- 5 sentiment/news metadata features
- 16 TF-IDF features

Total features: `50`

## BigQuery Tables

Core tables:

- `raw_market_data`
- `raw_news_data`
- `training_tickers`
- `training_feature_snapshots`
- `live_feature_snapshots`
- `model_metadata`
- `batch_inference_requests`
- `batch_inference_predictions`

Snapshot storage is split by purpose:

- `training_tickers` is the BigQuery source of truth for the active training universe
- `training_feature_snapshots` stores historical backfill rows used by training
- `live_feature_snapshots` stores endpoint and batch inference audit rows
- historical rows are written with `snapshot_source = historical_backfill`
- online endpoint rows are written with `snapshot_source = endpoint_live_api`
- batch rows are written with `snapshot_source = batch_live_api`
- training reads labeled rows from `training_feature_snapshots`

## Environment

Project root `.env` should include:

```bash
FINNHUB_KEY=your_finnhub_key
```

Frontend local config should include:

```bash
PREDICT_API_URL=https://...a.run.app/predict
```

Use `.env.example`, `frontend/.env.local.example`, and
`infra/terraform/terraform.tfvars.example` as templates. Do not commit real credentials, `.env`
files, `*.tfvars`, service-account JSON, or certificate/key material.

## Deployment

Use the `Makefile` from the project root.

Infrastructure code now lives under `infra/terraform`, while runtime code is split into
`frontend/` and `backend/`.

For a brand new GCP project, bootstrap it once from [`infra/terraform/bootstrap`](/Users/mitpatel/Desktop/code-exam/market_sentiment_predictor/infra/terraform/bootstrap/README.md). After the main Terraform stack is applied, the `Makefile` automatically reads the project ID, region, Artifact Registry repo, and model bucket from Terraform outputs when available.

Initial schema update:

```bash
make up
```

Build and deploy the training/batch image:

```bash
make deploy-pipeline
```

Run a training job:

```bash
make train
```

Build and deploy the prediction endpoint:

```bash
make deploy-endpoint
```

Verify the endpoint:

```bash
make verify-endpoint
```

Send one live prediction request:

```bash
PREDICT_API_URL=https://...a.run.app/predict make test-live-predict TICKER=AAPL
```

Full rollout:

```bash
make rollout
```

The deployment targets read `FINNHUB_KEY` from `.env` and push it into the Cloud Run job/service environment.

## Local Backend

Run the backend smoke tests from the project root:

```bash
make test
```

Run a tracked-file secret scan before pushing a public branch:

```bash
make scan-secrets
```

Run the full backend plus ML suite in an environment with the ML dependencies installed:

```bash
make test-all
```

## Local Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI is ticker-only now. Each result returns:

- prediction label
- confidence / `prob_up`
- live headlines used
- grouped feature snapshot with provider and fallback metadata

## Model Evaluation

Training writes:

- `model.joblib`
- `tfidf_vectorizer.joblib`
- `metadata.json`
- `performance_report.md`

Evaluation flow:

- historical snapshots are rebuilt into BigQuery
- date-based `TimeSeriesSplit` runs on the pre-holdout window
- a final chronological holdout is evaluated
- only models above the promotion threshold update `latest.json`

## Current Baseline

Recorded from the latest promoted `model_metadata` row on `2026-03-13 20:27:24 UTC`:

- model version: `2026-03-13T20-27-24.653025+00-00`
- training window: `2020-01-02` to `2026-03-12`
- training samples: `7,775`
- feature count: `50`
- 5-fold pre-holdout CV: accuracy `0.5277 +/- 0.0279`, AUC `0.5036 +/- 0.0185`
- chronological holdout (`2024-12-11` to `2026-03-11`): accuracy `0.5209`, AUC `0.5336`
- deployment status: `PROMOTED` because `cv_accuracy_mean` cleared the `0.50` gate

Top baseline drivers are still price/volume features such as `day_range_position`, `gap_pct`,
`volume_vs_avg_10d`, and `intraday_range_pct`. In this promoted baseline, market context,
sentiment, and TF-IDF headline features all carried zero importance. The full recorded report is
in `docs/superpowers/performance-report.md`.

## Monorepo Layout

```text
├── backend/
│   ├── api/                 # FastAPI prediction service + container
│   ├── ingestion/           # Cloud Function source for raw market/news ingestion
│   ├── ml/
│   │   ├── training/        # model training orchestration + feature snapshot backfill
│   │   ├── inference/       # online predictor + batch inference job
│   │   ├── testing/         # ML contract and feature engineering tests
│   │   └── shared/          # shared feature contract + live data helpers
│   ├── db/                  # schema dump + dbmate migrations
│   └── tests/               # backend service and ingestion tests
├── frontend/                # Next.js UI
├── infra/
│   └── terraform/           # infrastructure definitions
├── docs/                    # architecture notes, specs, reports
└── Makefile
```
