# Generalized Market Sentiment Model — Design Spec

**Date:** 2026-03-13
**Goal:** Make the model work well for any ticker by broadening training data, adding generalization features, and auto-fetching news at inference.

---

## Summary

Expand training from a handful of tickers to 50 diverse S&P 500 tickers across all 11 GICS sectors. Replace NewsAPI with yfinance news (free, no API key). Add company profile and market context features. Add VADER sentiment scoring. Simplify the frontend to ticker-only input (no manual headline entry). Keep exactly 50 features for transparency.

## Recorded Baseline Results

The currently promoted baseline in BigQuery `model_metadata` was trained on `2026-03-13 20:27:24 UTC`.

- Model version: `2026-03-13T20-27-24.653025+00-00`
- Training window: `2020-01-02` to `2026-03-12`
- Training samples: `7,775`
- Feature count: `50`
- Date-based 5-fold CV on the pre-holdout window: accuracy `0.5277 +/- 0.0279`, AUC `0.5036 +/- 0.0185`
- Chronological holdout (`2024-12-11` to `2026-03-11`): accuracy `0.5209`, AUC `0.5336`
- Deployment status: `PROMOTED` because `cv_accuracy_mean` cleared the `0.50` gate

Top baseline drivers are `day_range_position`, `gap_pct`, `volume_vs_avg_10d`, `intraday_range_pct`,
and `volume_vs_avg_3m`. All market-context features, sentiment features, and TF-IDF headline
features had zero importance in this promoted baseline. That result is the main motivation for
continuing feature/generalization work rather than treating the current feature mix as validated.

---

## Current Architecture (actual)

```
┌─────────────────────────────────────────────────────────────────────┐
│  GCP (project: llm-testing-489113, region: us-central1)             │
│                                                                     │
│  Cloud Scheduler (Sunday 00:00 UTC)                                 │
│    └─► Cloud Function v2 (market-data-ingestion)                   │
│          ├─ yfinance + NewsAPI → BigQuery (MERGE upsert)            │
│          └─ Trigger Cloud Run Training Job                          │
│                                                                     │
│  Cloud Run Job — Training (market-sentiment-train)                  │
│    ├─ Read raw_market_data + raw_news_data from BQ                  │
│    ├─ Build gold table (technical features + TF-IDF + target)       │
│    ├─ TimeSeriesSplit CV (5 folds) → LightGBM                      │
│    └─ IF acc > 0.50: upload artifacts + promote latest.json         │
│                                                                     │
│  Cloud Run Job — Batch Inference (market-sentiment-batch-infer)     │
│    ├─ Read pending requests from batch_inference_requests table     │
│    ├─ Load model from latest.json manifest                          │
│    ├─ Score all requests                                            │
│    └─ Write results to batch_inference_predictions table            │
│                                                                     │
│  Cloud Run Service — Online Prediction (market-sentiment-predict)   │
│    POST /predict  GET /health                                       │
│    ├─ Load model from latest.json (auto-refresh every 30s)          │
│    ├─ Fetch OHLCV from BQ (yfinance fallback)                       │
│    ├─ Compute technical features + TF-IDF                           │
│    └─ Return: prediction (Up/Down), confidence, prob_up             │
└─────────────────────────────────────────────────────────────────────┘

Vercel Frontend (Next.js)
  User enters tickers + headlines → POST to Cloud Run Service /predict
```

### Key Infrastructure

| Resource | Type | Purpose |
|----------|------|---------|
| `market_data_ingestion/` | Cloud Function v2 | Ingest data + trigger training job |
| `pipeline/job_runner.py` | Cloud Run Job | Training orchestration |
| `pipeline/batch_inference_job.py` | Cloud Run Job | Batch scoring |
| `endpoint/` | Cloud Run Service | Online prediction API (FastAPI) |
| `terraform/` | Terraform | All GCP resources |
| `frontend/` | Vercel (Next.js) | Advisor UI |

### Model Promotion

- Artifacts saved to: `gs://{bucket}/models/market_sentiment_lgbm/{timestamp}/`
- Live model pointer: `gs://{bucket}/models/market_sentiment_lgbm/latest.json`
- Promotion gate: CV accuracy > 0.50
- Online endpoint auto-refreshes from `latest.json` every 30 seconds

### BigQuery Tables

| Table | Purpose |
|-------|---------|
| `raw_market_data` | Daily OHLCV, partitioned by date, clustered by ticker |
| `raw_news_data` | Headlines, partitioned by date, clustered by ticker |
| `model_metadata` | Pipeline run results |
| `batch_inference_requests` | Incoming batch prediction requests |
| `batch_inference_predictions` | Completed batch predictions |

---

## Ticker Selection (50 tickers)

Scraped once per pipeline run from Wikipedia's S&P 500 list using `pandas.read_html`. The scraper selects ~4-5 tickers per GICS sector. The selected list is stored in the model's `metadata.json` for reproducibility. If Wikipedia scraping fails, fall back to a hardcoded default list committed in the repo.

| Sector | ~Count | Examples |
|--------|--------|----------|
| Technology | 5 | AAPL, MSFT, NVDA, GOOGL, META |
| Healthcare | 5 | JNJ, UNH, PFE, ABT, MRK |
| Financials | 5 | JPM, BAC, GS, V, MA |
| Consumer Discretionary | 4 | AMZN, TSLA, HD, NKE |
| Consumer Staples | 4 | PG, KO, PEP, WMT |
| Energy | 4 | XOM, CVX, COP, SLB |
| Industrials | 4 | CAT, HON, UPS, BA |
| Materials | 4 | LIN, APD, SHW, FCX |
| Utilities | 4 | NEE, DUK, SO, D |
| Real Estate | 3 | PLD, AMT, EQIX |
| Communication | 4 | DIS, NFLX, CMCSA, T |

If a ticker fails data fetch, it is skipped and the next in that sector is used.

---

## Feature Set (50 features total)

The feature count is enforced with an assertion at both training and inference time. If TF-IDF produces fewer than 19 columns, zero-padded columns are added to reach 19. If company profile features fail to fetch, defaults are used (see Fallbacks section).

### Technical Features (20)

| # | Feature | Description |
|---|---------|-------------|
| 1 | return_1d | 1-day price return |
| 2 | return_5d | 5-day price return |
| 3 | return_10d | 10-day price return |
| 4 | return_20d | 20-day price return |
| 5 | volatility_5d | 5-day rolling std of daily returns |
| 6 | volatility_20d | 20-day rolling std of daily returns |
| 7 | rsi_14 | Relative Strength Index (14-day) |
| 8 | macd_hist | MACD histogram (merged from line/signal/hist) |
| 9 | bb_position | Bollinger Band position (0-1) |
| 10 | stochastic_k | Stochastic %K oscillator |
| 11 | williams_r | Williams %R |
| 12 | atr_14 | Average True Range (14-day) |
| 13 | vwap_ratio | Price / Volume-Weighted Average Price |
| 14 | volume_ratio_5d | Today's volume / 5-day avg volume |
| 15 | volume_spike_20d | Today's volume / 20-day avg volume |
| 16 | intraday_range | (High - Low) / Close |
| 17 | gap | (Open - prev Close) / prev Close |
| 18 | price_vs_sma10 | Close / SMA(10) - 1 |
| 19 | obv_slope_5d | 5-day slope of On-Balance Volume |
| 20 | price_vs_sma20 | Close / SMA(20) - 1 |

Current code produces 17 technical features. Changes: drop return_3d, volatility_10d, macd_line, macd_signal, volume_ratio_10d, price_vs_sma5 (6 dropped). Add return_10d, return_20d, volatility_20d, stochastic_k, williams_r, atr_14, vwap_ratio, volume_spike_20d, price_vs_sma20 (9 added). Net: 17 - 6 + 9 = 20.

### Sentiment Features (22)

| # | Feature | Description |
|---|---------|-------------|
| 21-39 | tfidf_0 to tfidf_18 | TF-IDF features (`max_features=19`). Column names use ordinal indices (tfidf_0..tfidf_18) not vocabulary words, for stable alignment between train and serve. If the corpus yields fewer than 19 terms, remaining columns are zero-padded. |
| 40 | sentiment_score | VADER compound score (-1 to +1), averaged across headlines. Default: 0.0 when no headlines exist. |
| 41 | headline_count | Number of recent headlines for that ticker+date |
| 42 | has_news | Binary: 1 if any headlines exist, 0 otherwise |

### Company Profile Features (4)

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 43 | market_cap_bucket | yf.Ticker.info["marketCap"] | Ordinal 1-5: micro (<300M), small (<2B), mid (<10B), large (<200B), mega (>=200B) |
| 44 | sector_code | yf.Ticker.info["sector"] | Ordinal 1-11. Mapping: Communication Services=1, Consumer Discretionary=2, Consumer Staples=3, Energy=4, Financials=5, Healthcare=6, Industrials=7, Information Technology=8, Materials=9, Real Estate=10, Utilities=11. Unknown=0. |
| 45 | beta | yf.Ticker.info["beta"] | Market sensitivity |
| 46 | avg_volume_bucket | yf.Ticker.info["averageVolume"] | Ordinal 1-5 based on volume quintiles |

### Market Context Features (4)

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 47 | spy_return_1d | yfinance SPY | S&P 500 daily return |
| 48 | vix_level | yfinance ^VIX | CBOE Volatility Index close |
| 49 | day_of_week | date | 1-5 (Mon-Fri) |
| 50 | month | date | 1-12 |

For SPY/VIX: use the most recent available trading day's data (handles weekends, holidays, pre-market).

---

## Fallback Defaults

When external data is unavailable (yfinance rate-limited, ticker not found, etc.):

| Feature | Default | Rationale |
|---------|---------|-----------|
| sentiment_score | 0.0 | Neutral sentiment |
| headline_count | 0 | No news |
| has_news | 0 | No news |
| market_cap_bucket | 3 (mid) | Neutral middle bucket |
| sector_code | 0 (unknown) | Distinct from all valid sectors |
| beta | 1.0 | Market-neutral |
| avg_volume_bucket | 3 (mid) | Neutral middle bucket |
| spy_return_1d | 0.0 | Flat market |
| vix_level | 20.0 | Long-term VIX average |

---

## News Source Change

**Before:** NewsAPI (requires paid API key for volume)
**After:** yfinance (`yf.Ticker(t).news`) — free, no API key, covers all tickers

yfinance news returns a list of dicts with a `title` field. This is mapped to `headline` when stored in BQ's `raw_news_data` table (schema unchanged).

### Training (Ingestion Function)
- For each of the 50 tickers, call `yf.Ticker(t).news` to get recent headlines
- Map `title` → `headline` before MERGE upsert into BigQuery `raw_news_data` table
- Add 1-second delay between tickers to avoid yfinance rate limiting
- Compute VADER sentiment score at training time (not stored in BQ)

### Inference (Cloud Run Service)
- Auto-fetch latest headlines via `yf.Ticker(t).news`
- User no longer needs to paste headlines
- Compute VADER sentiment and TF-IDF at prediction time
- yfinance `.news` returns ~8 most recent items (no date filtering needed — these represent current sentiment)

### Train/Serve Alignment
During training, headlines are aggregated per (ticker, date) from BQ. At inference, `yf.Ticker(t).news` returns the latest ~8 headlines which represent "today's" news. Both paths feed into the same TF-IDF vectorizer and VADER scorer, so the feature semantics align. The time window difference (training: one day's headlines; inference: latest ~8 items) is acceptable since TF-IDF and VADER both operate on bag-of-words — the order and exact count don't materially affect the features.

---

## Component Changes

### Ingestion Function (`market_data_ingestion/`)
- Replace NewsAPI calls with `yf.Ticker(t).news`, mapping `title` → `headline`
- Scrape S&P 500 list from Wikipedia using `pandas.read_html`, sample 50 tickers across sectors
- Add fallback to hardcoded ticker list if Wikipedia scraping fails
- Update ticker list from current hardcoded set to the 50 selected tickers
- Add 1s delay between yfinance news calls to avoid rate limiting
- Remove `NEWSAPI_KEY` environment variable dependency
- Add `vaderSentiment` dependency

### Training Job (`pipeline/components/train.py` + `pipeline/job_runner.py`)
- Replace 17 current technical features with 20 (drop 6, add 9 — see Technical Features section)
- Reduce TF-IDF max_features from 100 to 19, rename columns to ordinal indices (tfidf_0..tfidf_18)
- Zero-pad if fewer than 19 TF-IDF columns produced
- Add VADER sentiment_score (default 0.0 when no headlines), headline_count, has_news
- Add company profile features (market_cap_bucket, sector_code, beta, avg_volume_bucket) fetched via yfinance
- Add market context features (spy_return_1d, vix_level, day_of_week, month)
- Fetch SPY and VIX data from yfinance during training
- Assert final feature count == 50
- Model promotion gate remains: accuracy > 0.50 to update `latest.json`

### Batch Inference Job (`pipeline/batch_inference_job.py`)
- Update feature engineering to match new 50-feature set
- Update model loading to expect new feature names
- Add VADER + company profile + market context feature computation
- Apply same fallback defaults when data unavailable

### Online Prediction Service (`endpoint/predictor.py`)
- Update `compute_technical_features()` to match new 20 technical features
- Auto-fetch headlines via `yf.Ticker(t).news` (remove `headlines` parameter from `predict()`)
- Add VADER sentiment computation
- Fetch company profile from `yf.Ticker(t).info` for profile features, with fallback defaults
- Fetch SPY/VIX for market context features (use last trading day's data)
- Cache SPY/VIX and company profile data for 1 hour to reduce yfinance calls and latency
- Assert feature count == 50 before prediction
- Model auto-refresh from `latest.json` every 30s (unchanged)

### Online Prediction API (`endpoint/main.py`)
- Update request schema: remove `headlines` field from instances, keep only `ticker`
- Update call to `predictor.predict(ticker)` — no headlines argument
- Old requests with `headlines` field are accepted but the field is ignored (backward compatible)

### Frontend (`frontend/src/app/page.tsx`)
- Remove headline input section entirely
- User enters ticker(s) only → predict
- Simplified UI: ticker input + add button + predict button + results
- Update request body to omit `headlines`

### Terraform (`terraform/main.tf`)
- Remove `NEWSAPI_KEY` from Cloud Function environment variables
- No other Terraform changes needed — Cloud Run Jobs, Service, BigQuery tables, GCS all stay the same

### Dependencies
- Add `vaderSentiment` to:
  - `pipeline/requirements.txt`
  - `endpoint/requirements.txt`
  - `requirements.txt` (root)
- yfinance already a dependency (no change)
- `pandas.read_html` uses `lxml` (add to ingestion function deps if not present)
- Remove `newsapi` dependency from ingestion function

### CLAUDE.md
- Update architecture diagram to match actual architecture (Cloud Run Jobs/Service, not Vertex AI Pipelines/Endpoint)
- Remove `NEWSAPI_KEY` from environment variables
- Update Cloud Function description: now in `market_data_ingestion/`, uses yfinance for news
- Update frontend description: "tickers + headlines" → "tickers only"
- Update feature count: 50 features (20 technical + 22 sentiment + 4 company + 4 market)
- Document model promotion strategy (latest.json manifest, accuracy > 0.50 gate)
- Add batch inference job documentation

---

## Inference Latency

At prediction time the Cloud Run Service makes several yfinance calls. Mitigation:
- **Cache company profile** (`yf.Ticker(t).info`): 1-hour TTL in-memory cache. Profile data rarely changes.
- **Cache SPY/VIX**: 1-hour TTL. Same values for all predictions within the hour.
- **yfinance news**: Not cached (needs to be fresh). Single call, typically <500ms.
- **Market history**: Already handled (BQ with yfinance fallback, as today).

Expected per-prediction latency: ~1-2s (acceptable for advisor-facing UI).

---

## Deployment

Frontend and Cloud Run Service can be deployed independently:
- Deploy training job first (new pipeline image) → triggers retrain with new features
- Deploy Cloud Run Service (new endpoint image) after model with 50 features is promoted
- Frontend can follow — the old frontend sending headlines still works (endpoint ignores extra fields)
- Old model artifacts (116 features) are NOT compatible with new endpoint code (50 features)
- Deployment sequence: pipeline image → trigger train → verify `latest.json` updated → endpoint image → frontend

---

## What Stays the Same

- LightGBM classifier with TimeSeriesSplit CV
- Model artifacts: model.joblib, tfidf_vectorizer.joblib, metadata.json on GCS
- Cloud Run Jobs for training and batch inference
- Cloud Run Service for online prediction (FastAPI)
- Model promotion via `latest.json` manifest (accuracy > 0.50 gate)
- Terraform-managed infrastructure
- BigQuery table schemas (raw_market_data, raw_news_data, model_metadata, batch tables)
- Frontend deployment on Vercel
- Docker images built from project root

---

## Target: Next-day price direction

Unchanged — binary classification: Up (next close > today's close) vs Down.
