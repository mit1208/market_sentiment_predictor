# Accuracy Improvement Design — Hedge Fund-Style Multi-Signal Model

**Date:** 2026-03-13
**Goal:** Maximize next-day stock price direction prediction accuracy by replacing zero-importance features with high-quality multi-source signals, using Gemini Flash for sentiment, adding options/flow/macro/temporal features, and letting Vertex AI AutoML select the final feature set for a LightGBM + XGBoost ensemble.

---

## Baseline (Current)

- Model version: `2026-03-13T20-27-24.653025+00-00`
- 5-fold CV: accuracy `0.5277 +/- 0.0279`, AUC `0.5036 +/- 0.0185`
- Chronological holdout: accuracy `0.5209`, AUC `0.5336`
- 50 features, but only 4 have non-zero importance (`day_range_position`, `gap_pct`, `volume_vs_avg_10d`, `intraday_range_pct`)
- All sentiment (VADER), TF-IDF (16 features), and market context features had **zero importance**
- Root cause: VADER + TF-IDF are too crude for financial text; many features are noise

## Target

- Push accuracy as high as realistically possible (estimated ceiling: 62-80% for daily direction with public data)
- AUC improvement from 0.50 → 0.65-0.80
- Every feature must earn its place via AutoML feature selection — no noise features
- No fallback defaults at inference — fail loudly if data sources are unavailable
- At training time, features without historical data (e.g., options) are excluded from the feature set entirely rather than filled with defaults

---

## Architecture Overview

### Philosophy

Instead of fixing 50 mediocre features, build ~70 candidate features from diverse high-quality data sources, then let Vertex AI AutoML Tabular select the best ~30-40 for the production ensemble. This is how quantitative funds build alpha: cast a wide net of signals, then prune ruthlessly.

### Integration Approach

**Flat feature vector** — Gemini sentiment scores become plain numeric columns alongside all other features in a single row per (ticker, date). LightGBM/XGBoost tree ensembles automatically discover non-linear cross-feature interactions (e.g., "bearish sentiment + high put/call ratio + near earnings" → strong Down signal). One model to train, evaluate, deploy.

One compound feature to assist discovery:
```
sentiment_market_alignment = sign(gemini_sentiment_avg) == sign(day_change_pct)
```
This binary feature tells the model whether news and price action agree — a signal that's hard for tree splits to discover from raw values alone.

### Pipeline

```
Phase 1: Feature Engineering (no model training)
  Finnhub + yfinance + FRED + Gemini Flash → ~70 raw features per (ticker, date)
  Store in BigQuery training_feature_snapshots

Phase 2: AutoML Feature Selection (one-time)
  Upload enriched dataset to Vertex AI AutoML Tabular
  AutoML identifies top ~30-40 features by importance
  Lock selected features into production feature contract

Phase 3: Production Model
  LightGBM + XGBoost ensemble on AutoML-selected features
  Optuna hyperparameter tuning
  Label noise filtering (drop |next_day_return| < 0.3%)
  TimeSeriesSplit CV + chronological holdout

Phase 4: Serving
  Same Cloud Run architecture
  Live feature computation from same sources
  Gemini sentiment cached 30min, options data cached 1hr, macro data cached 24hr
```

### Data Sources

| Source | Data | Cost | Status |
|--------|------|------|--------|
| **Finnhub** | Options flow, earnings calendar, analyst recommendations, insider transactions, company news | Free tier (60 calls/min) | Have API key |
| **yfinance** | Historical prices, options chains, company fundamentals, news | Free | Already using |
| **FRED** | Fed funds rate, yield curve, credit spreads, dollar index | Free | Need API key (free registration) |
| **Gemini Flash** | Financial headline sentiment scoring (via Vertex AI) | ~$0.01/1K headlines | Available via GCP |

### No Fallback Defaults (Inference)

If any data source is unavailable at inference time, the prediction request **fails with a clear error** rather than silently feeding neutral values.

### Training Data Policy

- Features with full historical coverage (technical, temporal, sentiment, macro, fundamentals): rows with insufficient history are **dropped** rather than filled
- Features without historical data (options chain features): **excluded from training entirely**. These features are only available for inference and for the one-time AutoML evaluation using a recent-period snapshot. AutoML evaluates them on recent data only; if they show signal, they are included in the production feature contract for inference going forward, but training still excludes them for the historical period

---

## Feature Groups

### Group 1: Technical / Price Features (~25 features)

Retained from current model with additions. Computed from `raw_market_data` in BigQuery.

**Kept from current model (25):**

| Feature | Description |
|---------|-------------|
| `current_price` | Current close price |
| `previous_close` | Previous day's close |
| `day_open` | Today's open |
| `day_high` | Today's high |
| `day_low` | Today's low |
| `day_change` | Close - previous close |
| `day_change_pct` | Close / previous close - 1 |
| `gap_pct` | Open / previous close - 1 |
| `intraday_range_pct` | (High - Low) / previous close |
| `close_vs_open_pct` | Close / open - 1 |
| `day_range_position` | (Close - Low) / (High - Low) |
| `volume` | Today's volume |
| `avg_volume_10d` | 10-day average volume |
| `avg_volume_3m` | 3-month average volume |
| `volume_vs_avg_10d` | Volume / 10-day avg |
| `volume_vs_avg_3m` | Volume / 3-month avg |
| `fifty_day_avg` | 50-day moving average |
| `two_hundred_day_avg` | 200-day moving average |
| `price_vs_50d_avg` | Close / 50d avg - 1 |
| `price_vs_200d_avg` | Close / 200d avg - 1 |
| `fifty_two_week_high` | 52-week high |
| `fifty_two_week_low` | 52-week low |
| `price_vs_52w_high` | Close / 52w high - 1 |
| `price_vs_52w_low` | Close / 52w low - 1 |
| `position_52w_range` | (Close - 52w low) / (52w high - 52w low) |

### Group 2: Temporal / Momentum Features (10 features)

Computed from existing `raw_market_data` + SPY history. No new API calls needed.

| Feature | Description |
|---------|-------------|
| `return_1d` | Yesterday's close-to-close return |
| `return_3d` | 3-day cumulative return |
| `return_5d` | 5-day cumulative return |
| `return_10d` | 10-day cumulative return |
| `return_20d` | 20-day cumulative return |
| `momentum_streak` | Consecutive up/down days (positive = up streak, negative = down) |
| `volatility_regime` | Rolling 5d stdev / rolling 20d stdev. >1.5 = expanding, <0.7 = contracting |
| `volume_trend_5d` | Slope of volume over last 5 days (normalized) |
| `price_acceleration` | return_5d - return_10d. Positive = accelerating upward |
| `relative_strength_spy` | Ticker's 5d return minus SPY's 5d return |

Rows with insufficient history for lag features (first ~20 trading days) are **dropped from training**.

### Group 3: Gemini Sentiment Features (8 features)

Replace all VADER + TF-IDF features (21 features → 8).

**New module:** `backend/ml/shared/gemini_sentiment.py`

**Gemini Flash prompt (per headline):**
```
Analyze this financial news headline for {ticker}.
Rate the likely impact on next-day stock price.

Headline: "{headline}"

Respond with JSON:
{
  "sentiment_score": <float -1.0 to 1.0>,
  "magnitude": <float 0.0 to 1.0, how strongly directional>,
  "category": <"earnings"|"macro"|"product"|"legal"|"analyst"|"insider"|"other">
}
```

**Aggregated features per (ticker, date):**

| Feature | Description |
|---------|-------------|
| `gemini_sentiment_avg` | Mean sentiment score across headlines (-1 to +1) |
| `gemini_sentiment_max` | Most bullish headline score |
| `gemini_sentiment_min` | Most bearish headline score |
| `gemini_magnitude_avg` | Mean conviction/strength of sentiment |
| `headline_count` | Number of headlines scored |
| `has_earnings_news` | Binary: any headline categorized as "earnings" |
| `has_analyst_news` | Binary: any headline categorized as "analyst" |
| `sentiment_dispersion` | Std dev of scores — measures disagreement/controversy |

**Training-time:**
- Read headlines from `raw_news_data` in BigQuery
- Batch-score all headlines via Gemini Flash (`google-genai` SDK via Vertex AI)
- ~40K headlines for 7,775 training samples — costs ~$0.40
- Cache results in new BQ table `gemini_sentiment_cache` to avoid re-scoring on retrains

**Inference-time:**
- Fetch 10 latest articles for the ticker via Finnhub + yfinance
- Score each headline with Gemini Flash — typically ~200ms for 10 headlines
- Cache per (ticker, headline_hash) for 30 minutes

**Compound feature:**
- `sentiment_market_alignment` = `sign(gemini_sentiment_avg) == sign(day_change_pct)` (binary: 1 if news and price agree)

### Group 4: Options / Flow Features (8 features)

**Source:** Finnhub API (free tier, already have API key)

| Feature | Finnhub Endpoint | Description |
|---------|-----------------|-------------|
| `put_call_ratio` | `/stock/option-chain` | Total put OI / total call OI. High = bearish positioning |
| `options_volume_ratio` | `/stock/option-chain` | Total options volume / stock volume. High = unusual activity |
| `iv_percentile` | `/stock/option-chain` | Current IV vs 30-day range (0-1) |
| `near_term_iv` | `/stock/option-chain` | IV of nearest expiry ATM options |
| `iv_skew` | `/stock/option-chain` | OTM put IV - OTM call IV. Positive = fear premium |
| `days_to_earnings` | `/calendar/earnings` | Trading days until next earnings report |
| `last_earnings_surprise` | `/stock/earnings` | Last quarter's EPS surprise % |
| `analyst_recommendation_trend` | `/stock/recommendation` | Net upgrades - downgrades in last 30 days |

**Training-time:**
- Historical options chains not available on Finnhub free tier — only current snapshots
- Earnings surprise and analyst recommendations **are** historical via Finnhub — full backfill available
- **Options chain features (`put_call_ratio`, `options_volume_ratio`, `iv_percentile`, `near_term_iv`, `iv_skew`) are excluded from historical training entirely.** They have no historical data and cannot be backfilled.
- These 5 options features are evaluated during the one-time AutoML run using a recent-period snapshot (~30 days of daily snapshots collected before AutoML). If AutoML shows they carry signal, they are added to the inference-time feature contract. The production LightGBM/XGBoost ensemble trains without them but receives them as additional inference-only features via a separate lightweight model or as supplementary signals.
- **Earnings and analyst features (`days_to_earnings`, `last_earnings_surprise`, `analyst_recommendation_trend`) have full history** and are included in both training and inference.

**Inference-time:**
- Fetch live options chain — single API call ~500ms
- Earnings calendar + analyst recommendations — cached 24hr
- Rate limit: 60 calls/min on Finnhub free tier; ~3 calls per prediction — comfortable margin

### Group 5: Fundamental / Company Features (8 features)

**Source:** yfinance (`yf.Ticker(t).info`) + Finnhub

| Feature | Source | Description |
|---------|--------|-------------|
| `market_cap` | yfinance | Raw market capitalization (log-scaled) |
| `pe_ratio` | yfinance | Price-to-earnings ratio |
| `forward_pe` | yfinance | Forward P/E ratio |
| `price_to_book` | yfinance | Price-to-book ratio |
| `debt_to_equity` | yfinance | Debt/equity ratio |
| `revenue_growth` | yfinance | Year-over-year revenue growth |
| `profit_margin` | yfinance | Net profit margin |
| `beta` | yfinance | Market sensitivity |

**Caching:** Company fundamentals cached 24hr at inference time (changes quarterly at most).

### Group 6: Macro / Regime Features (6 features)

**Source:** FRED API (free, requires API key registration)

| Feature | FRED Series | Description |
|---------|-------------|-------------|
| `yield_curve_spread` | `T10Y2Y` | 10Y - 2Y Treasury spread. Negative = inverted = recession signal |
| `fed_funds_rate` | `FEDFUNDS` | Current federal funds rate |
| `fed_funds_change_3m` | `FEDFUNDS` | Change in fed funds rate over last 3 months |
| `credit_spread` | `BAMLH0A0HYM2` | High-yield credit spread. Widening = risk-off |
| `dxy_change_5d` | `DTWEXBGS` | Dollar index 5-day change |
| `market_breadth` | Computed via yfinance | % of S&P 500 stocks above 50d MA |

**New env var:** `FRED_API_KEY`

**Training-time:** Fetch full FRED history from 2020 to present — single bulk download per series. Join to training data by date (forward-fill for weekends/holidays).

**Inference-time:** Fetch latest value per series — cached 24hr.

### Group 7: Market Context Features (4 features — kept from current)

| Feature | Description |
|---------|-------------|
| `spy_change_pct` | S&P 500 daily return |
| `spy_gap_pct` | SPY open gap |
| `vix_level` | CBOE Volatility Index close |
| `vix_change_pct` | VIX daily change |

---

## Total Raw Feature Count

| Group | Count |
|-------|-------|
| Technical / Price | 25 |
| Temporal / Momentum | 10 |
| Gemini Sentiment | 9 (8 + 1 compound) |
| Options / Flow | 8 |
| Fundamental / Company | 8 |
| Macro / Regime | 6 |
| Market Context | 4 |
| **Total raw candidates** | **70** |

AutoML selects the best ~30-40 for the production model.

---

## AutoML Feature Selection (One-Time)

### Process

1. Build the full 70-feature dataset for all training samples (Phase 1 output)
2. Upload to Vertex AI as a managed tabular dataset
3. Run Vertex AI AutoML Tabular training job with `next_day_direction` as target
4. Extract feature importance rankings from the trained AutoML model
5. Select features with importance > threshold (top ~30-40)
6. Lock the selected feature list into the production feature contract (`MODEL_FEATURE_NAMES`)
7. Discard the AutoML model itself — use its feature selection for the production LightGBM + XGBoost ensemble

### Why not just use AutoML as the production model?

- Vertex AI AutoML Tabular is expensive to serve (Vertex AI endpoint pricing)
- LightGBM + XGBoost ensemble on Cloud Run is much cheaper and we control the deployment
- AutoML's value here is its feature selection intelligence, not its model

---

## Production Model: LightGBM + XGBoost Ensemble

### Architecture

```
                      ┌──────────────────────────────────┐
                      │         Data Sources              │
                      ├──────────┬───────────┬────────────┤
                      │ Finnhub  │ yfinance  │   FRED     │
                      │ (quote,  │ (history, │ (yield     │
                      │  options,│  fundamen-│  curve,    │
                      │  news,   │  tals)    │  fed rate, │
                      │  earnings│           │  credit)   │
                      └────┬─────┴─────┬─────┴─────┬──────┘
                           │           │           │
                           ▼           ▼           ▼
┌──────────────┐    ┌─────────────────────────────────────┐
│ Gemini Flash │    │     Feature Engineering             │
│ (10 latest   │───▶│  Technical (25) + Temporal (10) +   │
│  headlines   │    │  Sentiment (9) + Options (8) +      │
│  → sentiment │    │  Fundamental (8) + Macro (6) +      │
│  scores)     │    │  Market Context (4) = 70 raw        │
└──────────────┘    └──────────────┬──────────────────────┘
                                   │
                          AutoML-selected (~30-40)
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
             ┌─────────────┐              ┌─────────────┐
             │  LightGBM   │              │   XGBoost   │
             │ (Optuna-    │              │ (Optuna-    │
             │  tuned)     │              │  tuned)     │
             └──────┬──────┘              └──────┬──────┘
                    │                             │
                    └──────── Average ─────────────┘
                              P(Up)
                                │
                         Up if P > 0.5
```

### Soft vote ensemble

- Average `lgb.predict_proba()[:, 1]` and `xgb.predict_proba()[:, 1]`
- Both models trained on the same data with the same CV splits
- Diverse learners (different boosting strategies) reduce variance

### Optuna Hyperparameter Tuning

- 50 trials per model, optimizing CV accuracy
- Key params: `num_leaves`, `learning_rate`, `min_child_samples`, `reg_alpha`, `reg_lambda`, `feature_fraction`, `bagging_fraction`, `max_depth`, `n_estimators`
- Uses the same TimeSeriesSplit CV as current training

### Label Noise Filtering

- During training, drop samples where `|next_day_return| < 0.3%`
- Small moves are noise from bid-ask spread and random fluctuations
- Reduces label noise, makes remaining labels more reliable
- Approximately ~15-20% of dataset may be near-zero
- Applied only to training data; inference always returns a prediction

### Artifacts

Saved to GCS (same structure as current):

| File | Description |
|------|-------------|
| `lgb_model.joblib` | Tuned LightGBM model |
| `xgb_model.joblib` | Tuned XGBoost model |
| `metadata.json` | Feature names, metrics, hyperparameters, feature importance |
| `performance_report.md` | Human-readable evaluation report |

**Removed:** `tfidf_vectorizer.joblib` — no longer needed.

### Model Promotion

- Same gate: ensemble CV accuracy must exceed threshold (raise from 0.50 to 0.52 given richer features)
- Same mechanism: update `latest.json` manifest in GCS
- Endpoint auto-refreshes from `latest.json` every 30s

---

## Component Changes

### New Files

| File | Purpose |
|------|---------|
| `backend/ml/shared/gemini_sentiment.py` | Gemini Flash headline scoring: `score_headlines_batch()` for training, `score_headlines_live()` for inference |
| `backend/ml/shared/options_features.py` | Finnhub options chain fetching + computation: put/call ratio, IV metrics, skew. Also earnings calendar and analyst recommendation fetchers. |
| `backend/ml/shared/macro_features.py` | FRED API fetching + computation: yield curve, fed funds, credit spread, DXY features |

### Modified Files

| File | Changes |
|------|---------|
| `backend/ml/shared/feature_engineering.py` | Add temporal/momentum features to `compute_market_features_from_history()`. Add new feature group constants. Remove TF-IDF logic (`fit_tfidf_safely`, `transform_tfidf_frame`, `TFIDF_FEATURE_NAMES`). Remove VADER (`compute_headline_features`, `get_sentiment_analyzer`). Update `MODEL_FEATURE_NAMES` and `EXPECTED_FEATURE_COUNT` to use AutoML-selected features. Update `build_live_feature_bundle()` to include all new feature groups. |
| `backend/ml/training/components/train.py` | Call Gemini batch scoring instead of VADER. Fetch fundamental/macro data. Add label noise filtering. Replace single LightGBM with LightGBM + XGBoost ensemble. Add Optuna tuning loop. Save both model artifacts. Remove TF-IDF vectorizer serialization. |
| `backend/ml/training/job_runner.py` | Update to work with new train module interface (two models instead of one). |
| `backend/api/predictor.py` | Replace VADER + TF-IDF with Gemini Flash call. Fetch options chain via Finnhub. Fetch fundamentals via yfinance. Fetch macro data via FRED. Load both LGB + XGB models. Average predictions. Add caching (sentiment 30min, options 1hr, fundamentals 24hr, macro 24hr). |
| `backend/ml/inference/batch_job.py` | Same changes as predictor — use new feature groups, ensemble prediction. |
| `backend/ml/shared/live_data.py` | Extend existing Finnhub/yfinance fetchers. Add options chain fetching. Keep as the central data-fetching module; new `options_features.py` and `macro_features.py` handle computation, while `live_data.py` handles raw API calls. |
| `backend/api/main.py` | No changes needed — still calls predictor, returns result. |

### Removed

| Item | Reason |
|------|--------|
| `vaderSentiment` dependency | Replaced by Gemini Flash |
| `TfidfVectorizer` fitting/serialization | Replaced by Gemini sentiment features |
| `tfidf_vectorizer.joblib` artifact | No longer produced |
| All `tfidf_*` feature names | Replaced |
| `sentiment_score` (VADER), `avg_news_age_hours`, `news_source_count` | Replaced by Gemini features |

### New Dependencies

| Package | Purpose |
|---------|---------|
| `google-genai` | Gemini Flash API calls via Vertex AI |
| `xgboost` | XGBoost model for ensemble |
| `optuna` | Hyperparameter tuning |
| `fredapi` | FRED economic data API |

### New Environment Variables

| Variable | Where | Description |
|----------|-------|-------------|
| `FRED_API_KEY` | ML Jobs, API Service | FRED API key for macro data |

Gemini Flash is accessed via Vertex AI using Application Default Credentials (no separate API key — uses existing `GCP_PROJECT_ID`).

---

## New BigQuery Table

| Table | Purpose |
|-------|---------|
| `gemini_sentiment_cache` | Cached Gemini sentiment scores per (headline_hash, ticker). Prevents re-scoring identical headlines across retrains. |

Schema:
```
headline_hash     STRING    (SHA-256 of headline text)
ticker            STRING
headline          STRING
sentiment_score   FLOAT
magnitude         FLOAT
category          STRING
scored_at         TIMESTAMP
```

---

## Inference Latency

| Step | Latency | Caching |
|------|---------|---------|
| Finnhub quote + options chain | ~800ms | None (must be live) |
| yfinance fundamentals | ~300ms | 24hr |
| FRED macro data | ~200ms | 24hr |
| Gemini Flash (10 headlines) | ~400ms | 30min per (ticker, headline_hash) |
| Feature computation + ensemble predict | ~50ms | None |
| **Total (cold)** | **~1.8s** | |
| **Total (warm cache)** | **~900ms** | |

Acceptable for advisor-facing UI. Cached paths hit on repeated predictions for same ticker within cache windows.

---

## Deployment Sequence

1. Register FRED API key (free), add to Cloud Run env vars
2. Deploy new pipeline image with all feature changes
3. Run feature-building job only — builds enriched 70-feature snapshots in BigQuery, scores headlines with Gemini (no model training yet)
4. Run one-time Vertex AI AutoML Tabular on the enriched dataset — extract feature importance rankings
5. Lock selected features (~30-40) into `MODEL_FEATURE_NAMES` in code
6. Run production training job — trains LightGBM + XGBoost ensemble on selected features only
7. Verify `latest.json` updated with new model (promotion gate: CV accuracy > 0.52)
8. Deploy new endpoint image
9. Frontend — no changes needed (still receives prediction + confidence + feature snapshot)

---

## What Stays the Same

- BigQuery raw tables (`raw_market_data`, `raw_news_data`)
- GCS artifact structure and `latest.json` manifest mechanism
- Cloud Run Jobs for training and batch inference
- Cloud Run Service for online prediction (FastAPI)
- Terraform-managed infrastructure
- Frontend deployment on Vercel
- Docker images built from project root
- Next-day Up/Down binary classification target

---

## Success Criteria

| Metric | Baseline | Target |
|--------|----------|--------|
| CV Accuracy | 0.5277 | > 0.55 |
| CV AUC | 0.5036 | > 0.58 |
| Holdout Accuracy | 0.5209 | > 0.54 |
| Holdout AUC | 0.5336 | > 0.57 |
| Non-zero importance features | 4 / 50 | > 20 / ~35 |
| Inference latency (p95) | ~1.5s | < 2.5s |

Realistic ceiling for daily direction prediction with public data is ~57-62% accuracy. Each percentage point above 55% represents a meaningful edge.

---

## Implementation Notes

- **Gemini structured output:** Use `response_mime_type="application/json"` with a response schema to guarantee valid JSON from Gemini Flash. Do not rely on prompt-only JSON formatting.
- **Label noise threshold:** The 0.3% threshold should be a named constant (`LABEL_NOISE_THRESHOLD`) to allow easy tuning.
- **Market breadth feature:** Computing % of S&P 500 stocks above 50d MA requires fetching price data for ~500 tickers. Use a pre-computed daily snapshot cached in BigQuery rather than fetching live for each prediction.
- **Module boundaries:** `live_data.py` handles raw API calls (Finnhub, yfinance, FRED). `options_features.py` and `macro_features.py` handle computation/transformation. `gemini_sentiment.py` handles both Gemini API calls and aggregation. `feature_engineering.py` orchestrates all groups into the final feature vector.
- **CLAUDE.md update:** After implementation, update CLAUDE.md with new env vars (`FRED_API_KEY`), updated architecture description, new feature counts, and updated baseline metrics.
