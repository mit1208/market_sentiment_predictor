# Accuracy Improvement Implementation Tracker

Spec: [`docs/superpowers/specs/2026-03-13-accuracy-improvement-design.md`](/Users/mitpatel/Desktop/code-exam/market_sentiment_predictor/docs/superpowers/specs/2026-03-13-accuracy-improvement-design.md)

## Candidate Contract

- Candidate feature version: `2026-03-14-fast-accuracy-pass-v1`
- Candidate feature count: `54`
- Historical source base: `raw_market_data` + `raw_news_data` + `raw_macro_data`
- Warehouse flow: `bronze raw_* -> silver training_feature_snapshots -> reviewed feature_selection_runs -> gold gold_training_features`
- Gemini model: `gemini-2.5-flash-lite`

## AutoML Selection

- Status: `ready for first cloud feature-selection run`
- Selected feature count: `TBD`
- Selected features: `TBD`
- AutoML run ID/version: `TBD`
- Review workflow: `Cloud Run feature-ranking submit job -> Vertex AutoML -> Cloud Run feature-ranking finalize job -> BigQuery feature_selection_runs -> stakeholder review -> mark APPROVED -> Cloud Run training job`

## Production Training

- Ensemble: `LightGBM + XGBoost`
- Label-noise threshold: `|next_day_return| < 0.003` dropped
- Optuna budget: `25` trials per model
- Promotion gate: `cv_accuracy_mean > 0.52`

## Deferred Feature Groups

- Fundamentals: deferred in this pass
- Options chain / flow: deferred in this pass

## Implementation Checklist

- [x] Replace TF-IDF/VADER contract with 54-feature candidate contract
- [x] Add Gemini sentiment scoring + cache module
- [x] Add macro/regime feature module
- [x] Rebuild training pipeline around AutoML-selected features
- [x] Switch serving and batch inference to the ensemble artifact contract
- [x] Add BigQuery migration for widened snapshot tables + Gemini cache
- [x] Add BigQuery-backed feature-selection review table + macro cache table
- [x] Add bronze raw macro table and gold training table
- [x] Split feature ranking into dedicated submit/finalize Cloud Run jobs
- [x] Gate training on an approved BigQuery feature contract
- [x] Remove post-ingestion API calls from feature ranking and training
- [ ] Run first AutoML selection and update selected-feature contract
- [ ] Run end-to-end training and record metrics here
