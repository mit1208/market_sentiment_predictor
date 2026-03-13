# System Architecture

```mermaid
graph TD
    subgraph "External Data"
        YF["yfinance (OHLCV + News)"]
    end

    subgraph "GCP - Ingestion & Orchestration"
        CS["Cloud Scheduler (Weekly, Sunday 00:00 UTC)"]
        CF["Cloud Function v2 (market-data-ingestion)"]
        BQ[("BigQuery")]
    end

    CS -->|Trigger| CF
    YF -->|Fetch OHLCV + Headlines| CF
    CF -->|MERGE Upsert| BQ
    CF -->|Trigger| TJ

    subgraph "Cloud Run Jobs"
        TJ["Training Job (market-sentiment-train)"]
        BJ["Batch Inference Job (market-sentiment-batch-infer)"]
    end

    TJ -->|Read raw_market_data + raw_news_data| BQ
    TJ -->|Write Artifacts| GCS[("GCS (model.joblib, tfidf.joblib, metadata.json)")]
    TJ -->|Promote if acc > 0.50| MF["latest.json (Model Manifest)"]
    MF --- GCS

    BJ -->|Read batch_inference_requests| BQ
    BJ -->|Load Model| MF
    BJ -->|Write batch_inference_predictions| BQ

    subgraph "Online Prediction"
        CRS["Cloud Run Service (market-sentiment-predict)"]
    end

    CRS -->|Load Model (auto-refresh 30s)| MF
    CRS -->|Fetch OHLCV History| BQ
    CRS -->|Fetch Live Data| YF

    subgraph "Frontend"
        FE["Vercel (Next.js)"]
    end

FE -->|User: Ticker Only| CRS
CRS -->|Prediction: Up/Down + Confidence| FE
```

## Current Baseline Model

Latest promoted baseline from BigQuery `model_metadata` (`2026-03-13 20:27:24 UTC`):

- Training window: `2020-01-02` to `2026-03-12`
- Training samples: `7,775`
- Feature count: `50`
- 5-fold CV: accuracy `0.5277 +/- 0.0279`, AUC `0.5036 +/- 0.0185`
- Chronological holdout: accuracy `0.5209`, AUC `0.5336`
- Promotion status: `PROMOTED`

The current promoted model is still dominated by price/volume snapshot features. Market context,
sentiment, and TF-IDF headline features all had zero importance in this baseline.
