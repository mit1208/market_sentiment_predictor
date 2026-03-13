# Pipeline Restructure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the project to use Vertex AI Pipelines (KFP) for train → eval → deploy, replace Gemini with TF-IDF text features, and clean up the folder layout.

**Architecture:** Cloud Function ingests data to BigQuery then triggers a Vertex AI Pipeline. The pipeline has 3 KFP components: train (BQ → technical features + TF-IDF → LightGBM), evaluate (accuracy check), and deploy (conditional deploy to Vertex AI endpoint if accuracy > 50%). The endpoint loads a TF-IDF vectorizer instead of calling Gemini.

**Tech Stack:** KFP v2, LightGBM, scikit-learn (TF-IDF), google-cloud-aiplatform, BigQuery, FastAPI, Terraform, Next.js

**Spec:** `docs/superpowers/specs/2026-03-12-pipeline-restructure-design.md`

---

## Chunk 1: Delete Old Directories & Create Pipeline Structure

### Task 1: Delete obsolete directories and files

**Files:**
- Delete: `ingestion/` (entire directory)
- Delete: `warehouse/` (entire directory)
- Delete: `data/` (entire directory)
- Delete: `ml/` (entire directory)
- Delete: `DATA_PIPELINE.md`
- Delete: `test_request.json`
- Delete: `architecture.svg`
- Delete: `ml/sentiment_analysis.py`

- [ ] **Step 1: Delete directories**

```bash
cd /Users/mitpatel/Desktop/code-exam/market_sentiment_predictor
rm -rf ingestion/ warehouse/ data/ ml/
rm -f DATA_PIPELINE.md test_request.json architecture.svg
```

- [ ] **Step 2: Verify deletion**

```bash
ls -la
```

Expected: Only `cloud_function/`, `endpoint/`, `frontend/`, `terraform/`, `docs/`, `CLAUDE.md`, `README.md`, `requirements.txt` remain.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove obsolete directories (ingestion, warehouse, data, ml)"
```

---

### Task 2: Create pipeline directory structure

**Files:**
- Create: `pipeline/components/__init__.py` (empty)
- Create: `pipeline/__init__.py` (empty)
- Create: `pipeline/requirements.txt`
- Create: `pipeline/Dockerfile`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p pipeline/components
touch pipeline/__init__.py pipeline/components/__init__.py
```

- [ ] **Step 2: Create `pipeline/requirements.txt`**

```
kfp>=2.7.0
google-cloud-aiplatform>=1.60.0
google-cloud-bigquery[pandas]>=3.0.0
google-cloud-storage>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
joblib>=1.3.0
db-dtypes>=1.0.0
pyarrow>=14.0.0
```

- [ ] **Step 3: Create `pipeline/Dockerfile`**

Build from project root: `docker build -f pipeline/Dockerfile .`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# LightGBM runtime dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline/ ./pipeline/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
```

- [ ] **Step 4: Commit**

```bash
git add pipeline/
git commit -m "chore: scaffold pipeline directory with Dockerfile and requirements"
```

---

## Chunk 2: Pipeline KFP Components

### Task 3: Create the train component

**Files:**
- Create: `pipeline/components/train.py`

This component:
1. Reads `raw_market_data` + `raw_news_data` from BigQuery
2. Computes technical features (RSI, MACD, Bollinger, etc.) — inlined from old `feature_engineering.py`
3. Applies TF-IDF to concatenated daily headlines per (ticker, date)
4. Creates target (next day up/down)
5. Splits data using TimeSeriesSplit — fits TF-IDF only on training split
6. Trains LightGBM with 5-fold TimeSeriesSplit CV
7. Saves `model.joblib`, `tfidf_vectorizer.joblib`, `metadata.json` to GCS
8. Outputs: GCS artifact URI, accuracy, AUC, test data path

- [ ] **Step 1: Write `pipeline/components/train.py`**

```python
"""
KFP Train Component — reads BQ, computes features, trains LightGBM.
Outputs model artifacts to GCS.
"""

import json
import logging
import os
import tempfile
from datetime import datetime

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "llm-testing-489113")
REGION = os.environ.get("GCP_REGION", "us-central1")
DATASET_ID = os.environ.get("BQ_DATASET", "market_sentiment_predictor")
GCS_BUCKET = os.environ.get("GCS_BUCKET", f"{PROJECT_ID}-model-artifacts")
MODEL_NAME = os.environ.get("MODEL_NAME", "market_sentiment_lgbm")

TFIDF_MAX_FEATURES = 100

BQ_MARKET_TABLE = f"{PROJECT_ID}.{DATASET_ID}.raw_market_data"
BQ_NEWS_TABLE = f"{PROJECT_ID}.{DATASET_ID}.raw_news_data"


# ---------------------------------------------------------------------------
# Technical feature functions (inlined from old feature_engineering.py)
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram}, index=close.index)


def compute_bollinger_position(close: pd.Series, window: int = 20) -> pd.Series:
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (close - lower) / (upper - lower)


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    volume = df["Volume"]

    features = pd.DataFrame(index=df.index)
    features["return_1d"] = close.pct_change(1)
    features["return_3d"] = close.pct_change(3)
    features["return_5d"] = close.pct_change(5)

    daily_ret = close.pct_change()
    features["volatility_5d"] = daily_ret.rolling(5).std()
    features["volatility_10d"] = daily_ret.rolling(10).std()
    features["rsi_14"] = compute_rsi(close, window=14)

    macd_df = compute_macd(close)
    features = features.join(macd_df)
    features["bb_position"] = compute_bollinger_position(close, window=20)

    features["volume_ratio_5d"] = volume / volume.rolling(5).mean()
    features["volume_ratio_10d"] = volume / volume.rolling(10).mean()
    features["intraday_range"] = (high - low) / close
    features["gap"] = (open_ - close.shift(1)) / close.shift(1)
    features["price_vs_sma5"] = close / close.rolling(5).mean() - 1
    features["price_vs_sma10"] = close / close.rolling(10).mean() - 1

    obv = (np.sign(close.diff()) * volume).cumsum()
    features["obv_slope_5d"] = obv.rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan, raw=False
    )
    return features


# ---------------------------------------------------------------------------
# Read from BigQuery
# ---------------------------------------------------------------------------

def read_bigquery() -> tuple[pd.DataFrame, pd.DataFrame]:
    client = bigquery.Client(project=PROJECT_ID)

    log.info("Reading market data from BigQuery...")
    market_df = client.query(f"""
        SELECT date, ticker, open, high, low, close, volume
        FROM `{BQ_MARKET_TABLE}`
        WHERE date >= '2020-01-01'
        ORDER BY ticker, date
    """).to_dataframe()
    market_df["date"] = pd.to_datetime(market_df["date"])

    log.info("Reading news data from BigQuery...")
    news_df = client.query(f"""
        SELECT date, ticker, headline
        FROM `{BQ_NEWS_TABLE}`
        WHERE date >= '2020-01-01'
        ORDER BY ticker, date
    """).to_dataframe()
    news_df["date"] = pd.to_datetime(news_df["date"])

    log.info(f"Market rows: {len(market_df):,}  News rows: {len(news_df):,}")
    return market_df, news_df


# ---------------------------------------------------------------------------
# Build features
# ---------------------------------------------------------------------------

def build_technical_features(market_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ticker, grp in market_df.groupby("ticker"):
        grp = grp.set_index("date").sort_index()
        grp.columns = [c.capitalize() for c in grp.columns]
        tech = compute_technical_features(grp)
        tech["close"] = grp["Close"]
        tech["ticker"] = ticker
        tech.index.name = "date"
        frames.append(tech.reset_index().set_index(["ticker", "date"]))
    return pd.concat(frames)


def aggregate_headlines(news_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate all headlines per (ticker, date) into a single string."""
    if news_df.empty:
        return pd.DataFrame(columns=["ticker", "date", "headlines_text"])
    agg = news_df.groupby(["ticker", "date"])["headline"].apply(lambda x: " ".join(x.astype(str))).reset_index()
    agg.columns = ["ticker", "date", "headlines_text"]
    return agg


def build_gold_table(market_df: pd.DataFrame, news_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (gold_df, headlines_df) where:
    - gold_df has technical features + target, indexed by (ticker, date)
    - headlines_df has (ticker, date, headlines_text) for TF-IDF later
    """
    log.info("Building technical features...")
    tech_df = build_technical_features(market_df)

    # Target: next-day close direction
    gold = tech_df.reset_index()
    gold["next_return"] = gold.groupby("ticker")["close"].pct_change().shift(-1)
    gold["target"] = (gold["next_return"] > 0).astype(int)
    gold = gold.dropna(subset=["target", "next_return"])
    gold = gold.drop(columns=["close", "next_return"])

    # Headlines aggregated per (ticker, date)
    headlines_df = aggregate_headlines(news_df)

    log.info(f"Gold table: {len(gold):,} rows")
    return gold, headlines_df


# ---------------------------------------------------------------------------
# LightGBM training params
# ---------------------------------------------------------------------------

_LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_estimators=500,
    verbose=-1,
)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_model(gold: pd.DataFrame, headlines_df: pd.DataFrame) -> tuple:
    """
    Train LightGBM with TimeSeriesSplit.
    TF-IDF is fit only on the training portion to avoid data leakage.
    Returns (final_model, tfidf_vectorizer, metrics_dict).
    """
    # Merge headlines into gold
    gold = gold.merge(headlines_df, on=["ticker", "date"], how="left")
    gold["headlines_text"] = gold["headlines_text"].fillna("")

    # Sort by date for time series split
    gold = gold.sort_values("date")

    tech_feature_cols = [c for c in gold.columns if c not in ("ticker", "date", "target", "headlines_text")]
    X_tech = gold[tech_feature_cols].fillna(0)
    y = gold["target"]
    texts = gold["headlines_text"]

    tscv = TimeSeriesSplit(n_splits=5)
    aucs, accs = [], []

    log.info("Cross-validating (TimeSeriesSplit, 5 folds)...")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_tech), 1):
        # Fit TF-IDF only on training fold
        tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))
        tfidf_train = tfidf.fit_transform(texts.iloc[tr_idx])
        tfidf_val = tfidf.transform(texts.iloc[val_idx])

        tfidf_train_df = pd.DataFrame(
            tfidf_train.toarray(),
            columns=[f"tfidf_{c}" for c in tfidf.get_feature_names_out()],
            index=X_tech.iloc[tr_idx].index,
        )
        tfidf_val_df = pd.DataFrame(
            tfidf_val.toarray(),
            columns=[f"tfidf_{c}" for c in tfidf.get_feature_names_out()],
            index=X_tech.iloc[val_idx].index,
        )

        X_tr = pd.concat([X_tech.iloc[tr_idx].reset_index(drop=True), tfidf_train_df.reset_index(drop=True)], axis=1)
        X_val = pd.concat([X_tech.iloc[val_idx].reset_index(drop=True), tfidf_val_df.reset_index(drop=True)], axis=1)
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        m = lgb.LGBMClassifier(**_LGB_PARAMS)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        proba = m.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        acc = accuracy_score(y_val, proba >= 0.5)
        aucs.append(auc)
        accs.append(acc)
        log.info(f"  Fold {fold}  AUC={auc:.4f}  Acc={acc:.4f}")

    # Final model: fit TF-IDF on all data, train on all data
    final_tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))
    tfidf_all = final_tfidf.fit_transform(texts)
    tfidf_all_df = pd.DataFrame(
        tfidf_all.toarray(),
        columns=[f"tfidf_{c}" for c in final_tfidf.get_feature_names_out()],
        index=X_tech.index,
    )
    X_full = pd.concat([X_tech.reset_index(drop=True), tfidf_all_df.reset_index(drop=True)], axis=1)

    final_model = lgb.LGBMClassifier(**_LGB_PARAMS)
    final_model.fit(X_full, y)

    feature_names = list(X_full.columns)
    metrics = {
        "cv_auc_mean": round(float(np.mean(aucs)), 4),
        "cv_auc_std": round(float(np.std(aucs)), 4),
        "cv_acc_mean": round(float(np.mean(accs)), 4),
        "n_features": len(feature_names),
        "n_train_rows": len(gold),
        "feature_names": feature_names,
        "tech_feature_names": tech_feature_cols,
        "trained_at": datetime.utcnow().isoformat(),
    }
    log.info(f"CV AUC: {metrics['cv_auc_mean']:.4f} +/- {metrics['cv_auc_std']:.4f}")
    log.info(f"CV Accuracy: {metrics['cv_acc_mean']:.4f}")
    return final_model, final_tfidf, metrics


# ---------------------------------------------------------------------------
# Save artifacts to GCS
# ---------------------------------------------------------------------------

def save_artifacts(model, tfidf_vectorizer, metrics: dict) -> str:
    """Save model, TF-IDF vectorizer, and metadata to GCS. Returns GCS URI."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    gcs_dir = f"models/{MODEL_NAME}/{ts}"
    gcs_uri = f"gs://{GCS_BUCKET}/{gcs_dir}"

    with tempfile.TemporaryDirectory() as tmpdir:
        joblib.dump(model, os.path.join(tmpdir, "model.joblib"))
        joblib.dump(tfidf_vectorizer, os.path.join(tmpdir, "tfidf_vectorizer.joblib"))
        with open(os.path.join(tmpdir, "metadata.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        bucket = storage.Client(project=PROJECT_ID).bucket(GCS_BUCKET)
        for fname in ("model.joblib", "tfidf_vectorizer.joblib", "metadata.json"):
            blob = bucket.blob(f"{gcs_dir}/{fname}")
            blob.upload_from_filename(os.path.join(tmpdir, fname))
            log.info(f"Uploaded gs://{GCS_BUCKET}/{gcs_dir}/{fname}")

    return gcs_uri


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(artifact_uri_path: str, accuracy_path: str, auc_path: str) -> None:
    """
    KFP component entry point.
    Writes outputs to file paths provided by KFP.
    """
    log.info("=" * 60)
    log.info("Train Component — Starting")
    log.info("=" * 60)

    market_df, news_df = read_bigquery()
    gold, headlines_df = build_gold_table(market_df, news_df)
    model, tfidf_vectorizer, metrics = train_model(gold, headlines_df)
    gcs_uri = save_artifacts(model, tfidf_vectorizer, metrics)

    # Write KFP outputs
    with open(artifact_uri_path, "w") as f:
        f.write(gcs_uri)
    with open(accuracy_path, "w") as f:
        f.write(str(metrics["cv_acc_mean"]))
    with open(auc_path, "w") as f:
        f.write(str(metrics["cv_auc_mean"]))

    log.info(f"Artifacts saved to {gcs_uri}")
    log.info(f"Accuracy: {metrics['cv_acc_mean']:.4f}")
    log.info("Train Component — Done")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-uri-path", required=True)
    parser.add_argument("--accuracy-path", required=True)
    parser.add_argument("--auc-path", required=True)
    args = parser.parse_args()
    main(args.artifact_uri_path, args.accuracy_path, args.auc_path)
```

- [ ] **Step 2: Commit**

```bash
git add pipeline/components/train.py
git commit -m "feat: add KFP train component with TF-IDF text features"
```

---

### Task 4: Create the evaluate component

**Files:**
- Create: `pipeline/components/evaluate.py`

This is a lightweight component. In the current design, CV accuracy is computed during training. The evaluate component receives the accuracy metric from the train step and makes it available for the conditional deploy step. It could also do additional validation (e.g., load model and run on a separate test set), but for the first pass it simply passes through the CV accuracy.

- [ ] **Step 1: Write `pipeline/components/evaluate.py`**

```python
"""
KFP Evaluate Component — receives training metrics and decides pass/fail.
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)


def main(accuracy: float, auc: float, min_accuracy: float, deploy_decision_path: str) -> None:
    """
    Evaluate model metrics against thresholds.
    Writes 'true' or 'false' to deploy_decision_path.
    """
    log.info(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, Min accuracy threshold: {min_accuracy:.4f}")

    should_deploy = accuracy > min_accuracy
    decision = "true" if should_deploy else "false"

    log.info(f"Deploy decision: {decision}")
    with open(deploy_decision_path, "w") as f:
        f.write(decision)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy", type=float, required=True)
    parser.add_argument("--auc", type=float, required=True)
    parser.add_argument("--min-accuracy", type=float, default=0.50)
    parser.add_argument("--deploy-decision-path", required=True)
    args = parser.parse_args()
    main(args.accuracy, args.auc, args.min_accuracy, args.deploy_decision_path)
```

- [ ] **Step 2: Commit**

```bash
git add pipeline/components/evaluate.py
git commit -m "feat: add KFP evaluate component"
```

---

### Task 5: Create the deploy component

**Files:**
- Create: `pipeline/components/deploy.py`

This component:
1. Registers model in Vertex AI Model Registry
2. Deploys to the Vertex AI endpoint (undeploys existing models first)
3. Writes metadata to BigQuery `model_metadata` table

- [ ] **Step 1: Write `pipeline/components/deploy.py`**

```python
"""
KFP Deploy Component — registers model in Vertex AI and deploys to endpoint.
"""

import json
import logging
import os
from datetime import datetime

from google.cloud import aiplatform, bigquery

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "llm-testing-489113")
REGION = os.environ.get("GCP_REGION", "us-central1")
DATASET_ID = os.environ.get("BQ_DATASET", "market_sentiment_predictor")
VERTEX_ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID", "")
SERVING_IMAGE = os.environ.get("SERVING_CONTAINER_IMAGE_URI", "")


def main(artifact_uri: str, accuracy: float, auc: float) -> None:
    log.info("=" * 60)
    log.info("Deploy Component — Starting")
    log.info("=" * 60)

    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Register model in Vertex AI Model Registry
    log.info(f"Registering model from {artifact_uri}...")
    vertex_model = aiplatform.Model.upload(
        display_name="market_sentiment_lgbm",
        artifact_uri=artifact_uri,
        serving_container_image_uri=SERVING_IMAGE,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
        serving_container_environment_variables={
            "GCP_PROJECT_ID": PROJECT_ID,
            "GCP_REGION": REGION,
            "BQ_DATASET": DATASET_ID,
        },
        labels={
            "cv_auc": str(round(auc, 4)).replace(".", "_"),
            "cv_acc": str(round(accuracy, 4)).replace(".", "_"),
        },
    )
    resource_name = vertex_model.resource_name
    log.info(f"Registered: {resource_name}")

    # Deploy to endpoint
    if not VERTEX_ENDPOINT_ID:
        log.error("VERTEX_ENDPOINT_ID not set — skipping deploy")
        return

    endpoint_id = VERTEX_ENDPOINT_ID
    if not endpoint_id.startswith("projects/"):
        endpoint_id = f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{VERTEX_ENDPOINT_ID}"

    endpoint = aiplatform.Endpoint(endpoint_id)

    # Undeploy existing models
    for dm in endpoint.list_models():
        if dm.id:
            log.info(f"Undeploying existing model {dm.id}...")
            endpoint.undeploy(deployed_model_id=dm.id)

    endpoint.deploy(
        model=vertex_model,
        traffic_percentage=100,
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=2,
    )
    log.info("Model deployed to endpoint")

    # Write metadata to BigQuery
    deployed_ids = [dm.id for dm in endpoint.list_models() if dm.id]
    deployed_model_id = deployed_ids[0] if deployed_ids else ""

    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{DATASET_ID}.model_metadata"
    row = {
        "model_id": resource_name,
        "version": datetime.utcnow().strftime("%Y%m%d%H%M"),
        "training_date": datetime.utcnow().isoformat(),
        "model_type": "LightGBM",
        "performance_metrics": json.dumps({
            "cv_auc_mean": round(auc, 4),
            "cv_acc_mean": round(accuracy, 4),
        }),
        "notes": f"Auto-deployed by pipeline. deployed_model_id={deployed_model_id}",
        "created_by": "vertex-ai-pipeline",
    }
    errors = client.insert_rows_json(table_id, [row])
    if errors:
        log.error(f"Failed to insert BQ metadata: {errors}")
    else:
        log.info("Metadata saved to BigQuery")

    log.info("Deploy Component — Done")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-uri", required=True)
    parser.add_argument("--accuracy", type=float, required=True)
    parser.add_argument("--auc", type=float, required=True)
    args = parser.parse_args()
    main(args.artifact_uri, args.accuracy, args.auc)
```

- [ ] **Step 2: Commit**

```bash
git add pipeline/components/deploy.py
git commit -m "feat: add KFP deploy component"
```

---

### Task 6: Create the KFP pipeline definition

**Files:**
- Create: `pipeline/pipeline.py`

This file defines the KFP pipeline DAG: train → evaluate → conditional deploy.

- [ ] **Step 1: Write `pipeline/pipeline.py`**

```python
"""
Vertex AI Pipeline definition — train → evaluate → deploy.

Usage:
    python pipeline/pipeline.py  # compiles to pipeline.json
"""

from kfp import dsl
from kfp.dsl import component, Input, Output, Artifact, Metrics, Dataset

PIPELINE_IMAGE = ""  # Set at compile time or via env var
import os
PIPELINE_IMAGE = os.environ.get(
    "PIPELINE_IMAGE",
    "us-central1-docker.pkg.dev/llm-testing-489113/market-sentiment/pipeline:latest",
)


@dsl.component(base_image=PIPELINE_IMAGE)
def train_component(
    artifact_uri: dsl.OutputPath(str),
    accuracy: dsl.OutputPath(float),
    auc: dsl.OutputPath(float),
):
    from pipeline.components.train import main
    main(artifact_uri, accuracy, auc)


@dsl.component(base_image=PIPELINE_IMAGE)
def evaluate_component(
    accuracy: float,
    auc: float,
    min_accuracy: float,
    deploy_decision: dsl.OutputPath(str),
):
    from pipeline.components.evaluate import main
    main(accuracy, auc, min_accuracy, deploy_decision)


@dsl.component(base_image=PIPELINE_IMAGE)
def deploy_component(
    artifact_uri: str,
    accuracy: float,
    auc: float,
):
    from pipeline.components.deploy import main
    main(artifact_uri, accuracy, auc)


@dsl.pipeline(
    name="market-sentiment-pipeline",
    description="Train, evaluate, and conditionally deploy market sentiment model",
)
def market_sentiment_pipeline(min_accuracy: float = 0.50):
    # Step 1: Train
    train_task = train_component()

    # Step 2: Evaluate
    eval_task = evaluate_component(
        accuracy=train_task.outputs["accuracy"],
        auc=train_task.outputs["auc"],
        min_accuracy=min_accuracy,
    )

    # Step 3: Deploy (conditional)
    with dsl.Condition(eval_task.outputs["deploy_decision"] == "true"):
        deploy_component(
            artifact_uri=train_task.outputs["artifact_uri"],
            accuracy=train_task.outputs["accuracy"],
            auc=train_task.outputs["auc"],
        )


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=market_sentiment_pipeline,
        package_path="pipeline.json",
    )
    print("Pipeline compiled to pipeline.json")
```

- [ ] **Step 2: Commit**

```bash
git add pipeline/pipeline.py
git commit -m "feat: add KFP pipeline definition (train → eval → deploy)"
```

---

## Chunk 3: Update Cloud Function, Endpoint, and Infrastructure

### Task 7: Update Cloud Function to trigger pipeline

**Files:**
- Modify: `cloud_function/main.py` — replace `trigger_training()` with `trigger_pipeline()`, call it at end of `run_pipeline()`
- Modify: `cloud_function/requirements.txt` — ensure `google-cloud-aiplatform` is present

- [ ] **Step 1: Update `cloud_function/main.py`**

Replace the `trigger_training()` function (lines 158-199) with `trigger_pipeline()`.
Update `run_pipeline()` to call `trigger_pipeline()` at the end.

Key changes:
- Remove `trigger_training()` function entirely
- Add `trigger_pipeline()` that submits a Vertex AI PipelineJob
- Add new env var reads: `GCP_REGION`, `GCS_BUCKET`, `PIPELINE_TEMPLATE_URI`, `VERTEX_ENDPOINT_ID`, `SERVING_CONTAINER_IMAGE_URI`, `MIN_ACCURACY`
- Call `trigger_pipeline()` at end of `run_pipeline()`

```python
# New function to replace trigger_training():
def trigger_pipeline():
    """Submit a Vertex AI Pipeline run after data ingestion."""
    from google.cloud import aiplatform

    region = os.environ.get("GCP_REGION", "us-central1")
    gcs_bucket = os.environ.get("GCS_BUCKET", "")
    template_uri = os.environ.get("PIPELINE_TEMPLATE_URI", "")
    min_accuracy = float(os.environ.get("MIN_ACCURACY", "0.50"))

    if not template_uri:
        logger.warning("PIPELINE_TEMPLATE_URI not set — skipping pipeline trigger")
        return

    aiplatform.init(project=PROJECT_ID, location=region)

    job = aiplatform.PipelineJob(
        display_name="market-sentiment-pipeline-run",
        template_path=template_uri,
        pipeline_root=f"gs://{gcs_bucket}/pipeline_runs",
        parameter_values={"min_accuracy": min_accuracy},
    )
    job.submit()
    logger.info(f"Submitted pipeline job: {job.resource_name}")
```

Update `run_pipeline()` to end with:
```python
    logger.info("Pipeline execution finished successfully.")
    trigger_pipeline()
    return "OK"
```

- [ ] **Step 2: Commit**

```bash
git add cloud_function/main.py cloud_function/requirements.txt
git commit -m "feat: cloud function triggers Vertex AI Pipeline after ingestion"
```

---

### Task 8: Update endpoint to use TF-IDF instead of Gemini

**Files:**
- Modify: `endpoint/predictor.py` — remove Gemini, load TF-IDF vectorizer, inline technical features
- Modify: `endpoint/Dockerfile` — remove `COPY ml/feature_engineering.py`
- Modify: `endpoint/requirements.txt` — remove `vaderSentiment`, `vertexai`; add `scikit-learn`

- [ ] **Step 1: Rewrite `endpoint/predictor.py`**

The full rewrite replaces Gemini scoring with TF-IDF vectorizer loaded from GCS. Technical feature computation is inlined. Remove all imports of `vertexai`, `GenerativeModel`, and `feature_engineering`.

Key changes:
- Remove: `import vertexai`, `from vertexai.generative_models import ...`, `from feature_engineering import ...`
- Remove: `GEMINI_MODEL` env var, `_GEMINI_PROMPT`, `_NEUTRAL_SCORE`, `_DIRECTION_MAP`, `_MAGNITUDE_MAP`, `_EVENT_TYPES`
- Remove: `_score_headlines()` method, `gemini` property
- Add: `self.tfidf_vectorizer` loaded from GCS alongside model
- Add: inlined `compute_technical_features()` and helpers (same as in train.py)
- Update `_build_feature_row()` to use TF-IDF on headlines instead of Gemini
- Update `load()` to also load `tfidf_vectorizer.joblib`

- [ ] **Step 2: Update `endpoint/Dockerfile`**

Remove the line:
```
COPY ml/feature_engineering.py ./feature_engineering.py
```

The Dockerfile should be:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY endpoint/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY endpoint/main.py      ./main.py
COPY endpoint/predictor.py  ./predictor.py
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 3: Update `endpoint/requirements.txt`**

```
fastapi
uvicorn[standard]
pydantic>=2
joblib
numpy
pandas
lightgbm
scikit-learn
google-cloud-bigquery[pandas]
google-cloud-storage
```

- [ ] **Step 4: Commit**

```bash
git add endpoint/predictor.py endpoint/Dockerfile endpoint/requirements.txt
git commit -m "feat: endpoint uses TF-IDF instead of Gemini, inline technical features"
```

---

### Task 9: Update Terraform

**Files:**
- Modify: `terraform/main.tf` — remove training trigger CF + scheduler, add pipeline env vars to ingestion CF
- Modify: `terraform/outputs.tf` — remove training-related outputs
- Modify: `terraform/variables.tf` — remove `frontend_url` (only used for approval API CORS)

- [ ] **Step 1: Update `terraform/main.tf`**

Remove sections 9 and 10 entirely:
- `google_cloudfunctions2_function.trigger_training` (lines 239-273)
- `google_cloud_scheduler_job.weekly_training` (lines 279-296)

Update section 7 (`google_cloudfunctions2_function.function`) environment_variables to add:
```hcl
GCP_REGION                  = var.region
GCS_BUCKET                  = google_storage_bucket.model_artifacts.name
PIPELINE_TEMPLATE_URI       = "gs://${google_storage_bucket.model_artifacts.name}/pipelines/market-sentiment-pipeline.json"
VERTEX_ENDPOINT_ID          = google_vertex_ai_endpoint.predictor.name
SERVING_CONTAINER_IMAGE_URI = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.market_sentiment.repository_id}/endpoint:latest"
MIN_ACCURACY                = "0.50"
```

Also increase timeout to 540s (already set) and memory to "1Gi" (pipeline submission may need more).

- [ ] **Step 2: Update `terraform/outputs.tf`**

Remove:
- `training_scheduler_job` output
- `training_trigger_function_uri` output

- [ ] **Step 3: Update `terraform/variables.tf`**

Remove `frontend_url` variable (no longer needed — approval API removed).

- [ ] **Step 4: Commit**

```bash
git add terraform/main.tf terraform/outputs.tf terraform/variables.tf
git commit -m "feat: terraform removes training trigger, adds pipeline env vars"
```

---

### Task 10: Update root requirements.txt

**Files:**
- Modify: `requirements.txt` — update for local dev

- [ ] **Step 1: Update `requirements.txt`**

```
pandas
numpy
scikit-learn
lightgbm
joblib
google-cloud-bigquery[pandas]
google-cloud-storage
google-cloud-aiplatform
kfp>=2.7.0
yfinance
newsapi-python
pyarrow
db-dtypes
python-dotenv
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: update root requirements.txt for new pipeline workflow"
```

---

## Chunk 4: Update Documentation

### Task 11: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

Key changes:
- Update repository layout to match new structure
- Update architecture diagram (text version)
- Remove approval_api references
- Update Dockerfile build commands (add pipeline, remove training)
- Remove Gemini references from env vars
- Update data flow description
- Update key technology choices table
- Remove `frontend_url` from terraform commands
- Update coding conventions

- [ ] **Step 1: Rewrite `CLAUDE.md`**

Full rewrite reflecting new architecture. The new CLAUDE.md should document:
- New folder structure (cloud_function, pipeline, endpoint, frontend, terraform)
- New architecture: CF → Pipeline (train → eval → deploy)
- TF-IDF instead of Gemini
- Updated env vars for all components
- Updated Dockerfile build commands
- Pipeline compilation step
- No approval API

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for pipeline restructure"
```

---

### Task 12: Update README.md

**Files:**
- Modify: `README.md`

Key changes:
- Update architecture diagram
- Update project structure
- Replace "Build & Push Docker Images" to include pipeline image, remove training/approval images
- Replace training section with pipeline info
- Remove approval API section
- Update local development section
- Add pipeline compilation step

- [ ] **Step 1: Rewrite `README.md`**

Full rewrite reflecting the new workflow:

```markdown
# Market Sentiment & Price Predictor

End-to-end ML pipeline: ingest financial news + stock data → BigQuery → Vertex AI Pipeline trains LightGBM → auto-deploy → advisor-facing Next.js UI predicts tomorrow's Up/Down price direction.

## Architecture

Cloud Scheduler (Sun 00:00)
  └─► Cloud Function v2           — Yahoo Finance + NewsAPI → BigQuery
        └─► Vertex AI Pipeline     — Train (TF-IDF + LightGBM) → Evaluate → Deploy (if acc > 50%)
Vertex AI Endpoint                — serves predictions (FastAPI + TF-IDF)
Vercel (frontend)                 — advisor enters tickers + headlines → Up/Down

## Prerequisites
... (same as before minus approval API)

## Deploy Infrastructure
... (terraform apply without frontend_url)

## Build & Push Docker Images
... (endpoint + pipeline images only)

## Pipeline Compilation
... (compile pipeline.py → upload to GCS)

## Run Data Ingestion
... (same)

## Frontend
... (same)

## Local Development
... (updated for new structure)

## Project Structure
... (new layout)
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for pipeline restructure"
```

---

### Task 13: Final verification

- [ ] **Step 1: Verify directory structure**

```bash
find . -type f -not -path './.git/*' -not -path './frontend/node_modules/*' -not -path './frontend/.next/*' -not -path './terraform/.terraform/*' -not -path './venv/*' | sort
```

Expected: Only files under `cloud_function/`, `pipeline/`, `endpoint/`, `frontend/`, `terraform/`, `docs/`, plus root files.

- [ ] **Step 2: Verify no broken references**

```bash
grep -r "feature_engineering" --include="*.py" --include="*.dockerfile" --include="Dockerfile" .
grep -r "gemini" --include="*.py" -i .
grep -r "approval_api" --include="*.py" --include="*.tf" --include="*.md" .
grep -r "trigger_training" --include="*.py" --include="*.tf" .
grep -r "vaderSentiment" --include="*.py" --include="*.txt" .
```

Expected: No hits for old references.

- [ ] **Step 3: Verify Dockerfiles build context**

Check that all Dockerfiles reference files relative to project root correctly:
- `endpoint/Dockerfile` — no longer copies `ml/feature_engineering.py`
- `pipeline/Dockerfile` — copies `pipeline/` directory

- [ ] **Step 4: Final commit if any cleanup needed**

```bash
git status
# If clean, no commit needed
```
