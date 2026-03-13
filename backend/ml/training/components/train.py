"""
Training component for the market sentiment model.

The training job first materializes live-feature-compatible snapshots into
BigQuery, then reads those stored rows back as the canonical training source.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from google.cloud import bigquery, storage
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from backend.config import get_csv_env, get_env, get_int_env, qualify_table_name
from backend.ml.shared.feature_engineering import (
    BASE_FEATURE_NAMES,
    DEFAULT_MARKET_CONTEXT_FEATURES,
    DEFAULT_SENTIMENT_FEATURES,
    EXPECTED_FEATURE_COUNT,
    MARKET_FEATURE_NAMES,
    MODEL_FEATURE_NAMES,
    NEWS_LOOKBACK_DAYS,
    SENTIMENT_FEATURE_NAMES,
    TFIDF_FEATURE_NAMES,
    align_model_features,
    build_market_context_feature_frame,
    build_market_context_history,
    build_training_news_feature_table,
    compute_market_features_from_history,
    fit_tfidf_safely,
    prepare_price_history_frame,
    transform_tfidf_frame,
)
from backend.ml.shared.snapshot_storage import (
    SNAPSHOT_SOURCE_HISTORICAL,
    build_live_feature_snapshot_row,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ID = get_env("GCP_PROJECT_ID")
REGION = get_env("GCP_REGION")
DATASET_ID = get_env("BQ_DATASET", "BQ_DATASET_ID")
GCS_BUCKET = get_env("GCS_BUCKET")
MODEL_NAME = get_env("MODEL_NAME")

BQ_MARKET_TABLE = get_env("BQ_MARKET_TABLE")
BQ_NEWS_TABLE = get_env("BQ_NEWS_TABLE")
BQ_LIVE_SNAPSHOT_TABLE = get_env("LIVE_FEATURE_SNAPSHOT_TABLE")
BQ_TRAINING_SNAPSHOT_TABLE = get_env("TRAINING_FEATURE_SNAPSHOT_TABLE")
BOOTSTRAP_TICKERS = [ticker.upper() for ticker in get_csv_env("TRAINING_TICKERS")]
BOOTSTRAP_MARKET_START = get_env("BOOTSTRAP_MARKET_START")
BOOTSTRAP_NEWS_DAYS = get_int_env("BOOTSTRAP_NEWS_DAYS")
TRAINING_HISTORY_START = get_env("TRAINING_HISTORY_START", "BOOTSTRAP_MARKET_START")
FINNHUB_KEY = get_env("FINNHUB_KEY", "FINNHUB_API_KEY")

CV_FOLDS = 5
HOLDOUT_FRACTION = 0.20
MIN_HOLDOUT_DATES = 20

_LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
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


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT_ID or None)


def _market_table() -> str:
    return BQ_MARKET_TABLE or qualify_table_name(
        "raw_market_data",
        project_id=PROJECT_ID or None,
        dataset_id=DATASET_ID or None,
    )


def _news_table() -> str:
    return BQ_NEWS_TABLE or qualify_table_name(
        "raw_news_data",
        project_id=PROJECT_ID or None,
        dataset_id=DATASET_ID or None,
    )


def _training_snapshot_table() -> str:
    return BQ_TRAINING_SNAPSHOT_TABLE or qualify_table_name(
        "training_feature_snapshots",
        project_id=PROJECT_ID or None,
        dataset_id=DATASET_ID or None,
    )


def read_bigquery_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    client = _client()
    market_where = f"WHERE DATE(date) >= '{TRAINING_HISTORY_START}'" if TRAINING_HISTORY_START else ""
    news_where = f"WHERE DATE(date) >= '{TRAINING_HISTORY_START}'" if TRAINING_HISTORY_START else ""

    log.info("Reading raw market data from BigQuery...")
    market_df = client.query(
        f"""
        SELECT date, ticker, open, high, low, close, volume
        FROM `{_market_table()}`
        {market_where}
        ORDER BY ticker, date
        """
    ).to_dataframe()
    market_df["date"] = pd.to_datetime(market_df["date"]).dt.normalize()

    log.info("Reading raw news data from BigQuery...")
    news_df = client.query(
        f"""
        SELECT date, timestamp, ticker, headline, source
        FROM `{_news_table()}`
        {news_where}
        ORDER BY ticker, date, timestamp
        """
    ).to_dataframe()
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.normalize()
    if "timestamp" in news_df.columns:
        news_df["timestamp"] = pd.to_datetime(news_df["timestamp"], errors="coerce", utc=True)

    log.info("Market rows: %s  News rows: %s", f"{len(market_df):,}", f"{len(news_df):,}")
    return market_df, news_df


def _bootstrap_market_history() -> pd.DataFrame:
    if not BOOTSTRAP_MARKET_START:
        raise RuntimeError("BOOTSTRAP_MARKET_START is required when bootstrapping market history.")
    rows = []
    start_date = pd.Timestamp(BOOTSTRAP_MARKET_START)
    end_date = pd.Timestamp.utcnow().tz_localize(None)

    for ticker in BOOTSTRAP_TICKERS:
        log.info("Bootstrapping Yahoo market history for %s ...", ticker)
        try:
            history = yf.Ticker(ticker).history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=False,
            )
        except Exception as exc:
            log.warning("Failed to bootstrap market history for %s: %s", ticker, exc)
            continue

        prepared = prepare_price_history_frame(history)
        if prepared.empty:
            continue
        prepared["ticker"] = ticker
        rows.append(prepared[["date", "ticker", "open", "high", "low", "close", "volume"]])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _bootstrap_news_history() -> pd.DataFrame:
    if BOOTSTRAP_NEWS_DAYS is None:
        raise RuntimeError("BOOTSTRAP_NEWS_DAYS is required when bootstrapping news history.")
    if not FINNHUB_KEY:
        log.warning("FINNHUB_KEY is not configured; skipping raw news bootstrap.")
        return pd.DataFrame()

    rows: list[dict] = []
    end_date = datetime.now(timezone.utc)
    start_date = end_date - pd.Timedelta(days=BOOTSTRAP_NEWS_DAYS)

    for ticker in BOOTSTRAP_TICKERS:
        log.info("Bootstrapping Finnhub company news for %s ...", ticker)
        try:
            response = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    "symbol": ticker,
                    "from": start_date.date().isoformat(),
                    "to": end_date.date().isoformat(),
                    "token": FINNHUB_KEY,
                },
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            log.warning("Failed to bootstrap news for %s: %s", ticker, exc)
            continue

        if not isinstance(payload, list):
            continue

        for item in payload:
            published_at = item.get("datetime")
            published_at = None if published_at is None else pd.to_datetime(int(published_at), unit="s", utc=True)
            rows.append(
                {
                    "date": published_at.date() if published_at is not None else end_date.date(),
                    "timestamp": published_at,
                    "ticker": ticker,
                    "headline": item.get("headline"),
                    "source": item.get("source") or "Finnhub",
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[df["headline"].fillna("").astype(str).str.strip().ne("")]


def _load_bootstrap_frames_to_bigquery(market_df: pd.DataFrame, news_df: pd.DataFrame) -> None:
    client = _client()

    if not market_df.empty:
        market_to_load = market_df.copy()
        market_to_load["date"] = pd.to_datetime(market_to_load["date"]).dt.to_pydatetime()
        client.load_table_from_dataframe(
            market_to_load,
            BQ_MARKET_TABLE,
            _market_table(),
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
        ).result()
        log.info("Loaded %s bootstrap market rows into %s", f"{len(market_df):,}", _market_table())

    if not news_df.empty:
        news_to_load = news_df.copy()
        news_to_load["date"] = pd.to_datetime(news_to_load["date"]).dt.to_pydatetime()
        news_to_load["timestamp"] = pd.to_datetime(news_to_load["timestamp"], errors="coerce", utc=True)
        client.load_table_from_dataframe(
            news_to_load,
            _news_table(),
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
        ).result()
        log.info("Loaded %s bootstrap news rows into %s", f"{len(news_df):,}", _news_table())


def ensure_raw_training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    market_df, news_df = read_bigquery_raw()
    if not market_df.empty:
        return market_df, news_df

    log.warning("Raw BigQuery tables are empty. Bootstrapping baseline history from live providers.")
    if not BOOTSTRAP_TICKERS:
        raise RuntimeError("TRAINING_TICKERS is required when bootstrapping raw training data.")
    bootstrap_market = _bootstrap_market_history()
    bootstrap_news = _bootstrap_news_history()
    if bootstrap_market.empty:
        raise RuntimeError("Unable to bootstrap baseline market history from Yahoo.")

    _load_bootstrap_frames_to_bigquery(bootstrap_market, bootstrap_news)
    return read_bigquery_raw()


def _download_symbol_history(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    buffered_start = (pd.Timestamp(start_date) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    buffered_end = (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    try:
        history = yf.Ticker(symbol).history(start=buffered_start, end=buffered_end, auto_adjust=False)
    except Exception as exc:
        log.warning("Failed to fetch %s context history from Yahoo: %s", symbol, exc)
        return pd.DataFrame()

    return prepare_price_history_frame(history)


def _fetch_market_context_history(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    spy_history = _download_symbol_history("SPY", start_date, end_date)
    vix_history = _download_symbol_history("^VIX", start_date, end_date)
    return build_market_context_history(spy_history, vix_history)


def build_historical_feature_snapshots(
    market_df: pd.DataFrame,
    news_df: pd.DataFrame,
) -> tuple[list[dict], dict]:
    frames = []
    for ticker, group in market_df.groupby("ticker", sort=False):
        history = group.sort_values("date").set_index("date")
        history.columns = [column.capitalize() for column in history.columns]
        market_features = compute_market_features_from_history(history)
        market_features["close"] = history["Close"]
        market_features["ticker"] = ticker
        market_features.index.name = "date"
        frames.append(market_features.reset_index())

    if not frames:
        raise RuntimeError("No market history available to build feature snapshots.")

    feature_df = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    feature_df["next_close"] = feature_df.groupby("ticker")["close"].shift(-1)
    feature_df["target"] = np.where(
        feature_df["next_close"].notna(),
        (feature_df["next_close"] > feature_df["close"]).astype(int),
        np.nan,
    )

    news_features = build_training_news_feature_table(
        news_df,
        feature_df[["ticker", "date"]],
        lookback_days=NEWS_LOOKBACK_DAYS,
    )
    snapshot_df = feature_df.merge(news_features, on=["ticker", "date"], how="left")
    snapshot_df["headlines_text"] = snapshot_df["headlines_text"].fillna(DEFAULT_SENTIMENT_FEATURES["headlines_text"])
    for column in SENTIMENT_FEATURE_NAMES:
        snapshot_df[column] = snapshot_df[column].fillna(DEFAULT_SENTIMENT_FEATURES[column])

    context_history = _fetch_market_context_history(snapshot_df["date"].min(), snapshot_df["date"].max())
    context_df = build_market_context_feature_frame(snapshot_df["date"], context_history)
    snapshot_df = pd.concat([snapshot_df.reset_index(drop=True), context_df], axis=1)
    for column, default_value in DEFAULT_MARKET_CONTEXT_FEATURES.items():
        snapshot_df[column] = snapshot_df[column].fillna(default_value)

    missing_base = [column for column in BASE_FEATURE_NAMES if column not in snapshot_df.columns]
    if missing_base:
        raise AssertionError(f"Missing required base snapshot columns: {missing_base}")

    created_at = datetime.now(timezone.utc)
    rows: list[dict] = []
    for row in snapshot_df.itertuples(index=False):
        snapshot_ts = pd.Timestamp(row.date).tz_localize("UTC") + pd.Timedelta(hours=16)
        base_values = {feature_name: getattr(row, feature_name) for feature_name in BASE_FEATURE_NAMES}
        target = None if pd.isna(row.target) else int(row.target)
        rows.append(
            build_live_feature_snapshot_row(
                snapshot_id=f"hist:{row.ticker}:{pd.Timestamp(row.date).strftime('%Y-%m-%d')}",
                ticker=row.ticker,
                snapshot_timestamp=snapshot_ts,
                headlines_text=str(row.headlines_text or ""),
                base_feature_values=base_values,
                target=target,
                snapshot_source=SNAPSHOT_SOURCE_HISTORICAL,
                created_at=created_at,
            )
        )

    metadata = {
        "tickers": sorted(snapshot_df["ticker"].dropna().unique().tolist()),
        "ticker_count": int(snapshot_df["ticker"].nunique()),
        "training_start_date": str(snapshot_df["date"].min().date()),
        "training_end_date": str(snapshot_df["date"].max().date()),
        "market_rows": int(len(market_df)),
        "news_rows": int(len(news_df)),
        "snapshot_rows": int(len(rows)),
        "snapshot_table": _training_snapshot_table(),
    }
    log.info("Built %s historical feature snapshots.", f"{len(rows):,}")
    return rows, metadata


def refresh_historical_feature_snapshots(rows: list[dict]) -> None:
    client = _client()
    training_snapshot_table = _training_snapshot_table()
    log.info("Refreshing training feature snapshots in %s ...", training_snapshot_table)
    snapshot_df = pd.DataFrame(rows)
    if snapshot_df.empty:
        raise RuntimeError("Historical snapshot refresh received no rows.")

    snapshot_df["snapshot_date"] = pd.to_datetime(snapshot_df["snapshot_date"]).dt.date
    snapshot_df["snapshot_timestamp"] = pd.to_datetime(snapshot_df["snapshot_timestamp"], utc=True)
    snapshot_df["created_at"] = pd.to_datetime(snapshot_df["created_at"], utc=True)

    client.load_table_from_dataframe(
        snapshot_df,
        training_snapshot_table,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    ).result()
    log.info("Training snapshot table refreshed with %s rows.", f"{len(rows):,}")


def read_live_feature_snapshots() -> pd.DataFrame:
    client = _client()
    select_columns = ", ".join([f"`{column}`" for column in BASE_FEATURE_NAMES])

    df = client.query(
        f"""
        SELECT
          snapshot_date AS date,
          ticker,
          headlines_text,
          target,
          {select_columns}
        FROM `{_training_snapshot_table()}`
        WHERE target IS NOT NULL
        ORDER BY snapshot_date, ticker
        """
    ).to_dataframe()

    if df.empty:
        raise RuntimeError("No labeled rows found in the live feature snapshot table.")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["target"] = df["target"].astype(int)
    return df


def _round_metric(value: float | None, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _safe_auc(y_true: pd.Series, proba: np.ndarray) -> float | None:
    if pd.Series(y_true).nunique() < 2:
        return None
    return float(roc_auc_score(y_true, proba))


def _compute_binary_metrics(y_true: pd.Series, proba: np.ndarray) -> dict:
    predicted = (proba >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    auc = _safe_auc(y_true, proba)

    return {
        "accuracy": _round_metric(accuracy_score(y_true, predicted)),
        "auc": _round_metric(auc),
        "precision": _round_metric(precision_score(y_true, predicted, zero_division=0)),
        "recall": _round_metric(recall_score(y_true, predicted, zero_division=0)),
        "f1": _round_metric(f1_score(y_true, predicted, zero_division=0)),
        "support": int(len(y_true)),
        "positive_rate": _round_metric(float(np.mean(predicted))),
        "class_balance": {
            "down": int((pd.Series(y_true) == 0).sum()),
            "up": int((pd.Series(y_true) == 1).sum()),
        },
        "confusion_matrix": {
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        },
    }


def _chronological_holdout_split(gold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    unique_dates = np.array(sorted(pd.to_datetime(gold["date"]).dt.normalize().unique()))
    if len(unique_dates) < 10:
        raise RuntimeError("Not enough unique dates to build a chronological holdout set.")

    holdout_count = max(int(np.ceil(len(unique_dates) * HOLDOUT_FRACTION)), MIN_HOLDOUT_DATES)
    if holdout_count >= len(unique_dates):
        holdout_count = max(1, len(unique_dates) // 5)

    holdout_start = pd.Timestamp(unique_dates[-holdout_count])
    train_df = gold[gold["date"] < holdout_start].copy()
    holdout_df = gold[gold["date"] >= holdout_start].copy()

    if train_df.empty or holdout_df.empty:
        raise RuntimeError("Chronological split produced an empty train or holdout partition.")

    return train_df, holdout_df, holdout_start


def _date_split_indices(dataset: pd.DataFrame, n_splits: int):
    unique_dates = np.array(sorted(pd.to_datetime(dataset["date"]).dt.normalize().unique()))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold, (train_date_idx, val_date_idx) in enumerate(splitter.split(unique_dates), start=1):
        train_dates = unique_dates[train_date_idx]
        val_dates = unique_dates[val_date_idx]
        train_idx = dataset.index[dataset["date"].isin(train_dates)].to_numpy()
        val_idx = dataset.index[dataset["date"].isin(val_dates)].to_numpy()
        yield fold, train_idx, val_idx, train_dates, val_dates


def train_model(gold: pd.DataFrame, dataset_metadata: dict) -> tuple[object, object, dict]:
    gold = gold.sort_values(["date", "ticker"]).reset_index(drop=True)
    train_df, holdout_df, holdout_start = _chronological_holdout_split(gold)

    train_unique_dates = train_df["date"].dt.normalize().nunique()
    n_splits = min(CV_FOLDS, train_unique_dates - 1)
    if n_splits < 2:
        raise RuntimeError("Not enough training dates available for time-series cross-validation.")

    X_train_base = train_df[BASE_FEATURE_NAMES].fillna(0.0)
    y_train = train_df["target"].astype(int)
    train_texts = train_df["headlines_text"].fillna("").astype(str)

    fold_metrics = []
    aucs: list[float] = []
    accs: list[float] = []

    log.info("Cross-validating with date-based TimeSeriesSplit (%s folds)...", n_splits)
    for fold, train_idx, val_idx, train_dates, val_dates in _date_split_indices(train_df, n_splits):
        tfidf_vectorizer, tfidf_train_df = fit_tfidf_safely(train_texts.iloc[train_idx])
        tfidf_val_df = transform_tfidf_frame(tfidf_vectorizer, train_texts.iloc[val_idx])

        X_tr = align_model_features(
            pd.concat(
                [
                    X_train_base.iloc[train_idx].reset_index(drop=True),
                    tfidf_train_df.reset_index(drop=True),
                ],
                axis=1,
            )
        )
        X_val = align_model_features(
            pd.concat(
                [
                    X_train_base.iloc[val_idx].reset_index(drop=True),
                    tfidf_val_df.reset_index(drop=True),
                ],
                axis=1,
            )
        )
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        model = lgb.LGBMClassifier(**_LGB_PARAMS)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        proba = model.predict_proba(X_val)[:, 1]
        fold_result = _compute_binary_metrics(y_val, proba)
        fold_result.update(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "train_start_date": str(pd.Timestamp(train_dates[0]).date()),
                "train_end_date": str(pd.Timestamp(train_dates[-1]).date()),
                "validation_start_date": str(pd.Timestamp(val_dates[0]).date()),
                "validation_end_date": str(pd.Timestamp(val_dates[-1]).date()),
            }
        )
        fold_metrics.append(fold_result)

        if fold_result["accuracy"] is not None:
            accs.append(fold_result["accuracy"])
        if fold_result["auc"] is not None:
            aucs.append(fold_result["auc"])

        log.info(
            "Fold %s  AUC=%s  Acc=%s",
            fold,
            "n/a" if fold_result["auc"] is None else f"{fold_result['auc']:.4f}",
            "n/a" if fold_result["accuracy"] is None else f"{fold_result['accuracy']:.4f}",
        )

    eval_tfidf, tfidf_train_full = fit_tfidf_safely(train_texts)
    tfidf_holdout = transform_tfidf_frame(eval_tfidf, holdout_df["headlines_text"].fillna("").astype(str))

    X_train_eval = align_model_features(
        pd.concat(
            [
                train_df[BASE_FEATURE_NAMES].reset_index(drop=True).fillna(0.0),
                tfidf_train_full.reset_index(drop=True),
            ],
            axis=1,
        )
    )
    X_holdout = align_model_features(
        pd.concat(
            [
                holdout_df[BASE_FEATURE_NAMES].reset_index(drop=True).fillna(0.0),
                tfidf_holdout.reset_index(drop=True),
            ],
            axis=1,
        )
    )

    eval_model = lgb.LGBMClassifier(**_LGB_PARAMS)
    eval_model.fit(
        X_train_eval,
        y_train,
        eval_set=[(X_holdout, holdout_df["target"].astype(int))],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    holdout_proba = eval_model.predict_proba(X_holdout)[:, 1]
    holdout_metrics = _compute_binary_metrics(holdout_df["target"].astype(int), holdout_proba)
    holdout_metrics.update(
        {
            "holdout_start_date": str(pd.Timestamp(holdout_start).date()),
            "holdout_end_date": str(holdout_df["date"].max().date()),
        }
    )

    final_tfidf, tfidf_all_df = fit_tfidf_safely(gold["headlines_text"].fillna("").astype(str))
    X_full = align_model_features(
        pd.concat(
            [
                gold[BASE_FEATURE_NAMES].reset_index(drop=True).fillna(0.0),
                tfidf_all_df.reset_index(drop=True),
            ],
            axis=1,
        )
    )

    final_model = lgb.LGBMClassifier(**_LGB_PARAMS)
    final_model.fit(X_full, gold["target"].astype(int))

    feature_importance = [
        {"feature": feature, "importance": round(float(importance), 6)}
        for feature, importance in sorted(
            zip(MODEL_FEATURE_NAMES, final_model.feature_importances_, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    trained_at = datetime.now(timezone.utc).isoformat()
    cv_auc_mean = _round_metric(float(np.mean(aucs)) if aucs else None)
    cv_auc_std = _round_metric(float(np.std(aucs)) if aucs else None)
    cv_acc_mean = _round_metric(float(np.mean(accs)) if accs else None)
    cv_acc_std = _round_metric(float(np.std(accs)) if accs else None)

    metrics = {
        "model_name": MODEL_NAME,
        "model_type": "LightGBM",
        "trained_at": trained_at,
        "feature_contract": "live_snapshot_v1",
        "feature_names": MODEL_FEATURE_NAMES,
        "base_feature_names": BASE_FEATURE_NAMES,
        "market_feature_names": MARKET_FEATURE_NAMES,
        "context_feature_names": [name for name in BASE_FEATURE_NAMES if name not in MARKET_FEATURE_NAMES + SENTIMENT_FEATURE_NAMES],
        "sentiment_feature_names": SENTIMENT_FEATURE_NAMES,
        "tfidf_feature_names": TFIDF_FEATURE_NAMES,
        "n_features": EXPECTED_FEATURE_COUNT,
        "n_train_rows": int(len(gold)),
        "train_rows_before_holdout": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std": cv_auc_std,
        "cv_acc_mean": cv_acc_mean,
        "cv_acc_std": cv_acc_std,
        "validation_metrics": {
            "strategy": "date-based TimeSeriesSplit on pre-holdout window",
            "n_splits": int(n_splits),
            "folds": fold_metrics,
            "summary": {
                "accuracy_mean": cv_acc_mean,
                "accuracy_std": cv_acc_std,
                "auc_mean": cv_auc_mean,
                "auc_std": cv_auc_std,
            },
        },
        "test_metrics": holdout_metrics,
        "performance_metrics": {
            "promotion_gate_metric": "cv_accuracy_mean",
            "promotion_gate_value": cv_acc_mean,
            "holdout_accuracy": holdout_metrics["accuracy"],
            "holdout_auc": holdout_metrics["auc"],
        },
        "feature_importance": feature_importance,
        "hyperparameters": {key: value for key, value in _LGB_PARAMS.items()},
        "class_weights": {
            "down": int((gold["target"] == 0).sum()),
            "up": int((gold["target"] == 1).sum()),
        },
        "evaluation_notes": (
            "Historical live-equivalent feature snapshots are stored in BigQuery, then the "
            "model is trained from that snapshot table using date-based cross-validation and "
            "a final chronological holdout."
        ),
        **dataset_metadata,
    }

    log.info("CV AUC: %s +/- %s", cv_auc_mean, cv_auc_std)
    log.info("CV Accuracy: %s +/- %s", cv_acc_mean, cv_acc_std)
    log.info("Holdout Accuracy: %s  Holdout AUC: %s", holdout_metrics["accuracy"], holdout_metrics["auc"])
    return final_model, final_tfidf, metrics


def build_performance_report(metrics: dict) -> str:
    validation_summary = metrics["validation_metrics"]["summary"]
    holdout_metrics = metrics["test_metrics"]
    top_features = metrics["feature_importance"][:10]

    feature_lines = "\n".join(
        f"{rank}. {row['feature']} ({row['importance']:.2f})"
        for rank, row in enumerate(top_features, start=1)
    )

    return f"""# Model Performance Report

## Overview

- Model: {metrics['model_name']} ({metrics['model_type']})
- Snapshot table: {metrics['snapshot_table']}
- Training window: {metrics['training_start_date']} to {metrics['training_end_date']}
- Tickers: {metrics['ticker_count']}
- Training samples: {metrics['n_train_rows']}
- Feature count: {metrics['n_features']}

## Validation

| Metric | Mean | Std Dev |
|--------|------|---------|
| Accuracy | {validation_summary['accuracy_mean']} | {validation_summary['accuracy_std']} |
| AUC | {validation_summary['auc_mean']} | {validation_summary['auc_std']} |

## Holdout

| Metric | Value |
|--------|-------|
| Accuracy | {holdout_metrics['accuracy']} |
| AUC | {holdout_metrics['auc']} |
| Precision | {holdout_metrics['precision']} |
| Recall | {holdout_metrics['recall']} |
| F1 | {holdout_metrics['f1']} |

## Feature Importance (Top 10)

{feature_lines or "No feature importance available."}

## Notes

{metrics['evaluation_notes']}
"""


def save_artifacts(model, tfidf_vectorizer, metrics: dict) -> str:
    if not GCS_BUCKET:
        raise RuntimeError("GCS_BUCKET is required to save model artifacts.")
    if not MODEL_NAME:
        raise RuntimeError("MODEL_NAME is required to save model artifacts.")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    gcs_dir = f"models/{MODEL_NAME}/{ts}"
    gcs_uri = f"gs://{GCS_BUCKET}/{gcs_dir}"
    performance_report = build_performance_report(metrics)

    with tempfile.TemporaryDirectory() as tmpdir:
        joblib.dump(model, os.path.join(tmpdir, "model.joblib"))
        joblib.dump(tfidf_vectorizer, os.path.join(tmpdir, "tfidf_vectorizer.joblib"))
        with open(os.path.join(tmpdir, "metadata.json"), "w") as handle:
            json.dump(metrics, handle, indent=2)
        with open(os.path.join(tmpdir, "performance_report.md"), "w") as handle:
            handle.write(performance_report)

        bucket = storage.Client(project=PROJECT_ID or None).bucket(GCS_BUCKET)
        for filename in ("model.joblib", "tfidf_vectorizer.joblib", "metadata.json", "performance_report.md"):
            blob = bucket.blob(f"{gcs_dir}/{filename}")
            blob.upload_from_filename(os.path.join(tmpdir, filename))
            log.info("Uploaded gs://%s/%s/%s", GCS_BUCKET, gcs_dir, filename)

    return gcs_uri


def prepare_training_dataset() -> tuple[pd.DataFrame, dict]:
    market_df, news_df = ensure_raw_training_data()
    snapshot_rows, dataset_metadata = build_historical_feature_snapshots(market_df, news_df)
    refresh_historical_feature_snapshots(snapshot_rows)
    gold = read_live_feature_snapshots()
    dataset_metadata["training_rows"] = int(len(gold))
    return gold, dataset_metadata


def main(artifact_uri_path: str, accuracy_path: str, auc_path: str) -> None:
    log.info("=" * 60)
    log.info("Train Component — Starting")
    log.info("=" * 60)

    gold, dataset_metadata = prepare_training_dataset()
    model, tfidf_vectorizer, metrics = train_model(gold, dataset_metadata)
    gcs_uri = save_artifacts(model, tfidf_vectorizer, metrics)

    with open(artifact_uri_path, "w") as handle:
        handle.write(gcs_uri)
    with open(accuracy_path, "w") as handle:
        handle.write(str(metrics["cv_acc_mean"]))
    with open(auc_path, "w") as handle:
        handle.write(str(metrics["cv_auc_mean"]))

    log.info("Artifacts saved to %s", gcs_uri)
    log.info("Accuracy: %s", metrics["cv_acc_mean"])
    log.info("Train Component — Done")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-uri-path", required=True)
    parser.add_argument("--accuracy-path", required=True)
    parser.add_argument("--auc-path", required=True)
    args = parser.parse_args()
    main(args.artifact_uri_path, args.accuracy_path, args.auc_path)
