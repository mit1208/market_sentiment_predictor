"""
Training component for the market sentiment model.

Feature selection writes silver candidate snapshots and an approved feature
contract. A dedicated ETL step materializes the approved gold table, and the
training job trains an LGB/XGB ensemble only on that selected gold subset.
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
import optuna
import pandas as pd
import xgboost as xgb
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
    CANDIDATE_FEATURE_NAMES,
    CANDIDATE_FEATURE_VERSION,
    DEFAULT_GEMINI_FEATURES,
    LABEL_NOISE_THRESHOLD,
    NEWS_LOOKBACK_DAYS,
    build_candidate_feature_frame,
    build_market_context_feature_frame,
    build_market_context_history,
    build_market_temporal_feature_frame_from_history,
    candidate_feature_version,
    coerce_scalarlike,
    is_missing_value,
    validate_online_feature_names,
)
from backend.ml.shared.gemini_sentiment import build_training_sentiment_feature_table
from backend.ml.shared.macro_features import build_training_macro_feature_frame
from backend.ml.shared.snapshot_storage import (
    SNAPSHOT_SOURCE_HISTORICAL,
    build_live_feature_snapshot_row,
)
from backend.ml.training.automl_selection import load_selected_feature_contract
from backend.ml.training.context_validation import validate_context_history_coverage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ID = get_env("GCP_PROJECT_ID")
REGION = get_env("GCP_REGION")
DATASET_ID = get_env("BQ_DATASET", "BQ_DATASET_ID")
GCS_BUCKET = get_env("GCS_BUCKET")
MODEL_NAME = get_env("MODEL_NAME", default="market_sentiment_lgbm")

BQ_MARKET_TABLE = get_env("BQ_MARKET_TABLE")
BQ_NEWS_TABLE = get_env("BQ_NEWS_TABLE")
BQ_RAW_MACRO_TABLE = get_env("BQ_RAW_MACRO_TABLE")
BQ_TRAINING_SNAPSHOT_TABLE = get_env("TRAINING_FEATURE_SNAPSHOT_TABLE")
BQ_GOLD_TRAINING_TABLE = get_env("GOLD_TRAINING_FEATURE_TABLE")
TRAINING_TICKERS = [ticker.upper() for ticker in get_csv_env("TRAINING_TICKERS")]
TRAINING_HISTORY_START = get_env("TRAINING_HISTORY_START", "BOOTSTRAP_MARKET_START")
CONTEXT_TICKERS = {"SPY", "^VIX"}

CV_FOLDS = 5
HOLDOUT_FRACTION = 0.20
MIN_HOLDOUT_DATES = 20
OPTUNA_TRIALS = get_int_env("OPTUNA_TRIALS", default=25) or 25
PROMOTION_ACCURACY_THRESHOLD = 0.52
BLEND_LGB_WEIGHT_CANDIDATES = (1.0, 0.85, 0.7, 0.5, 0.3, 0.15, 0.0)
CLASSIFICATION_THRESHOLD_CANDIDATES = tuple(round(value, 2) for value in np.linspace(0.4, 0.7, 31))


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


def _raw_macro_table() -> str:
    return BQ_RAW_MACRO_TABLE or qualify_table_name(
        "raw_macro_data",
        project_id=PROJECT_ID or None,
        dataset_id=DATASET_ID or None,
    )


def _gold_training_table() -> str:
    return BQ_GOLD_TRAINING_TABLE or qualify_table_name(
        "gold_training_features",
        project_id=PROJECT_ID or None,
        dataset_id=DATASET_ID or None,
    )


def _gemini_cache_table() -> str:
    return qualify_table_name(
        "gemini_sentiment_cache",
        project_id=PROJECT_ID or None,
        dataset_id=DATASET_ID or None,
    )


def _training_ticker_table() -> str:
    return qualify_table_name(
        "training_tickers",
        project_id=PROJECT_ID or None,
        dataset_id=DATASET_ID or None,
    )


def _resolve_training_tickers() -> list[str]:
    if TRAINING_TICKERS:
        return TRAINING_TICKERS

    try:
        rows = _client().query(
            f"""
            SELECT ticker
            FROM `{_training_ticker_table()}`
            WHERE is_active IS TRUE
            ORDER BY sector, ticker
            """
        ).result()
    except Exception as exc:
        log.warning("Failed to read training tickers from %s: %s", _training_ticker_table(), exc)
        return []

    return [str(row.ticker).upper() for row in rows if getattr(row, "ticker", None)]


def read_bigquery_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    client = _client()
    market_where = f"WHERE DATE(date) >= '{TRAINING_HISTORY_START}'" if TRAINING_HISTORY_START else ""
    news_where = f"WHERE DATE(date) >= '{TRAINING_HISTORY_START}'" if TRAINING_HISTORY_START else ""
    macro_where = (
        f"WHERE observation_date >= DATE_SUB(DATE('{TRAINING_HISTORY_START}'), INTERVAL 180 DAY)"
        if TRAINING_HISTORY_START
        else ""
    )

    market_df = client.query(
        f"""
        SELECT date, ticker, open, high, low, close, volume
        FROM `{_market_table()}`
        {market_where}
        ORDER BY ticker, date
        """
    ).to_dataframe()
    market_df["date"] = pd.to_datetime(market_df["date"]).dt.normalize()

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

    macro_df = client.query(
        f"""
        SELECT observation_date, series_id, value, source, fetched_at
        FROM `{_raw_macro_table()}`
        {macro_where}
        ORDER BY observation_date, series_id
        """
    ).to_dataframe()
    if not macro_df.empty:
        macro_df["observation_date"] = pd.to_datetime(macro_df["observation_date"]).dt.normalize()

    log.info(
        "Market rows: %s  News rows: %s  Macro rows: %s",
        f"{len(market_df):,}",
        f"{len(news_df):,}",
        f"{len(macro_df):,}",
    )
    return market_df, news_df, macro_df


def ensure_raw_training_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market_df, news_df, macro_df = read_bigquery_raw()
    if market_df.empty:
        raise RuntimeError("Bronze raw_market_data is empty. Run ingestion before feature selection or training.")
    if macro_df.empty:
        raise RuntimeError("Bronze raw_macro_data is empty. Run ingestion before feature selection or training.")

    available_tickers = set(market_df["ticker"].dropna().astype(str).str.upper())
    missing_context = sorted(CONTEXT_TICKERS - available_tickers)
    if missing_context:
        raise RuntimeError(
            f"Bronze raw_market_data is missing required context tickers: {', '.join(missing_context)}."
        )

    return market_df, news_df, macro_df


def build_historical_feature_snapshots(
    market_df: pd.DataFrame,
    news_df: pd.DataFrame,
    raw_macro_df: pd.DataFrame,
) -> tuple[list[dict], dict, pd.DataFrame]:
    training_tickers = _resolve_training_tickers()
    if not training_tickers:
        raise RuntimeError("TRAINING_TICKERS is not configured and training_tickers is empty.")

    normalized_market = market_df.copy()
    normalized_market["ticker"] = normalized_market["ticker"].fillna("").astype(str).str.upper()
    normalized_market["date"] = pd.to_datetime(normalized_market["date"], errors="coerce").dt.normalize()
    training_market_df = normalized_market[normalized_market["ticker"].isin(training_tickers)].copy()
    if training_market_df.empty:
        raise RuntimeError("Bronze raw_market_data does not contain any active training tickers.")

    spy_history = normalized_market[normalized_market["ticker"] == "SPY"][
        ["date", "open", "high", "low", "close", "volume"]
    ].copy()
    vix_history = normalized_market[normalized_market["ticker"] == "^VIX"][
        ["date", "open", "high", "low", "close", "volume"]
    ].copy()
    if spy_history.empty or vix_history.empty:
        raise RuntimeError("Bronze raw_market_data must contain both SPY and ^VIX rows for silver feature creation.")
    validate_context_history_coverage(
        training_market_df,
        spy_history=spy_history,
        vix_history=vix_history,
    )

    spy_return_5d = spy_history.sort_values("date").set_index("date")["close"].astype(float).pct_change(5)
    context_history = build_market_context_history(spy_history, vix_history)

    frames: list[pd.DataFrame] = []
    for ticker, group in training_market_df.groupby("ticker", sort=False):
        history = group.sort_values("date").set_index("date")
        history.columns = [column.capitalize() for column in history.columns]
        frame = build_market_temporal_feature_frame_from_history(history, spy_return_5d=spy_return_5d)
        frame["ticker"] = ticker
        frame["close"] = history["Close"]
        frame.index.name = "date"
        frames.append(frame.reset_index())

    if not frames:
        raise RuntimeError("No market history available to build feature snapshots.")

    feature_df = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    feature_df["next_close"] = feature_df.groupby("ticker")["close"].shift(-1)
    feature_df["next_day_return"] = feature_df["next_close"] / feature_df["close"] - 1
    feature_df["target"] = np.where(
        feature_df["next_close"].notna(),
        (feature_df["next_close"] > feature_df["close"]).astype(int),
        np.nan,
    )

    sentiment_df, newly_scored = build_training_sentiment_feature_table(
        news_df,
        feature_df[["ticker", "date", "day_change_pct"]],
        project_id=PROJECT_ID,
        region=REGION,
        cache_client=_client(),
        cache_table_name=_gemini_cache_table(),
        lookback_days=NEWS_LOOKBACK_DAYS,
    )
    snapshot_df = feature_df.merge(sentiment_df, on=["ticker", "date"], how="left")
    for column, default_value in DEFAULT_GEMINI_FEATURES.items():
        snapshot_df[column] = snapshot_df[column].fillna(default_value)

    context_df = build_market_context_feature_frame(snapshot_df["date"], context_history)
    macro_features = build_training_macro_feature_frame(
        snapshot_df["date"],
        market_df=training_market_df,
        raw_macro_df=raw_macro_df,
        active_tickers=training_tickers,
    )

    candidate_df = build_candidate_feature_frame(
        market_features=snapshot_df,
        temporal_features=snapshot_df,
        sentiment_features=snapshot_df,
        context_features=context_df,
        macro_features=macro_features,
        row_count=len(snapshot_df),
    )
    snapshot_df = pd.concat(
        [
            snapshot_df[["date", "ticker", "close", "target", "next_day_return"]].reset_index(drop=True),
            candidate_df.reset_index(drop=True),
        ],
        axis=1,
    )

    missing_candidates = [column for column in CANDIDATE_FEATURE_NAMES if column not in snapshot_df.columns]
    if missing_candidates:
        raise AssertionError(f"Missing candidate feature columns: {missing_candidates}")

    snapshot_df = snapshot_df.drop(columns=["next_close"], errors="ignore")
    row_count_before_drop = len(snapshot_df)
    snapshot_df = snapshot_df.dropna(subset=[*CANDIDATE_FEATURE_NAMES, "target", "next_day_return"]).copy()
    dropped_for_missing_history = row_count_before_drop - len(snapshot_df)
    if snapshot_df.empty:
        raise RuntimeError("Silver feature creation produced no fully trainable rows.")

    created_at = datetime.now(timezone.utc)
    rows: list[dict] = []
    for row in snapshot_df.itertuples(index=False):
        candidate_values = {feature_name: getattr(row, feature_name) for feature_name in CANDIDATE_FEATURE_NAMES}
        rows.append(
            build_live_feature_snapshot_row(
                snapshot_id=f"hist:{row.ticker}:{pd.Timestamp(row.date).strftime('%Y-%m-%d')}",
                ticker=row.ticker,
                snapshot_timestamp=pd.Timestamp(row.date).tz_localize("UTC") + pd.Timedelta(hours=16),
                headlines_text="",
                base_feature_values=candidate_values,
                target=None if pd.isna(row.target) else int(row.target),
                next_day_return=None if pd.isna(row.next_day_return) else float(row.next_day_return),
                snapshot_source=SNAPSHOT_SOURCE_HISTORICAL,
                created_at=created_at,
                feature_snapshot={
                    "candidate_feature_version": candidate_feature_version(),
                    "candidate_feature_count": len(CANDIDATE_FEATURE_NAMES),
                    "data_layer": "silver",
                },
            )
        )

    metadata = {
        "tickers": sorted(snapshot_df["ticker"].dropna().unique().tolist()),
        "ticker_count": int(snapshot_df["ticker"].nunique()),
        "training_start_date": str(snapshot_df["date"].min().date()),
        "training_end_date": str(snapshot_df["date"].max().date()),
        "market_rows": int(len(training_market_df)),
        "news_rows": int(len(news_df)),
        "macro_rows": int(len(raw_macro_df)),
        "snapshot_rows": int(len(rows)),
        "snapshot_table": _training_snapshot_table(),
        "gold_table": _gold_training_table(),
        "candidate_feature_count": len(CANDIDATE_FEATURE_NAMES),
        "candidate_feature_version": CANDIDATE_FEATURE_VERSION,
        "new_gemini_cache_rows": int(len(newly_scored)),
        "silver_rows_dropped_for_missing_history": int(dropped_for_missing_history),
    }
    return rows, metadata, snapshot_df


def refresh_historical_feature_snapshots(rows: list[dict]) -> None:
    client = _client()
    snapshot_df = pd.DataFrame(rows)
    if snapshot_df.empty:
        raise RuntimeError("Historical snapshot refresh received no rows.")

    snapshot_df["snapshot_date"] = pd.to_datetime(snapshot_df["snapshot_date"]).dt.date
    snapshot_df["snapshot_timestamp"] = pd.to_datetime(snapshot_df["snapshot_timestamp"], utc=True)
    snapshot_df["created_at"] = pd.to_datetime(snapshot_df["created_at"], utc=True)
    client.load_table_from_dataframe(
        snapshot_df,
        _training_snapshot_table(),
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    ).result()


def read_training_feature_snapshots() -> pd.DataFrame:
    client = _client()
    select_columns = ", ".join([f"`{column}`" for column in CANDIDATE_FEATURE_NAMES])
    df = client.query(
        f"""
        SELECT
          snapshot_date AS date,
          ticker,
          target,
          next_day_return,
          {select_columns}
        FROM `{_training_snapshot_table()}`
        WHERE target IS NOT NULL
        ORDER BY snapshot_date, ticker
        """
    ).to_dataframe()
    if df.empty:
        raise RuntimeError("No labeled rows found in the training snapshot table.")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["target"] = df["target"].astype(int)
    df["next_day_return"] = pd.to_numeric(df["next_day_return"], errors="coerce")
    return df


def build_snapshot_metadata(snapshot_df: pd.DataFrame) -> dict:
    if snapshot_df.empty:
        raise RuntimeError("No labeled rows found in the training snapshot table.")
    return {
        "tickers": sorted(snapshot_df["ticker"].dropna().astype(str).str.upper().unique().tolist()),
        "ticker_count": int(snapshot_df["ticker"].nunique()),
        "training_start_date": str(pd.Timestamp(snapshot_df["date"].min()).date()),
        "training_end_date": str(pd.Timestamp(snapshot_df["date"].max()).date()),
        "snapshot_rows": int(len(snapshot_df)),
        "snapshot_table": _training_snapshot_table(),
        "gold_table": _gold_training_table(),
        "candidate_feature_count": len(CANDIDATE_FEATURE_NAMES),
        "candidate_feature_version": CANDIDATE_FEATURE_VERSION,
        "market_rows": None,
        "news_rows": None,
        "macro_rows": None,
        "new_gemini_cache_rows": None,
        "silver_rows_dropped_for_missing_history": None,
    }


def _to_python_scalar(value):
    value = coerce_scalarlike(value)
    if is_missing_value(value):
        return None
    return value


def materialize_gold_training_features(
    silver_df: pd.DataFrame,
    *,
    feature_contract: dict,
) -> dict:
    if silver_df.empty:
        raise RuntimeError("Cannot materialize gold training features from an empty silver dataset.")

    selected_features = validate_online_feature_names(feature_contract["selected_feature_names"])
    materialized = silver_df[["date", "ticker", "target", "next_day_return", *selected_features]].dropna().copy()
    if materialized.empty:
        raise RuntimeError("No rows remained after selecting approved training features for the gold table.")

    created_at = datetime.now(timezone.utc)
    materialization_id = f"gold:{feature_contract['run_id']}:{created_at.strftime('%Y%m%dT%H%M%SZ')}"
    rows = []
    for row in materialized.itertuples(index=False):
        feature_values = {
            feature_name: _to_python_scalar(getattr(row, feature_name))
            for feature_name in selected_features
        }
        rows.append(
            {
                "materialization_id": materialization_id,
                "source_snapshot_id": f"hist:{row.ticker}:{pd.Timestamp(row.date).strftime('%Y-%m-%d')}",
                "feature_selection_run_id": feature_contract["run_id"],
                "candidate_feature_version": feature_contract["candidate_feature_version"],
                "selected_feature_count": len(selected_features),
                "selected_feature_names": selected_features,
                "ticker": row.ticker,
                "snapshot_date": pd.Timestamp(row.date).date().isoformat(),
                "target": int(row.target),
                "next_day_return": float(row.next_day_return),
                "feature_values": json.dumps(feature_values, separators=(",", ":")),
                "created_at": created_at.isoformat(),
            }
        )

    client = _client()
    delete_query = f"DELETE FROM `{_gold_training_table()}` WHERE feature_selection_run_id = @run_id"
    delete_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", feature_contract["run_id"])]
    )
    client.query(delete_query, job_config=delete_config).result()

    load_df = pd.DataFrame(rows)
    load_df["snapshot_date"] = pd.to_datetime(load_df["snapshot_date"]).dt.date
    load_df["created_at"] = pd.to_datetime(load_df["created_at"], utc=True)
    client.load_table_from_dataframe(
        load_df,
        _gold_training_table(),
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
    ).result()
    return {
        "gold_materialization_id": materialization_id,
        "gold_rows": int(len(load_df)),
        "gold_table": _gold_training_table(),
        "feature_selection_run_id": feature_contract["run_id"],
        "selected_feature_count": len(selected_features),
    }


def read_gold_training_features(feature_contract: dict) -> pd.DataFrame:
    selected_features = validate_online_feature_names(feature_contract["selected_feature_names"])
    client = _client()
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", feature_contract["run_id"])]
    )
    df = client.query(
        f"""
        SELECT snapshot_date AS date, ticker, target, next_day_return, feature_values
        FROM `{_gold_training_table()}`
        WHERE feature_selection_run_id = @run_id
        ORDER BY snapshot_date, ticker
        """,
        job_config=job_config,
    ).to_dataframe()
    if df.empty:
        raise RuntimeError(
            "Gold training table is empty for the approved feature-selection contract. "
            "Materialize the gold table before training."
        )

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["target"] = df["target"].astype(int)
    df["next_day_return"] = pd.to_numeric(df["next_day_return"], errors="coerce")
    feature_payloads = df["feature_values"].fillna("{}").map(json.loads)
    feature_frame = pd.DataFrame(list(feature_payloads))
    for feature_name in selected_features:
        if feature_name not in feature_frame.columns:
            raise RuntimeError(f"Gold training row is missing approved feature '{feature_name}'.")
        df[feature_name] = pd.to_numeric(feature_frame[feature_name], errors="coerce")
    return df[["date", "ticker", "target", "next_day_return", *selected_features]]


def build_gold_metadata(gold_df: pd.DataFrame, *, feature_contract: dict, silver_metadata: dict | None = None) -> dict:
    if gold_df.empty:
        raise RuntimeError("No labeled rows found in the gold training table.")

    selected_features = validate_online_feature_names(feature_contract["selected_feature_names"])
    is_temporary_contract = bool(feature_contract.get("is_temporary"))
    metadata = {
        "tickers": sorted(gold_df["ticker"].dropna().astype(str).str.upper().unique().tolist()),
        "ticker_count": int(gold_df["ticker"].nunique()),
        "training_start_date": str(pd.Timestamp(gold_df["date"].min()).date()),
        "training_end_date": str(pd.Timestamp(gold_df["date"].max()).date()),
        "snapshot_rows": int(len(gold_df)),
        "snapshot_table": _training_snapshot_table(),
        "gold_table": _gold_training_table(),
        "candidate_feature_count": len(CANDIDATE_FEATURE_NAMES),
        "candidate_feature_version": CANDIDATE_FEATURE_VERSION,
        "selected_feature_count": len(selected_features),
        "feature_selection_run_id": feature_contract["run_id"],
        "feature_selection_source": feature_contract.get("selection_source"),
        "feature_selection_mode": feature_contract.get("feature_selection_mode"),
        "feature_selection_review_status": feature_contract.get("review_status"),
        "feature_selection_is_temporary": is_temporary_contract,
        "market_rows": None,
        "news_rows": None,
        "macro_rows": None,
        "new_gemini_cache_rows": None,
        "silver_rows_dropped_for_missing_history": None,
    }
    if silver_metadata:
        metadata.update(
            {
                "market_rows": silver_metadata.get("market_rows"),
                "news_rows": silver_metadata.get("news_rows"),
                "macro_rows": silver_metadata.get("macro_rows"),
                "new_gemini_cache_rows": silver_metadata.get("new_gemini_cache_rows"),
                "silver_rows_dropped_for_missing_history": silver_metadata.get("silver_rows_dropped_for_missing_history"),
            }
        )
    return metadata


def _round_metric(value: float | None, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _safe_auc(y_true: pd.Series, proba: np.ndarray) -> float | None:
    if pd.Series(y_true).nunique() < 2:
        return None
    return float(roc_auc_score(y_true, proba))


def _compute_binary_metrics(y_true: pd.Series, proba: np.ndarray, *, threshold: float = 0.5) -> dict:
    predicted = (proba >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    auc = _safe_auc(y_true, proba)

    return {
        "accuracy": _round_metric(accuracy_score(y_true, predicted)),
        "auc": _round_metric(auc),
        "precision": _round_metric(precision_score(y_true, predicted, zero_division=0)),
        "recall": _round_metric(recall_score(y_true, predicted, zero_division=0)),
        "f1": _round_metric(f1_score(y_true, predicted, zero_division=0)),
        "support": int(len(y_true)),
        "confusion_matrix": {
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        },
        "classification_threshold": _round_metric(threshold, digits=3),
    }


def _blend_probabilities(lgb_proba: np.ndarray, xgb_proba: np.ndarray, *, lgb_weight: float) -> np.ndarray:
    lgb_array = np.asarray(lgb_proba, dtype=float)
    xgb_array = np.asarray(xgb_proba, dtype=float)
    return (float(lgb_weight) * lgb_array) + ((1.0 - float(lgb_weight)) * xgb_array)


def _strategy_name_for_weight(lgb_weight: float) -> str:
    if np.isclose(lgb_weight, 1.0):
        return "lightgbm_only"
    if np.isclose(lgb_weight, 0.0):
        return "xgboost_only"
    if np.isclose(lgb_weight, 0.5):
        return "equal_weight_ensemble"
    return f"weighted_ensemble_lgb_{int(round(lgb_weight * 100))}"


def _select_prediction_strategy(
    *,
    y_true: pd.Series,
    lgb_proba: np.ndarray,
    xgb_proba: np.ndarray,
) -> dict:
    y_series = pd.Series(y_true).astype(int).reset_index(drop=True)
    lgb_array = np.asarray(lgb_proba, dtype=float)
    xgb_array = np.asarray(xgb_proba, dtype=float)
    if len(y_series) != len(lgb_array) or len(y_series) != len(xgb_array):
        raise RuntimeError("Prediction-strategy selection received misaligned OOF arrays.")

    coverage_mask = ~(np.isnan(lgb_array) | np.isnan(xgb_array))
    covered_rows = int(np.count_nonzero(coverage_mask))
    total_rows = int(len(y_series))
    if covered_rows <= 0:
        raise RuntimeError("Prediction-strategy selection requires at least one covered OOF prediction.")

    y_series = y_series[coverage_mask].reset_index(drop=True)
    lgb_array = lgb_array[coverage_mask]
    xgb_array = xgb_array[coverage_mask]
    coverage_fraction = covered_rows / total_rows if total_rows else 0.0

    best: dict | None = None
    for lgb_weight in BLEND_LGB_WEIGHT_CANDIDATES:
        blended = _blend_probabilities(lgb_array, xgb_array, lgb_weight=lgb_weight)
        auc = _safe_auc(y_series, blended)
        for threshold in CLASSIFICATION_THRESHOLD_CANDIDATES:
            accuracy = float(accuracy_score(y_series, (blended >= threshold).astype(int)))
            candidate = {
                "prediction_strategy": _strategy_name_for_weight(lgb_weight),
                "ensemble_lgb_weight": round(float(lgb_weight), 4),
                "ensemble_xgb_weight": round(float(1.0 - lgb_weight), 4),
                "classification_threshold": round(float(threshold), 3),
                "cv_accuracy": float(accuracy),
                "cv_auc": None if auc is None else float(auc),
                "oof_rows": covered_rows,
                "oof_total_rows": total_rows,
                "oof_coverage": round(float(coverage_fraction), 4),
            }
            if best is None:
                best = candidate
                continue

            candidate_auc = candidate["cv_auc"] if candidate["cv_auc"] is not None else -1.0
            best_auc = best["cv_auc"] if best["cv_auc"] is not None else -1.0
            if accuracy > best["cv_accuracy"] + 1e-12:
                best = candidate
                continue
            if np.isclose(accuracy, best["cv_accuracy"]) and candidate_auc > best_auc + 1e-12:
                best = candidate

    if best is None:
        raise RuntimeError("Failed to select a prediction strategy from OOF predictions.")
    return best


def _chronological_holdout_split(gold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    unique_dates = np.array(sorted(pd.to_datetime(gold["date"]).dt.normalize().unique()))
    if len(unique_dates) < 10:
        raise RuntimeError("Not enough unique dates to build a chronological holdout set.")

    holdout_count = max(int(np.ceil(len(unique_dates) * HOLDOUT_FRACTION)), MIN_HOLDOUT_DATES)
    if holdout_count >= len(unique_dates):
        holdout_count = max(1, len(unique_dates) // 5)

    holdout_start = pd.Timestamp(unique_dates[-holdout_count])
    train_df = gold[gold["date"] < holdout_start].copy().reset_index(drop=True)
    holdout_df = gold[gold["date"] >= holdout_start].copy().reset_index(drop=True)
    if train_df.empty or holdout_df.empty:
        raise RuntimeError("Chronological split produced an empty train or holdout partition.")
    return train_df, holdout_df, holdout_start


def _date_split_indices(dataset: pd.DataFrame, n_splits: int):
    unique_dates = np.array(sorted(pd.to_datetime(dataset["date"]).dt.normalize().unique()))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    normalized_dates = pd.to_datetime(dataset["date"]).dt.normalize()
    for fold, (train_date_idx, val_date_idx) in enumerate(splitter.split(unique_dates), start=1):
        train_dates = unique_dates[train_date_idx]
        val_dates = unique_dates[val_date_idx]
        train_idx = np.flatnonzero(normalized_dates.isin(train_dates).to_numpy())
        val_idx = np.flatnonzero(normalized_dates.isin(val_dates).to_numpy())
        yield fold, train_idx, val_idx, train_dates, val_dates


def _lightgbm_search_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 24, 96),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
    }


def _xgboost_search_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 2.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "n_estimators": trial.suggest_int("n_estimators", 200, 700),
    }


def _build_lgb_model(params: dict) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        verbosity=-1,
        **params,
    )


def _build_xgb_model(params: dict) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        **params,
    )


def _fit_model(model_type: str, params: dict, X_tr: pd.DataFrame, y_tr: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
    if model_type == "lgb":
        model = _build_lgb_model(params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        return model

    model = _build_xgb_model(params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def _tune_model(
    model_type: str,
    *,
    train_df: pd.DataFrame,
    selected_features: list[str],
    n_splits: int,
) -> dict:
    y_train = train_df["target"].astype(int)

    def objective(trial: optuna.Trial) -> float:
        params = _lightgbm_search_space(trial) if model_type == "lgb" else _xgboost_search_space(trial)
        scores: list[float] = []
        for _fold, train_idx, val_idx, _train_dates, _val_dates in _date_split_indices(train_df, n_splits):
            X_tr = train_df.iloc[train_idx][selected_features].reset_index(drop=True)
            X_val = train_df.iloc[val_idx][selected_features].reset_index(drop=True)
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]
            model = _fit_model(model_type, params, X_tr, y_tr, X_val, y_val)
            proba = model.predict_proba(X_val)[:, 1]
            scores.append(float(accuracy_score(y_val, (proba >= 0.5).astype(int))))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    return study.best_params


def train_model(gold: pd.DataFrame, dataset_metadata: dict) -> tuple[object, object, dict]:
    gold = gold.sort_values(["date", "ticker"]).reset_index(drop=True)
    filtered = gold[gold["next_day_return"].abs() >= LABEL_NOISE_THRESHOLD].copy()
    if filtered.empty:
        raise RuntimeError("Label-noise filtering removed all rows from the training dataset.")

    feature_contract = load_selected_feature_contract(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        require_approved=True,
        allow_temporary_fallback=False,
    )
    if not feature_contract:
        raise RuntimeError(
            "No approved feature-selection contract exists for the current candidate feature version. "
            "Run the feature-ranking submit/finalize jobs, review the selected features, then mark the contract APPROVED in BigQuery."
        )
    selected_features = validate_online_feature_names(feature_contract["selected_feature_names"])
    missing_selected = [feature for feature in selected_features if feature not in gold.columns]
    if missing_selected:
        raise RuntimeError(
            "Gold training table does not match the approved feature contract. "
            f"Missing selected features: {missing_selected}"
        )

    train_df, holdout_df, holdout_start = _chronological_holdout_split(filtered)
    train_unique_dates = train_df["date"].dt.normalize().nunique()
    n_splits = min(CV_FOLDS, train_unique_dates - 1)
    if n_splits < 2:
        raise RuntimeError("Not enough training dates available for time-series cross-validation.")

    log.info("Tuning LightGBM on %d selected features...", len(selected_features))
    best_lgb_params = _tune_model("lgb", train_df=train_df, selected_features=selected_features, n_splits=n_splits)
    log.info("Tuning XGBoost on %d selected features...", len(selected_features))
    best_xgb_params = _tune_model("xgb", train_df=train_df, selected_features=selected_features, n_splits=n_splits)

    fold_records: list[dict] = []
    oof_lgb = np.full(len(train_df), np.nan, dtype=float)
    oof_xgb = np.full(len(train_df), np.nan, dtype=float)

    for fold, train_idx, val_idx, train_dates, val_dates in _date_split_indices(train_df, n_splits):
        X_tr = train_df.iloc[train_idx][selected_features].reset_index(drop=True)
        X_val = train_df.iloc[val_idx][selected_features].reset_index(drop=True)
        y_tr = train_df.iloc[train_idx]["target"].astype(int)
        y_val = train_df.iloc[val_idx]["target"].astype(int)

        lgb_model = _fit_model("lgb", best_lgb_params, X_tr, y_tr, X_val, y_val)
        xgb_model = _fit_model("xgb", best_xgb_params, X_tr, y_tr, X_val, y_val)
        lgb_proba = lgb_model.predict_proba(X_val)[:, 1]
        xgb_proba = xgb_model.predict_proba(X_val)[:, 1]
        oof_lgb[val_idx] = lgb_proba
        oof_xgb[val_idx] = xgb_proba
        fold_records.append(
            {
                "fold": fold,
                "y_val": y_val.reset_index(drop=True),
                "lgb_proba": lgb_proba,
                "xgb_proba": xgb_proba,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "train_start_date": str(pd.Timestamp(train_dates[0]).date()),
                "train_end_date": str(pd.Timestamp(train_dates[-1]).date()),
                "validation_start_date": str(pd.Timestamp(val_dates[0]).date()),
                "validation_end_date": str(pd.Timestamp(val_dates[-1]).date()),
            }
        )

    prediction_strategy = _select_prediction_strategy(
        y_true=train_df["target"].astype(int),
        lgb_proba=oof_lgb,
        xgb_proba=oof_xgb,
    )
    log.info(
        "Selected prediction strategy %s (lgb_weight=%.2f threshold=%.2f cv_accuracy=%.4f oof_coverage=%s/%s)",
        prediction_strategy["prediction_strategy"],
        prediction_strategy["ensemble_lgb_weight"],
        prediction_strategy["classification_threshold"],
        prediction_strategy["cv_accuracy"],
        prediction_strategy["oof_rows"],
        prediction_strategy["oof_total_rows"],
    )

    fold_metrics = []
    accs: list[float] = []
    aucs: list[float] = []
    for record in fold_records:
        proba = _blend_probabilities(
            record["lgb_proba"],
            record["xgb_proba"],
            lgb_weight=prediction_strategy["ensemble_lgb_weight"],
        )
        fold_result = _compute_binary_metrics(
            record["y_val"],
            proba,
            threshold=prediction_strategy["classification_threshold"],
        )
        fold_result.update(
            {
                "fold": record["fold"],
                "train_rows": record["train_rows"],
                "validation_rows": record["validation_rows"],
                "train_start_date": record["train_start_date"],
                "train_end_date": record["train_end_date"],
                "validation_start_date": record["validation_start_date"],
                "validation_end_date": record["validation_end_date"],
                "prediction_strategy": prediction_strategy["prediction_strategy"],
                "ensemble_lgb_weight": prediction_strategy["ensemble_lgb_weight"],
                "ensemble_xgb_weight": prediction_strategy["ensemble_xgb_weight"],
            }
        )
        fold_metrics.append(fold_result)
        if fold_result["accuracy"] is not None:
            accs.append(fold_result["accuracy"])
        if fold_result["auc"] is not None:
            aucs.append(fold_result["auc"])

    X_train_eval = train_df[selected_features].reset_index(drop=True)
    y_train = train_df["target"].astype(int)
    X_holdout = holdout_df[selected_features].reset_index(drop=True)
    y_holdout = holdout_df["target"].astype(int)
    lgb_eval_model = _fit_model("lgb", best_lgb_params, X_train_eval, y_train, X_holdout, y_holdout)
    xgb_eval_model = _fit_model("xgb", best_xgb_params, X_train_eval, y_train, X_holdout, y_holdout)
    holdout_lgb_proba = lgb_eval_model.predict_proba(X_holdout)[:, 1]
    holdout_xgb_proba = xgb_eval_model.predict_proba(X_holdout)[:, 1]
    holdout_proba = _blend_probabilities(
        holdout_lgb_proba,
        holdout_xgb_proba,
        lgb_weight=prediction_strategy["ensemble_lgb_weight"],
    )
    holdout_metrics = _compute_binary_metrics(
        y_holdout,
        holdout_proba,
        threshold=prediction_strategy["classification_threshold"],
    )
    holdout_metrics.update(
        {
            "holdout_start_date": str(pd.Timestamp(holdout_start).date()),
            "holdout_end_date": str(holdout_df["date"].max().date()),
            "prediction_strategy": prediction_strategy["prediction_strategy"],
            "ensemble_lgb_weight": prediction_strategy["ensemble_lgb_weight"],
            "ensemble_xgb_weight": prediction_strategy["ensemble_xgb_weight"],
        }
    )

    X_full = filtered[selected_features].reset_index(drop=True)
    y_full = filtered["target"].astype(int)
    lgb_final = _build_lgb_model(best_lgb_params)
    lgb_final.fit(X_full, y_full)
    xgb_final = _build_xgb_model(best_xgb_params)
    xgb_final.fit(X_full, y_full)

    lgb_importance = pd.Series(lgb_final.feature_importances_, index=selected_features, dtype=float)
    xgb_importance = pd.Series(xgb_final.feature_importances_, index=selected_features, dtype=float)
    lgb_weight = float(prediction_strategy["ensemble_lgb_weight"])
    xgb_weight = float(prediction_strategy["ensemble_xgb_weight"])
    combined_importance = (
        (lgb_importance / max(lgb_importance.max(), 1.0)) * lgb_weight
        + (xgb_importance / max(xgb_importance.max(), 1.0)) * xgb_weight
    )
    feature_importance = [
        {"feature": feature, "importance": round(float(score), 6)}
        for feature, score in combined_importance.sort_values(ascending=False).items()
    ]

    cv_auc_mean = _round_metric(float(np.mean(aucs)) if aucs else None)
    cv_auc_std = _round_metric(float(np.std(aucs)) if aucs else None)
    cv_acc_mean = _round_metric(float(np.mean(accs)) if accs else None)
    cv_acc_std = _round_metric(float(np.std(accs)) if accs else None)
    trained_at = datetime.now(timezone.utc).isoformat()
    is_temporary_contract = bool(feature_contract.get("is_temporary"))
    evaluation_notes = (
        "Bronze raw API tables feed silver candidate features, the approved AutoML feature "
        "contract is materialized into the gold training table, and the ensemble trains only "
        "on that gold subset."
    )
    if is_temporary_contract:
        evaluation_notes = (
            "Bronze raw API tables feed silver candidate features, a temporary deterministic "
            "40-feature subset is materialized into the gold training table while Vertex "
            "AutoML ranking is pending, and the ensemble trains only on that temporary subset."
        )

    metrics = {
        "model_name": MODEL_NAME,
        "model_type": "AdaptiveBlend(LightGBM,XGBoost)",
        "trained_at": trained_at,
        "candidate_feature_names": CANDIDATE_FEATURE_NAMES,
        "candidate_feature_count": len(CANDIDATE_FEATURE_NAMES),
        "candidate_feature_version": candidate_feature_version(),
        "feature_names": selected_features,
        "selected_feature_count": len(selected_features),
        "n_features": len(selected_features),
        "feature_selection_source": feature_contract.get("selection_source"),
        "feature_selection_mode": feature_contract.get("feature_selection_mode"),
        "feature_selection_is_temporary": is_temporary_contract,
        "feature_selection_review_status": feature_contract.get("review_status"),
        "prediction_strategy": prediction_strategy["prediction_strategy"],
        "ensemble_lgb_weight": prediction_strategy["ensemble_lgb_weight"],
        "ensemble_xgb_weight": prediction_strategy["ensemble_xgb_weight"],
        "classification_threshold": prediction_strategy["classification_threshold"],
        "oof_rows": prediction_strategy["oof_rows"],
        "oof_total_rows": prediction_strategy["oof_total_rows"],
        "oof_coverage": prediction_strategy["oof_coverage"],
        "n_train_rows": int(len(filtered)),
        "filtered_out_rows": int(len(gold) - len(filtered)),
        "train_rows_before_holdout": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "label_noise_threshold": LABEL_NOISE_THRESHOLD,
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
                "oof_rows": prediction_strategy["oof_rows"],
                "oof_total_rows": prediction_strategy["oof_total_rows"],
                "oof_coverage": prediction_strategy["oof_coverage"],
            },
        },
        "test_metrics": holdout_metrics,
        "performance_metrics": {
            "promotion_gate_metric": "cv_accuracy_mean",
            "promotion_gate_threshold": PROMOTION_ACCURACY_THRESHOLD,
            "promotion_gate_value": cv_acc_mean,
            "prediction_strategy": prediction_strategy["prediction_strategy"],
            "classification_threshold": prediction_strategy["classification_threshold"],
            "oof_coverage": prediction_strategy["oof_coverage"],
            "holdout_accuracy": holdout_metrics["accuracy"],
            "holdout_auc": holdout_metrics["auc"],
        },
        "feature_importance": feature_importance,
        "hyperparameters": {
            "lightgbm": best_lgb_params,
            "xgboost": best_xgb_params,
            "optuna_trials": OPTUNA_TRIALS,
        },
        "automl_feature_contract": feature_contract,
        "class_weights": {
            "down": int((filtered["target"] == 0).sum()),
            "up": int((filtered["target"] == 1).sum()),
        },
        "evaluation_notes": evaluation_notes,
        **dataset_metadata,
    }
    return lgb_final, xgb_final, metrics


def build_performance_report(metrics: dict) -> str:
    validation_summary = metrics["validation_metrics"]["summary"]
    holdout_metrics = metrics["test_metrics"]
    top_features = metrics["feature_importance"][:10]

    feature_lines = "\n".join(
        f"{rank}. {row['feature']} ({row['importance']:.4f})"
        for rank, row in enumerate(top_features, start=1)
    )

    return f"""# Model Performance Report

## Overview

- Model: {metrics['model_name']} ({metrics['model_type']})
- Silver table: {metrics['snapshot_table']}
- Gold table: {metrics['gold_table']}
- Training window: {metrics['training_start_date']} to {metrics['training_end_date']}
- Candidate features: {metrics['candidate_feature_count']}
- Selected features: {metrics['selected_feature_count']}
- Feature selection source: {metrics.get('feature_selection_source')}
- Temporary fallback: {metrics.get('feature_selection_is_temporary')}
- Prediction strategy: {metrics.get('prediction_strategy')}
- Classification threshold: {metrics.get('classification_threshold')}
- Training samples after label-noise filter: {metrics['n_train_rows']}
- Filtered rows removed: {metrics['filtered_out_rows']}

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

## Top Selected Features

{feature_lines or "No feature importance available."}

## Notes

{metrics['evaluation_notes']}
"""


def save_artifacts(lgb_model, xgb_model, metrics: dict) -> str:
    if not GCS_BUCKET:
        raise RuntimeError("GCS_BUCKET is required to save model artifacts.")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    gcs_dir = f"models/{MODEL_NAME}/{ts}"
    gcs_uri = f"gs://{GCS_BUCKET}/{gcs_dir}"
    performance_report = build_performance_report(metrics)

    with tempfile.TemporaryDirectory() as tmpdir:
        joblib.dump(lgb_model, os.path.join(tmpdir, "lgb_model.joblib"))
        joblib.dump(xgb_model, os.path.join(tmpdir, "xgb_model.joblib"))
        with open(os.path.join(tmpdir, "metadata.json"), "w") as handle:
            json.dump(metrics, handle, indent=2)
        with open(os.path.join(tmpdir, "performance_report.md"), "w") as handle:
            handle.write(performance_report)

        bucket = storage.Client(project=PROJECT_ID or None).bucket(GCS_BUCKET)
        for filename in ("lgb_model.joblib", "xgb_model.joblib", "metadata.json", "performance_report.md"):
            blob = bucket.blob(f"{gcs_dir}/{filename}")
            blob.upload_from_filename(os.path.join(tmpdir, filename))

    return gcs_uri


def prepare_candidate_dataset(*, rebuild_snapshots: bool = True) -> tuple[pd.DataFrame, dict]:
    if rebuild_snapshots:
        market_df, news_df, raw_macro_df = ensure_raw_training_data()
        snapshot_rows, dataset_metadata, _snapshot_df = build_historical_feature_snapshots(
            market_df,
            news_df,
            raw_macro_df,
        )
        refresh_historical_feature_snapshots(snapshot_rows)
        silver = read_training_feature_snapshots()
        dataset_metadata["training_rows"] = int(len(silver))
        dataset_metadata["data_layer"] = "silver"
        return silver, dataset_metadata

    silver = read_training_feature_snapshots()
    dataset_metadata = build_snapshot_metadata(silver)
    dataset_metadata["training_rows"] = int(len(silver))
    dataset_metadata["data_layer"] = "silver"
    return silver, dataset_metadata


def prepare_gold_training_dataset(*, rebuild_gold: bool = True) -> tuple[pd.DataFrame, dict]:
    feature_contract = load_selected_feature_contract(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        require_approved=True,
        allow_temporary_fallback=False,
    )
    if not feature_contract:
        raise RuntimeError(
            "No approved feature-selection contract exists for the current candidate feature version. "
            "Run the feature-ranking submit/finalize jobs, review the selected features, then mark the contract APPROVED in BigQuery."
        )

    silver_metadata = None
    if rebuild_gold:
        silver, silver_metadata = prepare_candidate_dataset(rebuild_snapshots=False)
        gold_materialization = materialize_gold_training_features(silver, feature_contract=feature_contract)
        silver_metadata.update(gold_materialization)

    try:
        gold = read_gold_training_features(feature_contract)
    except RuntimeError as exc:
        if not rebuild_gold:
            raise RuntimeError(
                "Gold training data is not prepared for the current feature contract. "
                "Run the silver-to-gold training ETL before starting model training."
            ) from exc
        raise
    dataset_metadata = build_gold_metadata(gold, feature_contract=feature_contract, silver_metadata=silver_metadata)
    dataset_metadata["training_rows"] = int(len(gold))
    dataset_metadata["data_layer"] = "gold"
    return gold, dataset_metadata


def run_gold_training_etl(*, rebuild_gold: bool = True) -> tuple[pd.DataFrame, dict]:
    """Materialize the selected silver subset into gold before training."""
    return prepare_gold_training_dataset(rebuild_gold=rebuild_gold)


def load_prepared_gold_training_dataset() -> tuple[pd.DataFrame, dict]:
    """Read the prepared gold training dataset without mutating warehouse tables."""
    return prepare_gold_training_dataset(rebuild_gold=False)


def prepare_training_dataset(*, rebuild_snapshots: bool = True) -> tuple[pd.DataFrame, dict]:
    return prepare_gold_training_dataset(rebuild_gold=rebuild_snapshots)


def main(artifact_uri_path: str, accuracy_path: str, auc_path: str) -> None:
    log.info("Train Component — Starting")
    gold, dataset_metadata = load_prepared_gold_training_dataset()
    lgb_model, xgb_model, metrics = train_model(gold, dataset_metadata)
    gcs_uri = save_artifacts(lgb_model, xgb_model, metrics)

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
