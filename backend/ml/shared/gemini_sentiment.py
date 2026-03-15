"""Gemini headline scoring and aggregation helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Callable, Sequence, TypeVar
from uuid import uuid4

import pandas as pd
from google.cloud import bigquery

from backend.ml.shared.feature_engineering import (
    DEFAULT_GEMINI_FEATURES,
    GEMINI_FEATURE_NAMES,
    LIVE_NEWS_SENTIMENT_ITEMS_KEY,
    NEWS_LOOKBACK_DAYS,
)

log = logging.getLogger(__name__)
ResponseT = TypeVar("ResponseT")

GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"
GEMINI_CACHE_BATCH_SIZE = 200
GEMINI_REQUEST_BATCH_SIZE = 20
GEMINI_MAX_RETRIES = 6
GEMINI_BACKOFF_SECONDS = 2.0

_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "sentiment_score": {"type": "NUMBER"},
        "magnitude": {"type": "NUMBER"},
        "category": {
            "type": "STRING",
            "enum": ["earnings", "macro", "product", "legal", "analyst", "insider", "other"],
        },
    },
    "required": ["sentiment_score", "magnitude", "category"],
}

_BATCH_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "index": {"type": "INTEGER"},
                    "sentiment_score": {"type": "NUMBER"},
                    "magnitude": {"type": "NUMBER"},
                    "category": {
                        "type": "STRING",
                        "enum": ["earnings", "macro", "product", "legal", "analyst", "insider", "other"],
                    },
                },
                "required": ["index", "sentiment_score", "magnitude", "category"],
            },
        }
    },
    "required": ["items"],
}

logging.getLogger("httpx").setLevel(logging.WARNING)


class GeminiResponseValidationError(ValueError):
    """Raised when Gemini returns an unusable payload."""


def headline_hash(headline: str) -> str:
    return hashlib.sha256(headline.strip().lower().encode("utf-8")).hexdigest()


def _normalize_category(value: object) -> str:
    normalized = str(value or "other").strip().lower()
    return normalized if normalized in {"earnings", "macro", "product", "legal", "analyst", "insider", "other"} else "other"


def _validated_score_payload(row: dict[str, Any]) -> dict[str, Any]:
    required_keys = ("sentiment_score", "magnitude", "category")
    missing_keys = [key for key in required_keys if key not in row]
    if missing_keys:
        raise GeminiResponseValidationError(
            f"Gemini payload missing required keys: {', '.join(sorted(missing_keys))}"
        )

    try:
        sentiment_score = float(row["sentiment_score"])
        magnitude = float(row["magnitude"])
    except (TypeError, ValueError) as exc:
        raise GeminiResponseValidationError("Gemini payload returned non-numeric score fields.") from exc

    if not math.isfinite(sentiment_score) or not -1.0 <= sentiment_score <= 1.0:
        raise GeminiResponseValidationError("Gemini sentiment_score must be a finite number in [-1, 1].")
    if not math.isfinite(magnitude) or not 0.0 <= magnitude <= 1.0:
        raise GeminiResponseValidationError("Gemini magnitude must be a finite number in [0, 1].")

    category = _normalize_category(row.get("category"))
    raw_category = str(row.get("category") or "").strip().lower()
    if not raw_category:
        raise GeminiResponseValidationError("Gemini payload category is empty.")
    if category != raw_category:
        raise GeminiResponseValidationError(f"Gemini payload category is unsupported: {raw_category}.")

    return {
        "sentiment_score": sentiment_score,
        "magnitude": magnitude,
        "category": category,
    }


def _normalize_score_row(row: dict[str, Any], *, ticker: str, headline: str) -> dict[str, Any]:
    normalized = _validated_score_payload(row)
    return {
        "headline_hash": headline_hash(headline),
        "ticker": ticker,
        "headline": headline,
        "sentiment_score": normalized["sentiment_score"],
        "magnitude": normalized["magnitude"],
        "category": normalized["category"],
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def _live_sentiment_label(sentiment_score: float | None) -> str:
    if sentiment_score is None:
        return "neutral"
    if sentiment_score >= 0.05:
        return "positive"
    if sentiment_score <= -0.05:
        return "negative"
    return "neutral"


def _gemini_client(project_id: str, region: str):
    from google import genai

    return genai.Client(vertexai=True, project=project_id, location=region)


def _generate_content_with_retries(
    *,
    client: Any,
    model_name: str,
    prompt: str,
    response_schema: dict[str, Any],
    response_validator: Callable[[str], ResponseT] | None = None,
) -> str | ResponseT:
    from google.genai import errors as genai_errors
    from google.genai import types

    delay_seconds = GEMINI_BACKOFF_SECONDS
    last_error: Exception | None = None
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            response_text = response.text or ""
            if response_validator is not None:
                return response_validator(response_text)
            return response_text
        except genai_errors.ClientError as exc:
            last_error = exc
            status_code = getattr(exc, "status_code", None)
            if status_code not in {429, 500, 503} or attempt == GEMINI_MAX_RETRIES:
                raise
            log.warning("Gemini request throttled (attempt %s/%s); retrying in %.1fs", attempt, GEMINI_MAX_RETRIES, delay_seconds)
            time.sleep(delay_seconds)
            delay_seconds *= 2
        except GeminiResponseValidationError as exc:
            last_error = exc
            if attempt == GEMINI_MAX_RETRIES:
                raise
            log.warning(
                "Gemini response validation failed (attempt %s/%s): %s. Retrying in %.1fs",
                attempt,
                GEMINI_MAX_RETRIES,
                exc,
                delay_seconds,
            )
            time.sleep(delay_seconds)
            delay_seconds *= 2
    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini request failed without an exception")


def _parse_json_object_response(response_text: str) -> dict[str, Any]:
    if not isinstance(response_text, str) or not response_text.strip():
        raise GeminiResponseValidationError("Gemini returned an empty response.")
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise GeminiResponseValidationError("Gemini returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise GeminiResponseValidationError("Gemini JSON response must be an object.")
    return payload


def _validate_single_response_text(response_text: str) -> dict[str, Any]:
    payload = _parse_json_object_response(response_text)
    _validated_score_payload(payload)
    return payload


def _validate_batch_response_text(response_text: str) -> dict[str, Any]:
    payload = _parse_json_object_response(response_text)
    items = payload.get("items")
    if not isinstance(items, list):
        raise GeminiResponseValidationError("Gemini batch response must include an items array.")
    return payload


def score_headline_with_gemini(
    *,
    ticker: str,
    headline: str,
    project_id: str,
    region: str,
    model_name: str = GEMINI_MODEL_NAME,
) -> dict[str, Any]:
    prompt = (
        f'Analyze this financial news headline for {ticker}. '
        'Rate the likely impact on next-day stock price.\n\n'
        f'Headline: "{headline}"'
    )
    client = _gemini_client(project_id, region)
    payload = _generate_content_with_retries(
        client=client,
        model_name=model_name,
        prompt=prompt,
        response_schema=_RESPONSE_SCHEMA,
        response_validator=_validate_single_response_text,
    )
    return _normalize_score_row(payload, ticker=ticker, headline=headline)


def score_headline_batch_with_gemini(
    headline_rows: list[dict[str, str]],
    *,
    project_id: str,
    region: str,
    model_name: str = GEMINI_MODEL_NAME,
) -> list[dict[str, Any]]:
    if not headline_rows:
        return []

    prompt_lines = [
        "Analyze each financial news headline for the specified ticker.",
        "Return one JSON item per input with the same index, a sentiment_score from -1 to 1, a magnitude from 0 to 1, and a category.",
        "",
    ]
    for index, row in enumerate(headline_rows):
        prompt_lines.append(
            f"{index}. ticker={json.dumps(row['ticker'])} headline={json.dumps(row['headline'])}"
        )

    client = _gemini_client(project_id, region)
    payload = _generate_content_with_retries(
        client=client,
        model_name=model_name,
        prompt="\n".join(prompt_lines),
        response_schema=_BATCH_RESPONSE_SCHEMA,
        response_validator=_validate_batch_response_text,
    )
    raw_items = payload.get("items", [])
    normalized_by_index: dict[int, dict[str, Any]] = {}
    for item in raw_items:
        try:
            index = int(item["index"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= index < len(headline_rows):
            source_row = headline_rows[index]
            try:
                normalized_by_index[index] = _normalize_score_row(
                    item,
                    ticker=source_row["ticker"],
                    headline=source_row["headline"],
                )
            except GeminiResponseValidationError:
                continue

    if len(normalized_by_index) != len(headline_rows):
        missing_indexes = [index for index in range(len(headline_rows)) if index not in normalized_by_index]
        log.warning("Gemini batch response missing indexes: %s; falling back to single-item scoring", missing_indexes[:5])
        for index in missing_indexes:
            source_row = headline_rows[index]
            normalized_by_index[index] = score_headline_with_gemini(
                ticker=source_row["ticker"],
                headline=source_row["headline"],
                project_id=project_id,
                region=region,
                model_name=model_name,
            )

    return [normalized_by_index[index] for index in range(len(headline_rows))]


def score_headlines_with_gemini(
    headline_rows: list[dict[str, str]],
    *,
    project_id: str,
    region: str,
    model_name: str = GEMINI_MODEL_NAME,
    batch_size: int = GEMINI_REQUEST_BATCH_SIZE,
) -> list[dict[str, Any]]:
    if not headline_rows:
        return []

    scored_rows: list[dict[str, Any]] = []
    effective_batch_size = max(1, batch_size)
    for offset in range(0, len(headline_rows), effective_batch_size):
        batch = headline_rows[offset : offset + effective_batch_size]
        if len(batch) == 1:
            row = batch[0]
            scored_rows.append(
                score_headline_with_gemini(
                    ticker=row["ticker"],
                    headline=row["headline"],
                    project_id=project_id,
                    region=region,
                    model_name=model_name,
                )
            )
            continue
        scored_rows.extend(
            score_headline_batch_with_gemini(
                batch,
                project_id=project_id,
                region=region,
                model_name=model_name,
            )
        )
    return scored_rows


def _query_cached_scores(
    client: bigquery.Client,
    *,
    table_name: str,
    tickers: list[str],
    headline_hashes: list[str],
) -> pd.DataFrame:
    if not tickers or not headline_hashes:
        return pd.DataFrame(columns=["headline_hash", "ticker", "headline", "sentiment_score", "magnitude", "category", "scored_at"])

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("tickers", "STRING", tickers),
            bigquery.ArrayQueryParameter("headline_hashes", "STRING", headline_hashes),
        ]
    )
    query = f"""
        SELECT headline_hash, ticker, headline, sentiment_score, magnitude, category, scored_at
        FROM `{table_name}`
        WHERE ticker IN UNNEST(@tickers)
          AND headline_hash IN UNNEST(@headline_hashes)
    """
    return client.query(query, job_config=job_config).to_dataframe()


def _append_cached_scores(
    client: bigquery.Client,
    *,
    table_name: str,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    if "." not in table_name:
        errors = client.insert_rows_json(table_name, rows)
        if errors:
            raise RuntimeError(f"Failed to insert Gemini cache rows: {errors[:3]}")
        return

    parts = table_name.split(".")
    if len(parts) != 3:
        raise RuntimeError(f"Expected fully qualified cache table name, got: {table_name}")
    project_id, dataset_id, _table_id = parts
    stage_table_name = f"{project_id}.{dataset_id}.__gemini_cache_stage_{uuid4().hex}"
    stage_df = pd.DataFrame(rows)
    stage_df["scored_at"] = pd.to_datetime(stage_df["scored_at"], utc=True)

    try:
        client.load_table_from_dataframe(
            stage_df,
            stage_table_name,
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
        ).result()
        merge_query = f"""
            DELETE FROM `{table_name}`
            WHERE STRUCT(ticker, headline_hash) IN (
              SELECT AS STRUCT ticker, headline_hash
              FROM `{stage_table_name}`
            );

            INSERT INTO `{table_name}` (
              headline_hash,
              ticker,
              headline,
              sentiment_score,
              magnitude,
              category,
              scored_at
            )
            SELECT headline_hash, ticker, headline, sentiment_score, magnitude, category, scored_at
            FROM (
              SELECT *,
                     ROW_NUMBER() OVER (PARTITION BY ticker, headline_hash ORDER BY scored_at DESC) AS row_num
              FROM `{stage_table_name}`
            )
            WHERE row_num = 1
        """
        client.query(merge_query).result()
    finally:
        client.delete_table(stage_table_name, not_found_ok=True)


def load_or_score_headlines(
    headline_rows: Sequence[dict[str, str]],
    *,
    project_id: str,
    region: str,
    model_name: str = GEMINI_MODEL_NAME,
    cache_client: bigquery.Client | None = None,
    cache_table_name: str | None = None,
    progress_label: str = "Gemini scoring progress",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not headline_rows:
        return pd.DataFrame(columns=["headline_hash", "ticker", "headline", "sentiment_score", "magnitude", "category", "scored_at"]), []

    distinct = pd.DataFrame(list(headline_rows))[["ticker", "headline"]].drop_duplicates().copy()
    distinct["headline_hash"] = distinct["headline"].apply(headline_hash)

    if cache_client is not None and cache_table_name:
        cached = _query_cached_scores(
            cache_client,
            table_name=cache_table_name,
            tickers=sorted(distinct["ticker"].unique().tolist()),
            headline_hashes=distinct["headline_hash"].unique().tolist(),
        ).drop_duplicates(subset=["ticker", "headline_hash"], keep="last")
    else:
        cached = pd.DataFrame(columns=["headline_hash", "ticker", "headline", "sentiment_score", "magnitude", "category", "scored_at"])

    cached_keys = set(zip(cached["ticker"], cached["headline_hash"]))
    uncached_rows = [
        {"ticker": row.ticker, "headline": row.headline}
        for row in distinct.itertuples(index=False)
        if (row.ticker, row.headline_hash) not in cached_keys
    ]

    newly_scored: list[dict[str, Any]] = []
    if uncached_rows:
        total_uncached = len(uncached_rows)
        log.info("Gemini cache miss count: %s headlines", total_uncached)
        for offset in range(0, total_uncached, GEMINI_CACHE_BATCH_SIZE):
            batch = uncached_rows[offset : offset + GEMINI_CACHE_BATCH_SIZE]
            scored_batch = score_headlines_with_gemini(
                batch,
                project_id=project_id,
                region=region,
                model_name=model_name,
            )
            if scored_batch and cache_client is not None and cache_table_name:
                _append_cached_scores(cache_client, table_name=cache_table_name, rows=scored_batch)
            newly_scored.extend(scored_batch)
            log.info(
                "%s: %s/%s headlines",
                progress_label,
                min(offset + len(batch), total_uncached),
                total_uncached,
            )

    all_scores = pd.concat([cached, pd.DataFrame(newly_scored)], ignore_index=True) if newly_scored else cached.copy()
    return all_scores.drop_duplicates(subset=["ticker", "headline_hash"], keep="last"), newly_scored


def build_training_sentiment_feature_table(
    news_df: pd.DataFrame,
    market_dates_df: pd.DataFrame,
    *,
    project_id: str,
    region: str,
    cache_client: bigquery.Client,
    cache_table_name: str,
    lookback_days: int = NEWS_LOOKBACK_DAYS,
    model_name: str = GEMINI_MODEL_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = market_dates_df[["ticker", "date", "day_change_pct"]].drop_duplicates().copy()
    targets["date"] = pd.to_datetime(targets["date"]).dt.normalize()
    if targets.empty:
        return (
            pd.DataFrame(columns=["ticker", "date", *GEMINI_FEATURE_NAMES]),
            pd.DataFrame(columns=["headline_hash", "ticker", "headline", "sentiment_score", "magnitude", "category", "scored_at"]),
        )

    working = news_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    if "timestamp" in working.columns:
        working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce", utc=True)
    else:
        working["timestamp"] = pd.to_datetime(working["date"], errors="coerce", utc=True)
    working["headline"] = working["headline"].fillna("").astype(str).str.strip()
    working = working[working["headline"].ne("")].copy()
    if working.empty:
        defaults = pd.DataFrame([DEFAULT_GEMINI_FEATURES] * len(targets))
        return pd.concat([targets[["ticker", "date"]].reset_index(drop=True), defaults], axis=1), pd.DataFrame()

    all_scores, newly_scored = load_or_score_headlines(
        working[["ticker", "headline"]].to_dict("records"),
        project_id=project_id,
        region=region,
        model_name=model_name,
        cache_client=cache_client,
        cache_table_name=cache_table_name,
        progress_label="Gemini scoring progress",
    )
    if all_scores.empty:
        defaults = pd.DataFrame([DEFAULT_GEMINI_FEATURES] * len(targets))
        return pd.concat([targets[["ticker", "date"]].reset_index(drop=True), defaults], axis=1), pd.DataFrame()

    working["headline_hash"] = working["headline"].apply(headline_hash)
    scored_news = working.merge(
        all_scores[["ticker", "headline_hash", "sentiment_score", "magnitude", "category"]],
        on=["ticker", "headline_hash"],
        how="left",
    )
    scored_news["category"] = scored_news["category"].fillna("other")

    rows: list[dict[str, Any]] = []
    for ticker, group in targets.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        ticker_news = scored_news[scored_news["ticker"] == ticker].copy()
        for target_row in group.itertuples(index=False):
            current_date = pd.Timestamp(target_row.date)
            day_change_pct = float(target_row.day_change_pct)
            window_start = current_date - pd.Timedelta(days=lookback_days - 1)
            window = ticker_news[(ticker_news["date"] >= window_start) & (ticker_news["date"] <= current_date)]
            feature_row = aggregate_scored_headlines(
                window.to_dict("records"),
                day_change_pct=day_change_pct,
            )
            rows.append({"ticker": ticker, "date": current_date, **feature_row})

    return pd.DataFrame(rows), pd.DataFrame(newly_scored or [])


def aggregate_scored_headlines(
    scored_rows: Sequence[dict[str, Any]],
    *,
    day_change_pct: float | None = None,
) -> dict[str, Any]:
    if not scored_rows:
        return dict(DEFAULT_GEMINI_FEATURES)

    frame = pd.DataFrame(list(scored_rows))
    scores = frame["sentiment_score"].astype(float)
    magnitude = frame["magnitude"].astype(float)
    categories = frame["category"].fillna("other").astype(str).str.lower()

    sentiment_avg = float(scores.mean())
    market_sign = 0 if day_change_pct is None or pd.isna(day_change_pct) else (1 if float(day_change_pct) > 0 else -1 if float(day_change_pct) < 0 else 0)
    sentiment_sign = 0 if sentiment_avg == 0 else (1 if sentiment_avg > 0 else -1)

    return {
        "gemini_sentiment_avg": round(sentiment_avg, 6),
        "gemini_sentiment_max": round(float(scores.max()), 6),
        "gemini_sentiment_min": round(float(scores.min()), 6),
        "gemini_magnitude_avg": round(float(magnitude.mean()), 6),
        "headline_count": int(len(frame)),
        "has_earnings_news": int(categories.eq("earnings").any()),
        "has_analyst_news": int(categories.eq("analyst").any()),
        "sentiment_dispersion": round(float(scores.std(ddof=0)) if len(frame) > 1 else 0.0, 6),
        "sentiment_market_alignment": int(market_sign != 0 and sentiment_sign == market_sign),
    }


def score_live_news_items(
    *,
    ticker: str,
    news_items: Sequence[dict[str, Any]],
    project_id: str,
    region: str,
    model_name: str = GEMINI_MODEL_NAME,
    cache_client: bigquery.Client | None = None,
    cache_table_name: str | None = None,
) -> dict[str, Any]:
    normalized_items = [item for item in news_items if str(item.get("headline") or "").strip()]
    if not normalized_items:
        return {
            **dict(DEFAULT_GEMINI_FEATURES),
            LIVE_NEWS_SENTIMENT_ITEMS_KEY: [],
        }

    headline_rows = [{"ticker": ticker, "headline": str(item["headline"]).strip()} for item in normalized_items]
    all_scores, _newly_scored = load_or_score_headlines(
        headline_rows,
        project_id=project_id,
        region=region,
        model_name=model_name,
        cache_client=cache_client,
        cache_table_name=cache_table_name,
        progress_label="Gemini live scoring progress",
    )
    scores_by_key = {
        (str(row.ticker), str(row.headline_hash)): {
            "headline_hash": row.headline_hash,
            "ticker": row.ticker,
            "headline": row.headline,
            "sentiment_score": row.sentiment_score,
            "magnitude": row.magnitude,
            "category": row.category,
            "scored_at": row.scored_at,
        }
        for row in all_scores.itertuples(index=False)
    }
    scored_rows = []
    enriched_news_items: list[dict[str, Any]] = []
    for item in normalized_items:
        headline = str(item["headline"]).strip()
        score_row = scores_by_key.get((ticker, headline_hash(headline)))
        if score_row:
            scored_rows.append(score_row)
        sentiment_score = None if not score_row else float(score_row["sentiment_score"])
        enriched_news_items.append(
            {
                **dict(item),
                "sentiment_score": sentiment_score,
                "magnitude": None if not score_row else float(score_row["magnitude"]),
                "category": None if not score_row else str(score_row["category"]),
                "scored_at": None if not score_row else score_row["scored_at"],
                "sentiment_label": _live_sentiment_label(sentiment_score),
            }
        )

    return {
        **aggregate_scored_headlines(scored_rows),
        LIVE_NEWS_SENTIMENT_ITEMS_KEY: enriched_news_items,
    }
