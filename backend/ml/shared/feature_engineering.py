"""Shared feature engineering helpers for training, batch inference, and serving."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger(__name__)

NEWS_LOOKBACK_DAYS = 7
TFIDF_MAX_FEATURES = 16

MARKET_FEATURE_NAMES = [
    "current_price",
    "previous_close",
    "day_open",
    "day_high",
    "day_low",
    "day_change",
    "day_change_pct",
    "gap_pct",
    "intraday_range_pct",
    "close_vs_open_pct",
    "day_range_position",
    "volume",
    "avg_volume_10d",
    "avg_volume_3m",
    "volume_vs_avg_10d",
    "volume_vs_avg_3m",
    "fifty_day_avg",
    "two_hundred_day_avg",
    "price_vs_50d_avg",
    "price_vs_200d_avg",
    "fifty_two_week_high",
    "fifty_two_week_low",
    "price_vs_52w_high",
    "price_vs_52w_low",
    "position_52w_range",
]

MARKET_CONTEXT_FEATURE_NAMES = [
    "spy_change_pct",
    "spy_gap_pct",
    "vix_level",
    "vix_change_pct",
]

SENTIMENT_FEATURE_NAMES = [
    "sentiment_score",
    "headline_count",
    "has_news",
    "avg_news_age_hours",
    "news_source_count",
]

TFIDF_FEATURE_NAMES = [f"tfidf_{idx}" for idx in range(TFIDF_MAX_FEATURES)]

BASE_FEATURE_NAMES = MARKET_FEATURE_NAMES + MARKET_CONTEXT_FEATURE_NAMES + SENTIMENT_FEATURE_NAMES
MODEL_FEATURE_NAMES = BASE_FEATURE_NAMES + TFIDF_FEATURE_NAMES
EXPECTED_FEATURE_COUNT = len(MODEL_FEATURE_NAMES)

DEFAULT_SENTIMENT_FEATURES = {
    "headlines_text": "",
    "sentiment_score": 0.0,
    "headline_count": 0,
    "has_news": 0,
    "avg_news_age_hours": float(NEWS_LOOKBACK_DAYS * 24),
    "news_source_count": 0,
}

DEFAULT_MARKET_CONTEXT_FEATURES = {
    "spy_change_pct": 0.0,
    "spy_gap_pct": 0.0,
    "vix_level": 20.0,
    "vix_change_pct": 0.0,
}

FEATURE_GROUPS = {
    "market": MARKET_FEATURE_NAMES,
    "context": MARKET_CONTEXT_FEATURE_NAMES,
    "sentiment": SENTIMENT_FEATURE_NAMES,
    "text_vector": TFIDF_FEATURE_NAMES,
}

FEATURE_GROUP_LABELS = {
    "market": "Live Market Snapshot",
    "context": "Market Context",
    "sentiment": "News Sentiment",
    "text_vector": "Headline Vector",
}

FEATURE_LABELS = {
    "current_price": "Current Price",
    "previous_close": "Previous Close",
    "day_open": "Open",
    "day_high": "Day High",
    "day_low": "Day Low",
    "day_change": "Day Change",
    "day_change_pct": "Day Change %",
    "gap_pct": "Open Gap %",
    "intraday_range_pct": "Intraday Range %",
    "close_vs_open_pct": "Close vs Open %",
    "day_range_position": "Day Range Position",
    "volume": "Volume",
    "avg_volume_10d": "Avg Volume 10D",
    "avg_volume_3m": "Avg Volume 3M",
    "volume_vs_avg_10d": "Volume vs Avg 10D",
    "volume_vs_avg_3m": "Volume vs Avg 3M",
    "fifty_day_avg": "50D Average",
    "two_hundred_day_avg": "200D Average",
    "price_vs_50d_avg": "Price vs 50D Avg",
    "price_vs_200d_avg": "Price vs 200D Avg",
    "fifty_two_week_high": "52W High",
    "fifty_two_week_low": "52W Low",
    "price_vs_52w_high": "Price vs 52W High",
    "price_vs_52w_low": "Price vs 52W Low",
    "position_52w_range": "52W Range Position",
    "spy_change_pct": "SPY Change %",
    "spy_gap_pct": "SPY Gap %",
    "vix_level": "VIX Level",
    "vix_change_pct": "VIX Change %",
    "sentiment_score": "Sentiment Score",
    "headline_count": "Headline Count",
    "has_news": "Has News",
    "avg_news_age_hours": "Avg News Age (Hours)",
    "news_source_count": "News Source Count",
}


def _replace_zero(values: pd.Series | float | int | None):
    if isinstance(values, pd.Series):
        return values.replace(0, np.nan)
    if values in (0, 0.0):
        return np.nan
    return values


def _as_python(value):
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _safe_float(value: object, default: float | None = None) -> float | None:
    if value is None or pd.isna(value):
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric):
        return default
    return numeric


def _safe_divide(numerator, denominator):
    if isinstance(denominator, pd.Series):
        denominator = _replace_zero(denominator)
    elif denominator in (0, 0.0):
        denominator = np.nan
    return numerator / denominator


def normalize_headlines(headlines: Sequence[str] | str | None) -> list[str]:
    if headlines is None:
        return []
    if isinstance(headlines, str):
        return [headlines.strip()] if headlines.strip() else []
    normalized: list[str] = []
    for item in headlines:
        if item is None:
            continue
        text = item.strip() if isinstance(item, str) else str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def extract_news_title(item: Mapping[str, object] | None) -> str | None:
    payload = dict(item or {})

    title = payload.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()

    headline = payload.get("headline")
    if isinstance(headline, str) and headline.strip():
        return headline.strip()

    content = payload.get("content")
    if isinstance(content, Mapping):
        nested_title = content.get("title")
        if isinstance(nested_title, str) and nested_title.strip():
            return nested_title.strip()

    return None


@lru_cache(maxsize=1)
def get_sentiment_analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def compute_headline_features(
    headlines: Sequence[str] | str | None,
    *,
    timestamps: Sequence[object] | None = None,
    sources: Sequence[str] | None = None,
    reference_time: object | None = None,
) -> dict:
    normalized = normalize_headlines(headlines)
    if not normalized:
        return DEFAULT_SENTIMENT_FEATURES.copy()

    analyzer = get_sentiment_analyzer()
    scores = [analyzer.polarity_scores(headline)["compound"] for headline in normalized]

    source_count = len(
        {
            str(source).strip()
            for source in (sources or [])
            if isinstance(source, str) and source.strip()
        }
    )

    avg_age_hours = DEFAULT_SENTIMENT_FEATURES["avg_news_age_hours"]
    if timestamps:
        normalized_timestamps = pd.to_datetime(pd.Series(list(timestamps)), errors="coerce", utc=True)
        normalized_timestamps = normalized_timestamps.dropna()
        if not normalized_timestamps.empty:
            anchor = pd.Timestamp(reference_time or pd.Timestamp.utcnow())
            if anchor.tzinfo is None:
                anchor = anchor.tz_localize("UTC")
            else:
                anchor = anchor.tz_convert("UTC")
            ages = (anchor - normalized_timestamps).dt.total_seconds().clip(lower=0) / 3600.0
            if not ages.empty:
                avg_age_hours = float(ages.mean())

    return {
        "headlines_text": " ".join(normalized),
        "sentiment_score": round(float(np.mean(scores)), 6),
        "headline_count": len(normalized),
        "has_news": 1,
        "avg_news_age_hours": round(float(avg_age_hours), 6),
        "news_source_count": source_count,
    }


def _tfidf_frame(matrix, index: pd.Index) -> pd.DataFrame:
    frame = pd.DataFrame(
        matrix.toarray(),
        columns=[f"tfidf_{idx}" for idx in range(matrix.shape[1])],
        index=index,
    )
    return frame.reindex(columns=TFIDF_FEATURE_NAMES, fill_value=0.0)


def fit_tfidf_safely(texts: pd.Series) -> tuple[TfidfVectorizer, pd.DataFrame]:
    cleaned = texts.fillna("").astype(str)
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))

    if cleaned.str.strip().eq("").all():
        vectorizer.fit(["__no_headlines__"])
        matrix = vectorizer.transform([""] * len(cleaned))
        return vectorizer, _tfidf_frame(matrix, cleaned.index)

    matrix = vectorizer.fit_transform(cleaned)
    return vectorizer, _tfidf_frame(matrix, cleaned.index)


def transform_tfidf_frame(vectorizer: TfidfVectorizer, texts: pd.Series) -> pd.DataFrame:
    cleaned = texts.fillna("").astype(str)
    matrix = vectorizer.transform(cleaned)
    return _tfidf_frame(matrix, cleaned.index)


def prepare_price_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    working = df.copy()
    if isinstance(working.columns, pd.MultiIndex):
        working.columns = [column[0] for column in working.columns]

    if "date" not in working.columns:
        working = working.reset_index()

    working.columns = [str(column).lower() for column in working.columns]
    date_column = "datetime" if "datetime" in working.columns else "date"
    if date_column not in working.columns:
        date_column = working.columns[0]

    rename_map = {date_column: "date"}
    if "adj close" in working.columns and "close" not in working.columns:
        rename_map["adj close"] = "close"
    working = working.rename(columns=rename_map)

    required = [column for column in ["date", "open", "high", "low", "close", "volume"] if column in working.columns]
    working = working[required].copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    if getattr(working["date"].dt, "tz", None) is not None:
        working["date"] = working["date"].dt.tz_convert(None)
    working["date"] = working["date"].astype("datetime64[ns]")
    return working.dropna(subset=["date"]).sort_values("date")


def compute_market_features_from_history(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    volume = df["Volume"].astype(float)
    previous_close = close.shift(1)

    features = pd.DataFrame(index=df.index)
    features["current_price"] = close
    features["previous_close"] = previous_close
    features["day_open"] = open_
    features["day_high"] = high
    features["day_low"] = low
    features["day_change"] = close - previous_close
    features["day_change_pct"] = _safe_divide(close, previous_close) - 1
    features["gap_pct"] = _safe_divide(open_, previous_close) - 1
    features["intraday_range_pct"] = _safe_divide(high - low, previous_close)
    features["close_vs_open_pct"] = _safe_divide(close, open_) - 1
    features["day_range_position"] = _safe_divide(close - low, high - low)
    features["volume"] = volume

    avg_volume_10d = volume.rolling(10, min_periods=3).mean()
    avg_volume_3m = volume.rolling(63, min_periods=10).mean()
    features["avg_volume_10d"] = avg_volume_10d
    features["avg_volume_3m"] = avg_volume_3m
    features["volume_vs_avg_10d"] = _safe_divide(volume, avg_volume_10d)
    features["volume_vs_avg_3m"] = _safe_divide(volume, avg_volume_3m)

    fifty_day_avg = close.rolling(50, min_periods=10).mean()
    two_hundred_day_avg = close.rolling(200, min_periods=30).mean()
    features["fifty_day_avg"] = fifty_day_avg
    features["two_hundred_day_avg"] = two_hundred_day_avg
    features["price_vs_50d_avg"] = _safe_divide(close, fifty_day_avg) - 1
    features["price_vs_200d_avg"] = _safe_divide(close, two_hundred_day_avg) - 1

    fifty_two_week_high = high.rolling(252, min_periods=30).max()
    fifty_two_week_low = low.rolling(252, min_periods=30).min()
    features["fifty_two_week_high"] = fifty_two_week_high
    features["fifty_two_week_low"] = fifty_two_week_low
    features["price_vs_52w_high"] = _safe_divide(close, fifty_two_week_high) - 1
    features["price_vs_52w_low"] = _safe_divide(close, fifty_two_week_low) - 1
    features["position_52w_range"] = _safe_divide(close - fifty_two_week_low, fifty_two_week_high - fifty_two_week_low)

    return features[MARKET_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)


def build_market_context_history(spy_history: pd.DataFrame, vix_history: pd.DataFrame) -> pd.DataFrame:
    spy = prepare_price_history_frame(spy_history)
    vix = prepare_price_history_frame(vix_history)

    if spy.empty and vix.empty:
        return pd.DataFrame(columns=["date", *MARKET_CONTEXT_FEATURE_NAMES])

    index = pd.Index([], dtype="datetime64[ns]")
    if not spy.empty:
        index = index.union(pd.Index(spy["date"]))
    if not vix.empty:
        index = index.union(pd.Index(vix["date"]))

    context = pd.DataFrame(index=index.sort_values())
    if not spy.empty:
        spy_frame = spy.set_index("date").sort_index()
        spy_close = spy_frame["close"].astype(float)
        spy_open = spy_frame["open"].astype(float)
        previous_close = spy_close.shift(1)
        context["spy_change_pct"] = _safe_divide(spy_close, previous_close) - 1
        context["spy_gap_pct"] = _safe_divide(spy_open, previous_close) - 1

    if not vix.empty:
        vix_frame = vix.set_index("date").sort_index()
        vix_close = vix_frame["close"].astype(float)
        previous_close = vix_close.shift(1)
        context["vix_level"] = vix_close
        context["vix_change_pct"] = _safe_divide(vix_close, previous_close) - 1

    context = context.sort_index().ffill().reset_index().rename(columns={"index": "date"})
    context["date"] = pd.to_datetime(context["date"], errors="coerce").astype("datetime64[ns]")
    return context


def build_market_context_feature_frame(
    dates: pd.Series | Sequence[pd.Timestamp | str],
    context_history: pd.DataFrame,
) -> pd.DataFrame:
    date_series = pd.to_datetime(pd.Series(dates), errors="coerce")
    if getattr(date_series.dt, "tz", None) is not None:
        date_series = date_series.dt.tz_convert(None)
    date_series = date_series.dt.normalize().astype("datetime64[ns]")

    frame = pd.DataFrame(
        {
            "date": date_series,
            "_row_id": np.arange(len(date_series)),
        }
    )
    for column, default_value in DEFAULT_MARKET_CONTEXT_FEATURES.items():
        frame[column] = default_value

    if not context_history.empty:
        context = context_history.copy()
        context["date"] = pd.to_datetime(context["date"], errors="coerce")
        if getattr(context["date"].dt, "tz", None) is not None:
            context["date"] = context["date"].dt.tz_convert(None)
        context["date"] = context["date"].dt.normalize().astype("datetime64[ns]")
        for column, default_value in DEFAULT_MARKET_CONTEXT_FEATURES.items():
            if column not in context.columns:
                context[column] = default_value

        valid = frame[frame["date"].notna()].sort_values("date")
        if not valid.empty:
            merged = pd.merge_asof(
                valid,
                context.sort_values("date"),
                on="date",
                direction="backward",
            )
            for column, default_value in DEFAULT_MARKET_CONTEXT_FEATURES.items():
                merged_values = merged.get(column, pd.Series(default_value, index=merged.index))
                frame.loc[merged["_row_id"], column] = merged_values.fillna(default_value).to_numpy()

    return frame.sort_values("_row_id")[MARKET_CONTEXT_FEATURE_NAMES].reset_index(drop=True)


def build_training_news_feature_table(
    news_df: pd.DataFrame,
    market_dates_df: pd.DataFrame,
    *,
    lookback_days: int = NEWS_LOOKBACK_DAYS,
) -> pd.DataFrame:
    targets = market_dates_df[["ticker", "date"]].drop_duplicates().copy()
    targets["date"] = pd.to_datetime(targets["date"]).dt.normalize()

    if targets.empty:
        return pd.DataFrame(columns=["ticker", "date", *DEFAULT_SENTIMENT_FEATURES.keys()])

    if news_df.empty:
        defaults = pd.DataFrame([DEFAULT_SENTIMENT_FEATURES] * len(targets))
        return pd.concat([targets.reset_index(drop=True), defaults], axis=1)

    working = news_df.copy()
    working["date"] = pd.to_datetime(working["date"]).dt.normalize()
    if "timestamp" in working.columns:
        working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce", utc=True)
    else:
        working["timestamp"] = pd.to_datetime(working["date"], errors="coerce", utc=True) + pd.Timedelta(hours=16)
    working["headline"] = working["headline"].fillna("").astype(str)
    working["source"] = working.get("source", "").fillna("").astype(str)
    working = working[working["headline"].str.strip().ne("")].copy()

    news_lookup = {
        ticker: group.sort_values("timestamp").reset_index(drop=True)
        for ticker, group in working.groupby("ticker", sort=False)
    }

    rows: list[dict] = []
    for ticker, group in targets.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        ticker_news = news_lookup.get(ticker, pd.DataFrame(columns=working.columns))
        for current_date in group["date"]:
            reference_time = pd.Timestamp(current_date).tz_localize("UTC") + pd.Timedelta(hours=16)
            window_start = pd.Timestamp(current_date) - pd.Timedelta(days=lookback_days - 1)
            if ticker_news.empty:
                features = DEFAULT_SENTIMENT_FEATURES.copy()
            else:
                window = ticker_news[
                    (ticker_news["date"] >= window_start)
                    & (ticker_news["date"] <= current_date)
                ]
                features = compute_headline_features(
                    window["headline"].tolist(),
                    timestamps=window["timestamp"].tolist(),
                    sources=window["source"].tolist(),
                    reference_time=reference_time,
                )
            rows.append({"ticker": ticker, "date": current_date, **features})

    return pd.DataFrame(rows)


def _trace_value(trace: Mapping[str, object] | None, default: float = 0.0) -> float:
    if not isinstance(trace, Mapping):
        return default
    return _safe_float(trace.get("value"), default) or default


def _raw_feature_detail(
    name: str,
    value: float,
    group: str,
    trace: Mapping[str, object] | None,
) -> dict:
    trace = dict(trace or {})
    provider = str(trace.get("provider") or "default")
    return {
        "name": name,
        "label": FEATURE_LABELS.get(name, name),
        "value": _as_python(value),
        "group": group,
        "source": provider,
        "providers": [provider],
        "raw_fields": [str(trace.get("raw_field") or name)],
        "fallback_used": bool(trace.get("fallback_used", False)),
    }


def _derived_feature_detail(
    name: str,
    value: float | int,
    group: str,
    traces: Sequence[Mapping[str, object] | None],
) -> dict:
    providers = [str(trace.get("provider")) for trace in traces if isinstance(trace, Mapping) and trace.get("provider")]
    raw_fields = [str(trace.get("raw_field")) for trace in traces if isinstance(trace, Mapping) and trace.get("raw_field")]
    return {
        "name": name,
        "label": FEATURE_LABELS.get(name, name),
        "value": _as_python(value),
        "group": group,
        "source": "derived",
        "providers": sorted(set(providers)) or ["derived"],
        "raw_fields": raw_fields or [name],
        "fallback_used": any(bool(trace.get("fallback_used", False)) for trace in traces if isinstance(trace, Mapping)),
    }


def normalize_live_news_items(news_items: Sequence[Mapping[str, object]] | None) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    normalized: list[dict] = []
    for item in news_items or []:
        headline = extract_news_title(item)
        if not headline:
            continue

        url = str(item.get("url") or "").strip()
        key = (headline.lower(), url)
        if key in seen:
            continue
        seen.add(key)

        published_at = pd.to_datetime(
            item.get("published_at") or item.get("datetime") or item.get("providerPublishTime"),
            errors="coerce",
            utc=True,
        )
        normalized.append(
            {
                "headline": headline,
                "source": str(item.get("source") or item.get("provider_name") or item.get("provider") or "").strip(),
                "provider": str(item.get("provider") or "").strip() or "unknown",
                "url": url,
                "published_at": None if pd.isna(published_at) else published_at.isoformat(),
            }
        )

    normalized.sort(key=lambda row: row["published_at"] or "", reverse=True)
    return normalized


def build_live_feature_bundle(
    live_snapshot: Mapping[str, object],
    tfidf_vectorizer,
    feature_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    quote = dict(live_snapshot.get("quote") or {})
    context_payload = dict(live_snapshot.get("market_context") or {})
    company_name = live_snapshot.get("company_name")
    company_name = company_name.strip() if isinstance(company_name, str) else None
    quote_currency = live_snapshot.get("quote_currency")
    quote_currency = quote_currency.strip().upper() if isinstance(quote_currency, str) else None
    as_of = pd.Timestamp(live_snapshot.get("as_of") or pd.Timestamp.utcnow())
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")

    market_values: dict[str, float] = {}
    market_details: list[dict] = []
    raw_market_traces = {
        "current_price": quote.get("current_price"),
        "previous_close": quote.get("previous_close"),
        "day_open": quote.get("day_open"),
        "day_high": quote.get("day_high"),
        "day_low": quote.get("day_low"),
        "volume": quote.get("volume"),
        "avg_volume_10d": quote.get("avg_volume_10d"),
        "avg_volume_3m": quote.get("avg_volume_3m"),
        "fifty_day_avg": quote.get("fifty_day_avg"),
        "two_hundred_day_avg": quote.get("two_hundred_day_avg"),
        "fifty_two_week_high": quote.get("fifty_two_week_high"),
        "fifty_two_week_low": quote.get("fifty_two_week_low"),
    }

    current_price = _trace_value(raw_market_traces["current_price"])
    previous_close = _trace_value(raw_market_traces["previous_close"])
    day_open = _trace_value(raw_market_traces["day_open"])
    day_high = _trace_value(raw_market_traces["day_high"])
    day_low = _trace_value(raw_market_traces["day_low"])
    volume = _trace_value(raw_market_traces["volume"])
    avg_volume_10d = _trace_value(raw_market_traces["avg_volume_10d"])
    avg_volume_3m = _trace_value(raw_market_traces["avg_volume_3m"])
    fifty_day_avg = _trace_value(raw_market_traces["fifty_day_avg"])
    two_hundred_day_avg = _trace_value(raw_market_traces["two_hundred_day_avg"])
    fifty_two_week_high = _trace_value(raw_market_traces["fifty_two_week_high"])
    fifty_two_week_low = _trace_value(raw_market_traces["fifty_two_week_low"])

    market_values.update(
        {
            "current_price": current_price,
            "previous_close": previous_close,
            "day_open": day_open,
            "day_high": day_high,
            "day_low": day_low,
            "day_change": current_price - previous_close,
            "day_change_pct": 0.0 if previous_close == 0 else current_price / previous_close - 1,
            "gap_pct": 0.0 if previous_close == 0 else day_open / previous_close - 1,
            "intraday_range_pct": 0.0 if previous_close == 0 else (day_high - day_low) / previous_close,
            "close_vs_open_pct": 0.0 if day_open == 0 else current_price / day_open - 1,
            "day_range_position": 0.0 if day_high == day_low else (current_price - day_low) / (day_high - day_low),
            "volume": volume,
            "avg_volume_10d": avg_volume_10d,
            "avg_volume_3m": avg_volume_3m,
            "volume_vs_avg_10d": 0.0 if avg_volume_10d == 0 else volume / avg_volume_10d,
            "volume_vs_avg_3m": 0.0 if avg_volume_3m == 0 else volume / avg_volume_3m,
            "fifty_day_avg": fifty_day_avg,
            "two_hundred_day_avg": two_hundred_day_avg,
            "price_vs_50d_avg": 0.0 if fifty_day_avg == 0 else current_price / fifty_day_avg - 1,
            "price_vs_200d_avg": 0.0 if two_hundred_day_avg == 0 else current_price / two_hundred_day_avg - 1,
            "fifty_two_week_high": fifty_two_week_high,
            "fifty_two_week_low": fifty_two_week_low,
            "price_vs_52w_high": 0.0 if fifty_two_week_high == 0 else current_price / fifty_two_week_high - 1,
            "price_vs_52w_low": 0.0 if fifty_two_week_low == 0 else current_price / fifty_two_week_low - 1,
            "position_52w_range": (
                0.0
                if fifty_two_week_high == fifty_two_week_low
                else (current_price - fifty_two_week_low) / (fifty_two_week_high - fifty_two_week_low)
            ),
        }
    )

    for name in [
        "current_price",
        "previous_close",
        "day_open",
        "day_high",
        "day_low",
        "volume",
        "avg_volume_10d",
        "avg_volume_3m",
        "fifty_day_avg",
        "two_hundred_day_avg",
        "fifty_two_week_high",
        "fifty_two_week_low",
    ]:
        market_details.append(_raw_feature_detail(name, market_values[name], "market", raw_market_traces[name]))

    market_details.extend(
        [
            _derived_feature_detail(
                "day_change",
                market_values["day_change"],
                "market",
                [raw_market_traces["current_price"], raw_market_traces["previous_close"]],
            ),
            _derived_feature_detail(
                "day_change_pct",
                market_values["day_change_pct"],
                "market",
                [raw_market_traces["current_price"], raw_market_traces["previous_close"]],
            ),
            _derived_feature_detail(
                "gap_pct",
                market_values["gap_pct"],
                "market",
                [raw_market_traces["day_open"], raw_market_traces["previous_close"]],
            ),
            _derived_feature_detail(
                "intraday_range_pct",
                market_values["intraday_range_pct"],
                "market",
                [raw_market_traces["day_high"], raw_market_traces["day_low"], raw_market_traces["previous_close"]],
            ),
            _derived_feature_detail(
                "close_vs_open_pct",
                market_values["close_vs_open_pct"],
                "market",
                [raw_market_traces["current_price"], raw_market_traces["day_open"]],
            ),
            _derived_feature_detail(
                "day_range_position",
                market_values["day_range_position"],
                "market",
                [raw_market_traces["current_price"], raw_market_traces["day_high"], raw_market_traces["day_low"]],
            ),
            _derived_feature_detail(
                "volume_vs_avg_10d",
                market_values["volume_vs_avg_10d"],
                "market",
                [raw_market_traces["volume"], raw_market_traces["avg_volume_10d"]],
            ),
            _derived_feature_detail(
                "volume_vs_avg_3m",
                market_values["volume_vs_avg_3m"],
                "market",
                [raw_market_traces["volume"], raw_market_traces["avg_volume_3m"]],
            ),
            _derived_feature_detail(
                "price_vs_50d_avg",
                market_values["price_vs_50d_avg"],
                "market",
                [raw_market_traces["current_price"], raw_market_traces["fifty_day_avg"]],
            ),
            _derived_feature_detail(
                "price_vs_200d_avg",
                market_values["price_vs_200d_avg"],
                "market",
                [raw_market_traces["current_price"], raw_market_traces["two_hundred_day_avg"]],
            ),
            _derived_feature_detail(
                "price_vs_52w_high",
                market_values["price_vs_52w_high"],
                "market",
                [raw_market_traces["current_price"], raw_market_traces["fifty_two_week_high"]],
            ),
            _derived_feature_detail(
                "price_vs_52w_low",
                market_values["price_vs_52w_low"],
                "market",
                [raw_market_traces["current_price"], raw_market_traces["fifty_two_week_low"]],
            ),
            _derived_feature_detail(
                "position_52w_range",
                market_values["position_52w_range"],
                "market",
                [
                    raw_market_traces["current_price"],
                    raw_market_traces["fifty_two_week_high"],
                    raw_market_traces["fifty_two_week_low"],
                ],
            ),
        ]
    )

    context_values: dict[str, float] = {}
    context_details: list[dict] = []

    spy = dict(context_payload.get("spy") or {})
    vix = dict(context_payload.get("vix") or {})
    spy_current = _trace_value(spy.get("current_price"))
    spy_previous_close = _trace_value(spy.get("previous_close"))
    spy_open = _trace_value(spy.get("day_open"))
    vix_current = _trace_value(vix.get("current_price"), DEFAULT_MARKET_CONTEXT_FEATURES["vix_level"])
    vix_previous_close = _trace_value(vix.get("previous_close"))

    context_values.update(
        {
            "spy_change_pct": 0.0 if spy_previous_close == 0 else spy_current / spy_previous_close - 1,
            "spy_gap_pct": 0.0 if spy_previous_close == 0 else spy_open / spy_previous_close - 1,
            "vix_level": vix_current or DEFAULT_MARKET_CONTEXT_FEATURES["vix_level"],
            "vix_change_pct": 0.0 if vix_previous_close == 0 else vix_current / vix_previous_close - 1,
        }
    )

    context_details.extend(
        [
            _derived_feature_detail(
                "spy_change_pct",
                context_values["spy_change_pct"],
                "context",
                [spy.get("current_price"), spy.get("previous_close")],
            ),
            _derived_feature_detail(
                "spy_gap_pct",
                context_values["spy_gap_pct"],
                "context",
                [spy.get("day_open"), spy.get("previous_close")],
            ),
            _raw_feature_detail("vix_level", context_values["vix_level"], "context", vix.get("current_price")),
            _derived_feature_detail(
                "vix_change_pct",
                context_values["vix_change_pct"],
                "context",
                [vix.get("current_price"), vix.get("previous_close")],
            ),
        ]
    )

    normalized_news = normalize_live_news_items(live_snapshot.get("news"))
    sentiment_features = compute_headline_features(
        [item["headline"] for item in normalized_news],
        timestamps=[item["published_at"] for item in normalized_news],
        sources=[item["source"] or item["provider"] for item in normalized_news],
        reference_time=as_of,
    )
    sentiment_details = [
        _derived_feature_detail(
            name,
            sentiment_features[name],
            "sentiment",
            [{"provider": item["provider"], "raw_field": item["source"] or item["provider"]} for item in normalized_news],
        )
        for name in SENTIMENT_FEATURE_NAMES
    ]

    tfidf_df = transform_tfidf_frame(
        tfidf_vectorizer,
        pd.Series([sentiment_features["headlines_text"]]),
    ).reset_index(drop=True)
    tfidf_values = tfidf_df.iloc[0].to_dict()
    tfidf_details = [
        _derived_feature_detail(
            name,
            tfidf_values[name],
            "text_vector",
            [{"provider": item["provider"], "raw_field": "news.headline"} for item in normalized_news],
        )
        for name in TFIDF_FEATURE_NAMES
    ]

    combined = {
        **market_values,
        **context_values,
        **{name: sentiment_features[name] for name in SENTIMENT_FEATURE_NAMES},
        **tfidf_values,
    }
    feature_df = align_model_features(pd.DataFrame([combined]), feature_names)

    grouped_details = defaultdict(list)
    for detail in market_details + context_details + sentiment_details + tfidf_details:
        grouped_details[detail["group"]].append(detail)

    feature_snapshot = {
        "as_of": as_of.isoformat(),
        "company_name": company_name,
        "quote_currency": quote_currency,
        "feature_count": len(combined),
        "headlines_text": sentiment_features["headlines_text"],
        "groups": [
            {
                "key": group_key,
                "label": FEATURE_GROUP_LABELS[group_key],
                "features": grouped_details.get(group_key, []),
            }
            for group_key in FEATURE_GROUPS
        ],
        "news_items": normalized_news,
        "provider_summary": {
            "finnhub": sum(1 for detail in market_details + context_details if "finnhub" in detail["providers"]),
            "yahoo": sum(1 for detail in market_details + context_details if "yahoo" in detail["providers"]),
            "derived": sum(1 for detail in sentiment_details + tfidf_details),
            "fallback_fields": sum(
                1
                for detail in market_details + context_details + sentiment_details + tfidf_details
                if detail["fallback_used"]
            ),
        },
    }
    return feature_df, feature_snapshot


def align_model_features(
    feature_df: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    expected = list(feature_names or MODEL_FEATURE_NAMES)

    missing_base = [column for column in BASE_FEATURE_NAMES if column not in feature_df.columns]
    if missing_base:
        raise AssertionError(f"Missing required non-TFIDF features: {missing_base}")

    aligned = feature_df.reindex(columns=expected, fill_value=0.0).copy()
    aligned = aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if len(expected) != EXPECTED_FEATURE_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_FEATURE_COUNT} model features, received {len(expected)}."
        )
    if aligned.shape[1] != EXPECTED_FEATURE_COUNT:
        raise AssertionError(
            f"Feature frame has {aligned.shape[1]} columns, expected {EXPECTED_FEATURE_COUNT}."
        )

    return aligned
