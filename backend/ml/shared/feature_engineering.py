"""Shared feature engineering helpers for training, batch inference, and serving."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

NEWS_LOOKBACK_DAYS = 7
LABEL_NOISE_THRESHOLD = 0.003
CANDIDATE_FEATURE_VERSION = "2026-03-14-fast-accuracy-pass-v1"
TEMPORARY_SELECTED_FEATURE_COUNT = 40
TEMPORARY_FEATURE_SELECTION_SOURCE = "temporary_seeded_random_with_all_gemini"
TEMPORARY_FEATURE_SELECTION_MODE = "seeded_random_subset"
LIVE_NEWS_PROVIDERS = {"finnhub", "yahoo"}
LIVE_NEWS_SENTIMENT_ITEMS_KEY = "_news_item_sentiments"

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

TEMPORAL_FEATURE_NAMES = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "momentum_streak",
    "volatility_regime",
    "volume_trend_5d",
    "price_acceleration",
    "relative_strength_spy",
]

GEMINI_FEATURE_NAMES = [
    "gemini_sentiment_avg",
    "gemini_sentiment_max",
    "gemini_sentiment_min",
    "gemini_magnitude_avg",
    "headline_count",
    "has_earnings_news",
    "has_analyst_news",
    "sentiment_dispersion",
    "sentiment_market_alignment",
]

MARKET_CONTEXT_FEATURE_NAMES = [
    "spy_change_pct",
    "spy_gap_pct",
    "vix_level",
    "vix_change_pct",
]

MACRO_FEATURE_NAMES = [
    "yield_curve_spread",
    "fed_funds_rate",
    "fed_funds_change_3m",
    "credit_spread",
    "dxy_change_5d",
    "universe_breadth_50d",
]

CANDIDATE_FEATURE_NAMES = (
    MARKET_FEATURE_NAMES
    + TEMPORAL_FEATURE_NAMES
    + GEMINI_FEATURE_NAMES
    + MARKET_CONTEXT_FEATURE_NAMES
    + MACRO_FEATURE_NAMES
)

FEATURE_GROUPS = {
    "market": MARKET_FEATURE_NAMES,
    "temporal": TEMPORAL_FEATURE_NAMES,
    "sentiment": GEMINI_FEATURE_NAMES,
    "context": MARKET_CONTEXT_FEATURE_NAMES,
    "macro": MACRO_FEATURE_NAMES,
}

FEATURE_GROUP_LABELS = {
    "market": "Live Market Snapshot",
    "temporal": "Temporal / Momentum",
    "sentiment": "Gemini Sentiment",
    "context": "Market Context",
    "macro": "Macro / Regime",
}

DEFAULT_MARKET_FEATURES = {feature_name: 0.0 for feature_name in MARKET_FEATURE_NAMES}
DEFAULT_TEMPORAL_FEATURES = {feature_name: 0.0 for feature_name in TEMPORAL_FEATURE_NAMES}
DEFAULT_GEMINI_FEATURES = {
    "gemini_sentiment_avg": 0.0,
    "gemini_sentiment_max": 0.0,
    "gemini_sentiment_min": 0.0,
    "gemini_magnitude_avg": 0.0,
    "headline_count": 0,
    "has_earnings_news": 0,
    "has_analyst_news": 0,
    "sentiment_dispersion": 0.0,
    "sentiment_market_alignment": 0,
}

DEFAULT_MARKET_CONTEXT_FEATURES = {
    "spy_change_pct": 0.0,
    "spy_gap_pct": 0.0,
    "vix_level": 20.0,
    "vix_change_pct": 0.0,
}

DEFAULT_MACRO_FEATURES = {
    "yield_curve_spread": 0.0,
    "fed_funds_rate": 0.0,
    "fed_funds_change_3m": 0.0,
    "credit_spread": 0.0,
    "dxy_change_5d": 0.0,
    "universe_breadth_50d": 0.0,
}

DEFAULT_CANDIDATE_FEATURES = {
    **DEFAULT_MARKET_FEATURES,
    **DEFAULT_TEMPORAL_FEATURES,
    **DEFAULT_GEMINI_FEATURES,
    **DEFAULT_MARKET_CONTEXT_FEATURES,
    **DEFAULT_MACRO_FEATURES,
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    default_value: float | int
    online_servable: bool = True


FEATURE_SPECS = tuple(
    [FeatureSpec(name, "market", DEFAULT_MARKET_FEATURES[name]) for name in MARKET_FEATURE_NAMES]
    + [FeatureSpec(name, "temporal", DEFAULT_TEMPORAL_FEATURES[name]) for name in TEMPORAL_FEATURE_NAMES]
    + [FeatureSpec(name, "sentiment", DEFAULT_GEMINI_FEATURES[name]) for name in GEMINI_FEATURE_NAMES]
    + [FeatureSpec(name, "context", DEFAULT_MARKET_CONTEXT_FEATURES[name]) for name in MARKET_CONTEXT_FEATURE_NAMES]
    + [FeatureSpec(name, "macro", DEFAULT_MACRO_FEATURES[name]) for name in MACRO_FEATURE_NAMES]
)
FEATURE_SPECS_BY_NAME = {spec.name: spec for spec in FEATURE_SPECS}
ONLINE_SERVABLE_FEATURE_NAMES = [spec.name for spec in FEATURE_SPECS if spec.online_servable]

def build_temporary_selected_feature_contract(feature_count: int = TEMPORARY_SELECTED_FEATURE_COUNT) -> dict:
    resolved_feature_count = max(len(GEMINI_FEATURE_NAMES), min(feature_count, len(CANDIDATE_FEATURE_NAMES)))
    remaining_feature_names = [name for name in CANDIDATE_FEATURE_NAMES if name not in GEMINI_FEATURE_NAMES]
    seed_material = f"{CANDIDATE_FEATURE_VERSION}:{resolved_feature_count}"
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)
    shuffled_feature_names = list(remaining_feature_names)
    random.Random(seed).shuffle(shuffled_feature_names)
    selected_feature_names = list(GEMINI_FEATURE_NAMES) + shuffled_feature_names[
        : resolved_feature_count - len(GEMINI_FEATURE_NAMES)
    ]

    return {
        "run_id": f"temporary-{CANDIDATE_FEATURE_VERSION}-random-{resolved_feature_count}",
        "candidate_feature_version": CANDIDATE_FEATURE_VERSION,
        "candidate_feature_count": len(CANDIDATE_FEATURE_NAMES),
        "selected_feature_names": selected_feature_names,
        "selected_feature_count": len(selected_feature_names),
        "selection_source": TEMPORARY_FEATURE_SELECTION_SOURCE,
        "feature_selection_mode": TEMPORARY_FEATURE_SELECTION_MODE,
        "feature_selection_seed": seed_material,
        "generated_at": None,
        "review_status": "TEMPORARY_FALLBACK",
        "execution_status": "WAITING_FOR_AUTOML",
        "notes": (
            "Temporary deterministic 40-feature subset while the Vertex AutoML feature "
            "ranking contract is still pending. All Gemini news-text features are retained."
        ),
        "is_temporary": True,
    }


_DEFAULT_SELECTED_FEATURE_CONTRACT = build_temporary_selected_feature_contract()

_SELECTED_FEATURES_PATH = Path(__file__).with_name("selected_features.json")


def _load_selected_feature_contract() -> dict:
    default_contract = build_temporary_selected_feature_contract()
    if not _SELECTED_FEATURES_PATH.exists():
        return default_contract

    try:
        payload = json.loads(_SELECTED_FEATURES_PATH.read_text())
    except Exception as exc:
        log.warning("Failed to read selected feature contract from %s: %s", _SELECTED_FEATURES_PATH, exc)
        return default_contract

    feature_names = [
        str(name)
        for name in (payload.get("selected_feature_names") or [])
        if str(name) in CANDIDATE_FEATURE_NAMES
    ]
    if not feature_names:
        return default_contract

    contract = {
        **default_contract,
        **payload,
        "selected_feature_names": feature_names,
        "selected_feature_count": int(payload.get("selected_feature_count") or len(feature_names)),
    }
    contract["is_temporary"] = bool(payload.get("is_temporary")) or (
        contract.get("selection_source") == TEMPORARY_FEATURE_SELECTION_SOURCE
    )
    if contract["is_temporary"] and "feature_selection_mode" not in payload:
        contract["feature_selection_mode"] = TEMPORARY_FEATURE_SELECTION_MODE
    return contract


SELECTED_FEATURE_CONTRACT = _load_selected_feature_contract()
MODEL_FEATURE_NAMES = list(SELECTED_FEATURE_CONTRACT["selected_feature_names"])
EXPECTED_FEATURE_COUNT = len(MODEL_FEATURE_NAMES)

def get_selected_feature_contract() -> dict:
    return dict(SELECTED_FEATURE_CONTRACT)


def candidate_feature_version() -> str:
    return CANDIDATE_FEATURE_VERSION


def _replace_zero(values: pd.Series | np.ndarray | float | int | None):
    if isinstance(values, pd.Series):
        return values.replace(0, np.nan)
    if isinstance(values, np.ndarray):
        out = np.asarray(values, dtype=float).copy()
        out[out == 0] = np.nan
        return out
    if values in (0, 0.0):
        return np.nan
    return values


def _safe_divide(numerator, denominator):
    if isinstance(denominator, pd.Series):
        denominator = _replace_zero(denominator)
    elif isinstance(denominator, np.ndarray):
        denominator = _replace_zero(denominator)
    elif denominator in (0, 0.0):
        denominator = np.nan
    return numerator / denominator


def is_missing_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return is_missing_value(value.item())
        return False
    if isinstance(value, (list, tuple, dict, set, pd.Series, pd.DataFrame, pd.Index)):
        return False
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    if isinstance(missing, (np.ndarray, pd.Series, pd.DataFrame, pd.Index, list, tuple)):
        return False
    return bool(missing)


def coerce_scalarlike(value: object) -> object:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return coerce_scalarlike(value.item())
        if value.size == 1:
            return coerce_scalarlike(value.reshape(-1)[0])
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_float(value: object, default: float | None = None) -> float | None:
    value = coerce_scalarlike(value)
    if is_missing_value(value):
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric):
        return default
    return numeric


def _safe_int(value: object, default: int = 0) -> int:
    numeric = _safe_float(value)
    if numeric is None:
        return default
    return int(round(numeric))


def _safe_is_zero(x: object) -> bool:
    """Return True if x is scalar zero. Safe for numpy arrays (avoids ambiguous truth value)."""
    s = coerce_scalarlike(x)
    if s is None:
        return False
    try:
        return float(s) == 0.0
    except (TypeError, ValueError):
        return False


def _safe_scalar_equals(a: object, b: object) -> bool:
    """Return True if a and b coerce to the same scalar. Safe for numpy arrays."""
    sa, sb = coerce_scalarlike(a), coerce_scalarlike(b)
    if sa is None or sb is None:
        return False
    try:
        return float(sa) == float(sb)
    except (TypeError, ValueError):
        return False


def _as_python(value: Any) -> Any:
    value = coerce_scalarlike(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if is_missing_value(value):
        return None
    return value


def get_feature_spec(feature_name: str) -> FeatureSpec:
    spec = FEATURE_SPECS_BY_NAME.get(str(feature_name))
    if spec is None:
        raise KeyError(f"Unknown feature '{feature_name}'.")
    return spec


def feature_default_values(feature_names: Sequence[str] | None = None) -> dict[str, float | int]:
    names = list(feature_names or CANDIDATE_FEATURE_NAMES)
    return {name: get_feature_spec(name).default_value for name in names}


def validate_online_feature_names(feature_names: Sequence[str]) -> list[str]:
    invalid = [name for name in feature_names if name not in FEATURE_SPECS_BY_NAME]
    if invalid:
        raise ValueError(f"Unknown feature names in contract: {invalid}")
    unsupported = [name for name in feature_names if name not in ONLINE_SERVABLE_FEATURE_NAMES]
    if unsupported:
        raise ValueError(
            "Selected feature contract includes features that are not supported by the shared online feature builder: "
            f"{unsupported}"
        )
    return list(feature_names)


def _coerce_feature_group_frame(
    payload: pd.DataFrame | Mapping[str, object] | None,
    *,
    feature_names: Sequence[str],
    row_count: int,
) -> pd.DataFrame:
    if payload is None:
        frame = pd.DataFrame(index=range(row_count))
    elif isinstance(payload, pd.DataFrame):
        frame = payload.copy().reset_index(drop=True)
    elif isinstance(payload, Mapping):
        frame = pd.DataFrame([dict(payload)]).reset_index(drop=True)
    else:
        raise TypeError(f"Unsupported feature payload type: {type(payload)!r}")

    if len(frame) not in (0, row_count):
        raise ValueError(
            f"Feature payload row count mismatch. Expected {row_count} rows, received {len(frame)} rows."
        )
    if len(frame) == 0 and row_count:
        frame = pd.DataFrame(index=range(row_count))
    for feature_name in feature_names:
        if feature_name not in frame.columns:
            frame[feature_name] = get_feature_spec(feature_name).default_value
    return frame[list(feature_names)].reset_index(drop=True)


def build_candidate_feature_frame(
    *,
    market_features: pd.DataFrame | Mapping[str, object] | None = None,
    temporal_features: pd.DataFrame | Mapping[str, object] | None = None,
    sentiment_features: pd.DataFrame | Mapping[str, object] | None = None,
    context_features: pd.DataFrame | Mapping[str, object] | None = None,
    macro_features: pd.DataFrame | Mapping[str, object] | None = None,
    row_count: int | None = None,
) -> pd.DataFrame:
    resolved_row_count = row_count
    for payload in (market_features, temporal_features, sentiment_features, context_features, macro_features):
        if resolved_row_count is not None:
            break
        if isinstance(payload, pd.DataFrame):
            resolved_row_count = len(payload)
        elif isinstance(payload, Mapping):
            resolved_row_count = 1
    resolved_row_count = 1 if resolved_row_count is None else int(resolved_row_count)

    frames = [
        _coerce_feature_group_frame(market_features, feature_names=MARKET_FEATURE_NAMES, row_count=resolved_row_count),
        _coerce_feature_group_frame(temporal_features, feature_names=TEMPORAL_FEATURE_NAMES, row_count=resolved_row_count),
        _coerce_feature_group_frame(sentiment_features, feature_names=GEMINI_FEATURE_NAMES, row_count=resolved_row_count),
        _coerce_feature_group_frame(context_features, feature_names=MARKET_CONTEXT_FEATURE_NAMES, row_count=resolved_row_count),
        _coerce_feature_group_frame(macro_features, feature_names=MACRO_FEATURE_NAMES, row_count=resolved_row_count),
    ]
    candidate_df = pd.concat(frames, axis=1)
    return candidate_df.reindex(columns=CANDIDATE_FEATURE_NAMES).replace([np.inf, -np.inf], np.nan)


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


def normalize_live_news_items(news_items: Sequence[Mapping[str, object]] | None) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    normalized: list[dict] = []
    for item in news_items or []:
        headline = extract_news_title(item)
        if not headline:
            continue

        url = str(item.get("url") or "").strip()
        dedupe_key = (headline.lower(), url)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        published_at = pd.to_datetime(
            item.get("published_at") or item.get("datetime") or item.get("providerPublishTime"),
            errors="coerce",
            utc=True,
        )
        normalized.append(
            {
                "headline": headline,
                "source": str(item.get("source") or item.get("provider_name") or item.get("provider") or "").strip(),
                "provider": str(item.get("provider") or "unknown").strip() or "unknown",
                "url": url or None,
                "published_at": None if pd.isna(published_at) else published_at.isoformat(),
            }
        )

    normalized.sort(key=lambda row: row["published_at"] or "", reverse=True)
    return normalized


def require_live_news_items(
    news_items: Sequence[Mapping[str, object]] | None,
    *,
    ticker: str | None = None,
) -> list[dict]:
    normalized = normalize_live_news_items(news_items)
    live_news_items = [
        item for item in normalized if str(item.get("provider") or "").strip().lower() in LIVE_NEWS_PROVIDERS
    ]
    if live_news_items:
        return normalized

    ticker_label = f" for {ticker.strip().upper()}" if isinstance(ticker, str) and ticker.strip() else ""
    raise ValueError(
        f"Live news is required for inference{ticker_label}. No live news items were available from Finnhub or Yahoo."
    )


def extract_live_news_sentiment_items(sentiment_features: Mapping[str, object]) -> list[dict]:
    candidate_items = sentiment_features.get(LIVE_NEWS_SENTIMENT_ITEMS_KEY)
    if not isinstance(candidate_items, Sequence) or isinstance(candidate_items, (str, bytes)):
        return []
    return [dict(item) for item in candidate_items if isinstance(item, Mapping)]


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
    return working.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


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

    avg_volume_10d = volume.rolling(10, min_periods=10).mean()
    avg_volume_3m = volume.rolling(63, min_periods=20).mean()
    features["avg_volume_10d"] = avg_volume_10d
    features["avg_volume_3m"] = avg_volume_3m
    features["volume_vs_avg_10d"] = _safe_divide(volume, avg_volume_10d)
    features["volume_vs_avg_3m"] = _safe_divide(volume, avg_volume_3m)

    fifty_day_avg = close.rolling(50, min_periods=50).mean()
    two_hundred_day_avg = close.rolling(200, min_periods=200).mean()
    features["fifty_day_avg"] = fifty_day_avg
    features["two_hundred_day_avg"] = two_hundred_day_avg
    features["price_vs_50d_avg"] = _safe_divide(close, fifty_day_avg) - 1
    features["price_vs_200d_avg"] = _safe_divide(close, two_hundred_day_avg) - 1

    fifty_two_week_high = high.rolling(252, min_periods=252).max()
    fifty_two_week_low = low.rolling(252, min_periods=252).min()
    features["fifty_two_week_high"] = fifty_two_week_high
    features["fifty_two_week_low"] = fifty_two_week_low
    features["price_vs_52w_high"] = _safe_divide(close, fifty_two_week_high) - 1
    features["price_vs_52w_low"] = _safe_divide(close, fifty_two_week_low) - 1
    features["position_52w_range"] = _safe_divide(close - fifty_two_week_low, fifty_two_week_high - fifty_two_week_low)

    return features[MARKET_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)


def _rolling_slope(series: pd.Series) -> float:
    values = series.astype(float).to_numpy()
    if len(values) < 2 or np.isnan(values).any():
        return np.nan
    x = np.arange(len(values), dtype=float)
    slope = np.polyfit(x, values, 1)[0]
    baseline = np.nanmean(np.abs(values)) or 1.0
    return float(slope / baseline)


def _momentum_streak(returns: pd.Series) -> pd.Series:
    streaks: list[int] = []
    running = 0
    for value in returns.fillna(0.0):
        if value > 0:
            running = running + 1 if running >= 0 else 1
        elif value < 0:
            running = running - 1 if running <= 0 else -1
        else:
            running = 0
        streaks.append(running)
    return pd.Series(streaks, index=returns.index, dtype=float)


def compute_temporal_features_from_history(
    df: pd.DataFrame,
    *,
    spy_return_5d: pd.Series | None = None,
) -> pd.DataFrame:
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    daily_ret = close.pct_change()

    features = pd.DataFrame(index=df.index)
    features["return_1d"] = close.pct_change(1)
    features["return_3d"] = close.pct_change(3)
    features["return_5d"] = close.pct_change(5)
    features["return_10d"] = close.pct_change(10)
    features["return_20d"] = close.pct_change(20)
    features["momentum_streak"] = _momentum_streak(daily_ret)
    rolling_vol_5d = daily_ret.rolling(5, min_periods=5).std()
    rolling_vol_20d = daily_ret.rolling(20, min_periods=20).std()
    features["volatility_regime"] = _safe_divide(rolling_vol_5d, rolling_vol_20d)
    features["volume_trend_5d"] = volume.rolling(5, min_periods=5).apply(_rolling_slope, raw=False)
    features["price_acceleration"] = features["return_5d"] - features["return_10d"]

    if spy_return_5d is None:
        features["relative_strength_spy"] = np.nan
    else:
        aligned_spy = spy_return_5d.reindex(df.index)
        features["relative_strength_spy"] = features["return_5d"] - aligned_spy

    return features[TEMPORAL_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)


def build_market_temporal_feature_frame_from_history(
    df: pd.DataFrame,
    *,
    spy_return_5d: pd.Series | None = None,
) -> pd.DataFrame:
    market_features = compute_market_features_from_history(df)
    temporal_features = compute_temporal_features_from_history(df, spy_return_5d=spy_return_5d)
    return build_candidate_feature_frame(
        market_features=market_features,
        temporal_features=temporal_features,
        row_count=len(df.index),
    )[MARKET_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES]


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

    frame = pd.DataFrame({"date": date_series, "_row_id": np.arange(len(date_series))})
    for column, default_value in DEFAULT_MARKET_CONTEXT_FEATURES.items():
        frame[column] = default_value

    if not context_history.empty:
        context = context_history.copy()
        context["date"] = pd.to_datetime(context["date"], errors="coerce")
        if getattr(context["date"].dt, "tz", None) is not None:
            context["date"] = context["date"].dt.tz_convert(None)
        context["date"] = context["date"].dt.normalize().astype("datetime64[ns]")

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


def align_model_features(
    feature_df: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    expected = list(feature_names or MODEL_FEATURE_NAMES)
    missing_columns = [column for column in expected if column not in feature_df.columns]
    if missing_columns:
        raise AssertionError(f"Missing required feature columns: {missing_columns}")

    aligned = feature_df.reindex(columns=expected).copy()
    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    if aligned.isna().any().any():
        missing = aligned.columns[aligned.isna().any()].tolist()
        raise AssertionError(f"Required features contain null values: {missing}")
    return aligned


def _trace_value(trace: Mapping[str, object] | None, default: float = 0.0) -> float:
    if not isinstance(trace, Mapping):
        return default
    value = _safe_float(trace.get("value"), default)
    return default if value is None else value


def _titleize_feature_name(name: str) -> str:
    return name.replace("_", " ").title()


def _build_feature_detail(
    *,
    name: str,
    value: Any,
    group: str,
    source: str,
    providers: Sequence[str] | None = None,
    raw_fields: Sequence[str] | None = None,
    fallback_used: bool = False,
) -> dict:
    return {
        "name": name,
        "label": _titleize_feature_name(name),
        "value": _as_python(value),
        "group": group,
        "source": source,
        "providers": list(dict.fromkeys(providers or [source])),
        "raw_fields": list(dict.fromkeys(raw_fields or [name])),
        "fallback_used": bool(fallback_used),
    }


def _normalize_history_rows(rows: Sequence[Mapping[str, object]] | pd.DataFrame | None) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return prepare_price_history_frame(rows)
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    return prepare_price_history_frame(pd.DataFrame(list(rows)))


def _compute_live_candidate_features(
    live_snapshot: Mapping[str, object],
    *,
    sentiment_features: Mapping[str, object],
    macro_features: Mapping[str, object],
) -> tuple[dict[str, float | int], dict[str, dict]]:
    quote = dict(live_snapshot.get("quote") or {})
    context_payload = dict(live_snapshot.get("market_context") or {})

    feature_sources: dict[str, dict] = {}
    candidate_values: dict[str, float | int] = {}

    raw_market_fields = {
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

    current_price = _trace_value(raw_market_fields["current_price"])
    previous_close = _trace_value(raw_market_fields["previous_close"])
    day_open = _trace_value(raw_market_fields["day_open"])
    day_high = _trace_value(raw_market_fields["day_high"])
    day_low = _trace_value(raw_market_fields["day_low"])
    volume = _trace_value(raw_market_fields["volume"])
    avg_volume_10d = _trace_value(raw_market_fields["avg_volume_10d"])
    avg_volume_3m = _trace_value(raw_market_fields["avg_volume_3m"])
    fifty_day_avg = _trace_value(raw_market_fields["fifty_day_avg"])
    two_hundred_day_avg = _trace_value(raw_market_fields["two_hundred_day_avg"])
    fifty_two_week_high = _trace_value(raw_market_fields["fifty_two_week_high"])
    fifty_two_week_low = _trace_value(raw_market_fields["fifty_two_week_low"])

    candidate_values.update(
        {
            "current_price": current_price,
            "previous_close": previous_close,
            "day_open": day_open,
            "day_high": day_high,
            "day_low": day_low,
            "day_change": current_price - previous_close,
            "day_change_pct": 0.0 if _safe_is_zero(previous_close) else current_price / previous_close - 1,
            "gap_pct": 0.0 if _safe_is_zero(previous_close) else day_open / previous_close - 1,
            "intraday_range_pct": 0.0 if _safe_is_zero(previous_close) else (day_high - day_low) / previous_close,
            "close_vs_open_pct": 0.0 if _safe_is_zero(day_open) else current_price / day_open - 1,
            "day_range_position": 0.0 if _safe_scalar_equals(day_high, day_low) else (current_price - day_low) / (day_high - day_low),
            "volume": volume,
            "avg_volume_10d": avg_volume_10d,
            "avg_volume_3m": avg_volume_3m,
            "volume_vs_avg_10d": 0.0 if _safe_is_zero(avg_volume_10d) else volume / avg_volume_10d,
            "volume_vs_avg_3m": 0.0 if _safe_is_zero(avg_volume_3m) else volume / avg_volume_3m,
            "fifty_day_avg": fifty_day_avg,
            "two_hundred_day_avg": two_hundred_day_avg,
            "price_vs_50d_avg": 0.0 if _safe_is_zero(fifty_day_avg) else current_price / fifty_day_avg - 1,
            "price_vs_200d_avg": 0.0 if _safe_is_zero(two_hundred_day_avg) else current_price / two_hundred_day_avg - 1,
            "fifty_two_week_high": fifty_two_week_high,
            "fifty_two_week_low": fifty_two_week_low,
            "price_vs_52w_high": 0.0 if _safe_is_zero(fifty_two_week_high) else current_price / fifty_two_week_high - 1,
            "price_vs_52w_low": 0.0 if _safe_is_zero(fifty_two_week_low) else current_price / fifty_two_week_low - 1,
            "position_52w_range": (
                0.0
                if _safe_scalar_equals(fifty_two_week_high, fifty_two_week_low)
                else (current_price - fifty_two_week_low) / (fifty_two_week_high - fifty_two_week_low)
            ),
        }
    )
    for feature_name, trace in raw_market_fields.items():
        trace_payload = dict(trace or {})
        feature_sources[feature_name] = {
            "source": str(trace_payload.get("provider") or "unknown"),
            "providers": [str(trace_payload.get("provider") or "unknown")],
            "raw_fields": [str(trace_payload.get("raw_field") or feature_name)],
            "fallback_used": bool(trace_payload.get("fallback_used", False)),
        }

    ticker_history = _normalize_history_rows(live_snapshot.get("ticker_history"))
    spy_history = _normalize_history_rows(live_snapshot.get("spy_history"))
    if ticker_history.empty:
        raise RuntimeError("Ticker history is unavailable for temporal feature computation.")
    ticker_history = ticker_history.set_index("date")
    ticker_history.columns = [column.capitalize() for column in ticker_history.columns]
    spy_return_5d = None
    if not spy_history.empty:
        spy_history = spy_history.set_index("date")
        spy_return_5d = spy_history["close"].astype(float).pct_change(5)
    history_feature_frame = build_market_temporal_feature_frame_from_history(
        ticker_history,
        spy_return_5d=spy_return_5d,
    )
    temporal_features = history_feature_frame[TEMPORAL_FEATURE_NAMES].iloc[-1]
    for feature_name in TEMPORAL_FEATURE_NAMES:
        candidate_values[feature_name] = _safe_float(temporal_features.get(feature_name), 0.0) or 0.0
        feature_sources[feature_name] = {
            "source": "derived",
            "providers": ["yahoo"],
            "raw_fields": ["ticker_history", "spy_history"],
            "fallback_used": False,
        }

    spy = dict(context_payload.get("spy") or {})
    vix = dict(context_payload.get("vix") or {})
    spy_current = _trace_value(spy.get("current_price"))
    spy_previous_close = _trace_value(spy.get("previous_close"))
    spy_open = _trace_value(spy.get("day_open"))
    vix_current = _trace_value(vix.get("current_price"), DEFAULT_MARKET_CONTEXT_FEATURES["vix_level"])
    vix_previous_close = _trace_value(vix.get("previous_close"))
    candidate_values.update(
        {
            "spy_change_pct": 0.0 if _safe_is_zero(spy_previous_close) else spy_current / spy_previous_close - 1,
            "spy_gap_pct": 0.0 if _safe_is_zero(spy_previous_close) else spy_open / spy_previous_close - 1,
            "vix_level": vix_current,
            "vix_change_pct": 0.0 if _safe_is_zero(vix_previous_close) else vix_current / vix_previous_close - 1,
        }
    )
    feature_sources.update(
        {
            "spy_change_pct": {"source": "derived", "providers": ["finnhub", "yahoo"], "raw_fields": ["spy.quote"], "fallback_used": False},
            "spy_gap_pct": {"source": "derived", "providers": ["finnhub", "yahoo"], "raw_fields": ["spy.quote"], "fallback_used": False},
            "vix_level": {"source": "yahoo", "providers": ["yahoo"], "raw_fields": ["vix.quote"], "fallback_used": False},
            "vix_change_pct": {"source": "derived", "providers": ["yahoo"], "raw_fields": ["vix.quote"], "fallback_used": False},
        }
    )

    normalized_sentiment = {**DEFAULT_GEMINI_FEATURES, **{key: _as_python(value) for key, value in dict(sentiment_features).items()}}
    day_change_pct = float(candidate_values["day_change_pct"])
    sentiment_avg = _safe_float(normalized_sentiment.get("gemini_sentiment_avg"), 0.0) or 0.0
    market_sign = 1 if day_change_pct > 0 else -1 if day_change_pct < 0 else 0
    sentiment_sign = 1 if sentiment_avg > 0 else -1 if sentiment_avg < 0 else 0
    normalized_sentiment["sentiment_market_alignment"] = int(market_sign != 0 and sentiment_sign == market_sign)
    for feature_name in GEMINI_FEATURE_NAMES:
        default_value = DEFAULT_GEMINI_FEATURES[feature_name]
        if isinstance(default_value, int):
            candidate_values[feature_name] = _safe_int(normalized_sentiment.get(feature_name), default_value)
        else:
            candidate_values[feature_name] = _safe_float(normalized_sentiment.get(feature_name), float(default_value)) or float(default_value)
        feature_sources[feature_name] = {
            "source": "gemini",
            "providers": ["gemini"],
            "raw_fields": ["raw_news_data", "gemini-2.5-flash-lite"],
            "fallback_used": False,
        }

    normalized_macro = {**DEFAULT_MACRO_FEATURES, **dict(macro_features)}
    for feature_name in MACRO_FEATURE_NAMES:
        candidate_values[feature_name] = _safe_float(normalized_macro.get(feature_name), DEFAULT_MACRO_FEATURES[feature_name]) or DEFAULT_MACRO_FEATURES[feature_name]
        feature_sources[feature_name] = {
            "source": str((dict(macro_features).get("_sources") or {}).get(feature_name, "fred")),
            "providers": [str((dict(macro_features).get("_sources") or {}).get(feature_name, "fred"))],
            "raw_fields": [feature_name],
            "fallback_used": False,
        }

    return candidate_values, feature_sources


def build_live_feature_bundle(
    live_snapshot: Mapping[str, object],
    *,
    sentiment_features: Mapping[str, object],
    macro_features: Mapping[str, object],
    feature_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    company_name = live_snapshot.get("company_name")
    company_name = company_name.strip() if isinstance(company_name, str) else None
    quote_currency = live_snapshot.get("quote_currency")
    quote_currency = quote_currency.strip().upper() if isinstance(quote_currency, str) else None
    as_of = pd.Timestamp(live_snapshot.get("as_of") or pd.Timestamp.utcnow())
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")

    candidate_values, feature_sources = _compute_live_candidate_features(
        live_snapshot,
        sentiment_features=sentiment_features,
        macro_features=macro_features,
    )
    expected_feature_names = list(feature_names or MODEL_FEATURE_NAMES)
    validate_online_feature_names(expected_feature_names)
    candidate_df = build_candidate_feature_frame(
        market_features={name: candidate_values.get(name) for name in MARKET_FEATURE_NAMES},
        temporal_features={name: candidate_values.get(name) for name in TEMPORAL_FEATURE_NAMES},
        sentiment_features={name: candidate_values.get(name) for name in GEMINI_FEATURE_NAMES},
        context_features={name: candidate_values.get(name) for name in MARKET_CONTEXT_FEATURE_NAMES},
        macro_features={name: candidate_values.get(name) for name in MACRO_FEATURE_NAMES},
        row_count=1,
    )
    selected_df = align_model_features(candidate_df, feature_names)

    normalized_news = extract_live_news_sentiment_items(sentiment_features)
    if not normalized_news:
        normalized_news = normalize_live_news_items(live_snapshot.get("news"))

    grouped_details: dict[str, list[dict]] = defaultdict(list)
    for group_key, feature_names_in_group in FEATURE_GROUPS.items():
        for feature_name in feature_names_in_group:
            source_meta = feature_sources.get(feature_name, {})
            grouped_details[group_key].append(
                _build_feature_detail(
                    name=feature_name,
                    value=candidate_values.get(feature_name),
                    group=group_key,
                    source=str(source_meta.get("source") or "derived"),
                    providers=source_meta.get("providers"),
                    raw_fields=source_meta.get("raw_fields"),
                    fallback_used=bool(source_meta.get("fallback_used", False)),
                )
            )

    feature_snapshot = {
        "as_of": as_of.isoformat(),
        "company_name": company_name,
        "quote_currency": quote_currency,
        "feature_count": len(CANDIDATE_FEATURE_NAMES),
        "selected_feature_count": len(selected_df.columns),
        "headlines_text": " ".join(item["headline"] for item in normalized_news),
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
            "gemini": len(GEMINI_FEATURE_NAMES),
            "fred": len([name for name in MACRO_FEATURE_NAMES if feature_sources.get(name, {}).get("source") == "fred"]),
            "bigquery": len([name for name in MACRO_FEATURE_NAMES if feature_sources.get(name, {}).get("source") == "bigquery"]),
            "derived": len(TEMPORAL_FEATURE_NAMES) + 2,
        },
    }
    return candidate_df, selected_df, feature_snapshot
