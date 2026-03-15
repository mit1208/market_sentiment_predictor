import numpy as np
import pandas as pd
import pytest

from backend.ml.shared.feature_engineering import (
    CANDIDATE_FEATURE_NAMES,
    DEFAULT_GEMINI_FEATURES,
    DEFAULT_MACRO_FEATURES,
    DEFAULT_MARKET_CONTEXT_FEATURES,
    DEFAULT_MARKET_FEATURES,
    DEFAULT_TEMPORAL_FEATURES,
    GEMINI_FEATURE_NAMES,
    LIVE_NEWS_SENTIMENT_ITEMS_KEY,
    LIVE_NEWS_PROVIDERS,
    MODEL_FEATURE_NAMES,
    TEMPORARY_SELECTED_FEATURE_COUNT,
    build_candidate_feature_frame,
    build_live_feature_bundle,
    build_temporary_selected_feature_contract,
    extract_news_title,
    normalize_headlines,
    require_live_news_items,
    validate_online_feature_names,
)


def test_normalize_headlines():
    assert normalize_headlines(None) == []
    assert normalize_headlines("") == []
    assert normalize_headlines("Headline 1") == ["Headline 1"]
    assert normalize_headlines(["Headline 1", "  ", None, "Headline 2"]) == ["Headline 1", "Headline 2"]


def test_extract_news_title():
    assert extract_news_title(None) is None
    assert extract_news_title({}) is None
    assert extract_news_title({"title": "Test Title"}) == "Test Title"
    assert extract_news_title({"headline": "Test Headline"}) == "Test Headline"
    assert extract_news_title({"content": {"title": "Nested Title"}}) == "Nested Title"


def test_temporary_selected_feature_contract_keeps_all_gemini_features():
    contract = build_temporary_selected_feature_contract()

    assert contract["is_temporary"] is True
    assert contract["selected_feature_count"] == TEMPORARY_SELECTED_FEATURE_COUNT
    assert contract["selected_feature_names"][: len(GEMINI_FEATURE_NAMES)] == GEMINI_FEATURE_NAMES
    assert set(contract["selected_feature_names"]).issubset(set(CANDIDATE_FEATURE_NAMES))


def test_build_candidate_feature_frame_fills_missing_groups_with_shared_defaults():
    candidate_frame = build_candidate_feature_frame(
        market_features={"current_price": 103.5},
        row_count=1,
    )

    assert list(candidate_frame.columns) == CANDIDATE_FEATURE_NAMES
    assert candidate_frame.iloc[0]["current_price"] == 103.5
    assert candidate_frame.iloc[0]["previous_close"] == DEFAULT_MARKET_FEATURES["previous_close"]
    assert candidate_frame.iloc[0]["return_5d"] == DEFAULT_TEMPORAL_FEATURES["return_5d"]
    assert candidate_frame.iloc[0]["gemini_sentiment_avg"] == DEFAULT_GEMINI_FEATURES["gemini_sentiment_avg"]
    assert candidate_frame.iloc[0]["spy_change_pct"] == DEFAULT_MARKET_CONTEXT_FEATURES["spy_change_pct"]
    assert candidate_frame.iloc[0]["credit_spread"] == DEFAULT_MACRO_FEATURES["credit_spread"]


def test_build_candidate_feature_frame_preserves_empty_row_count():
    candidate_frame = build_candidate_feature_frame(row_count=0)

    assert list(candidate_frame.columns) == CANDIDATE_FEATURE_NAMES
    assert candidate_frame.empty


def test_validate_online_feature_names_rejects_unknown_features():
    with pytest.raises(ValueError) as exc_info:
        validate_online_feature_names(["current_price", "unknown_feature"])

    assert str(exc_info.value) == "Unknown feature names in contract: ['unknown_feature']"


def test_require_live_news_items_rejects_request_only_headlines():
    news_items = [
        {"headline": "User supplied headline", "provider": "request", "published_at": "2026-03-14T13:00:00Z"},
    ]

    with pytest.raises(ValueError) as exc_info:
        require_live_news_items(news_items, ticker="AAPL")

    assert str(exc_info.value) == (
        "Live news is required for inference for AAPL. "
        "No live news items were available from Finnhub or Yahoo."
    )


def test_require_live_news_items_accepts_live_provider_and_preserves_supplemental_headlines():
    news_items = [
        {"headline": "Reuters-style live item", "provider": sorted(LIVE_NEWS_PROVIDERS)[0], "published_at": "2026-03-14T13:00:00Z"},
        {"headline": "User supplied headline", "provider": "request", "published_at": "2026-03-14T14:00:00Z"},
    ]

    normalized = require_live_news_items(news_items, ticker="AAPL")

    assert len(normalized) == 2
    assert {item["provider"] for item in normalized} == {"request", sorted(LIVE_NEWS_PROVIDERS)[0]}


def _trace(value, provider="yahoo", raw_field="test.field"):
    return {
        "value": value,
        "provider": provider,
        "raw_field": raw_field,
        "fallback_used": False,
    }


def test_build_live_feature_bundle_includes_candidate_and_selected_rows():
    live_snapshot = {
        "as_of": "2026-03-14T15:30:00Z",
        "quote_currency": "usd",
        "company_name": "Apple Inc.",
        "quote": {
            "current_price": _trace(103.5, raw_field="quote.c"),
            "previous_close": _trace(100.0, raw_field="quote.pc"),
            "day_open": _trace(101.2, raw_field="quote.o"),
            "day_high": _trace(104.0, raw_field="quote.h"),
            "day_low": _trace(99.8, raw_field="quote.l"),
            "volume": _trace(1500000, raw_field="history.volume"),
            "avg_volume_10d": _trace(1200000, raw_field="history.avg_volume_10d"),
            "avg_volume_3m": _trace(1100000, raw_field="history.avg_volume_3m"),
            "fifty_day_avg": _trace(98.0, raw_field="history.fifty_day_avg"),
            "two_hundred_day_avg": _trace(95.0, raw_field="history.two_hundred_day_avg"),
            "fifty_two_week_high": _trace(120.0, raw_field="history.fifty_two_week_high"),
            "fifty_two_week_low": _trace(80.0, raw_field="history.fifty_two_week_low"),
        },
        "market_context": {
            "spy": {
                "current_price": _trace(510.0, provider="finnhub", raw_field="quote.c"),
                "previous_close": _trace(505.0, provider="finnhub", raw_field="quote.pc"),
                "day_open": _trace(507.0, provider="finnhub", raw_field="quote.o"),
            },
            "vix": {
                "current_price": _trace(18.0, raw_field="quote.c"),
                "previous_close": _trace(17.5, raw_field="quote.pc"),
            },
        },
        "ticker_history": [
            {"date": "2026-01-01", "open": 90, "high": 91, "low": 89, "close": 90, "volume": 1000},
            {"date": "2026-01-02", "open": 91, "high": 92, "low": 90, "close": 91, "volume": 1100},
            {"date": "2026-01-05", "open": 92, "high": 93, "low": 91, "close": 92, "volume": 1200},
            {"date": "2026-01-06", "open": 93, "high": 94, "low": 92, "close": 93, "volume": 1300},
            {"date": "2026-01-07", "open": 94, "high": 95, "low": 93, "close": 94, "volume": 1400},
            {"date": "2026-01-08", "open": 95, "high": 96, "low": 94, "close": 95, "volume": 1500},
            {"date": "2026-01-09", "open": 96, "high": 97, "low": 95, "close": 96, "volume": 1600},
            {"date": "2026-01-12", "open": 97, "high": 98, "low": 96, "close": 97, "volume": 1700},
            {"date": "2026-01-13", "open": 98, "high": 99, "low": 97, "close": 98, "volume": 1800},
            {"date": "2026-01-14", "open": 99, "high": 100, "low": 98, "close": 99, "volume": 1900},
            {"date": "2026-01-15", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 2000},
            {"date": "2026-01-16", "open": 101, "high": 102, "low": 100, "close": 101, "volume": 2100},
            {"date": "2026-01-20", "open": 102, "high": 103, "low": 101, "close": 102, "volume": 2200},
            {"date": "2026-01-21", "open": 103, "high": 104, "low": 102, "close": 103, "volume": 2300},
            {"date": "2026-01-22", "open": 104, "high": 105, "low": 103, "close": 104, "volume": 2400},
            {"date": "2026-01-23", "open": 105, "high": 106, "low": 104, "close": 105, "volume": 2500},
            {"date": "2026-01-26", "open": 106, "high": 107, "low": 105, "close": 106, "volume": 2600},
            {"date": "2026-01-27", "open": 107, "high": 108, "low": 106, "close": 107, "volume": 2700},
            {"date": "2026-01-28", "open": 108, "high": 109, "low": 107, "close": 108, "volume": 2800},
            {"date": "2026-01-29", "open": 109, "high": 110, "low": 108, "close": 109, "volume": 2900},
            {"date": "2026-01-30", "open": 110, "high": 111, "low": 109, "close": 110, "volume": 3000},
        ],
        "spy_history": [
            {"date": "2026-01-23", "open": 500, "high": 501, "low": 499, "close": 500, "volume": 1000},
            {"date": "2026-01-26", "open": 501, "high": 502, "low": 500, "close": 501, "volume": 1000},
            {"date": "2026-01-27", "open": 502, "high": 503, "low": 501, "close": 502, "volume": 1000},
            {"date": "2026-01-28", "open": 503, "high": 504, "low": 502, "close": 503, "volume": 1000},
            {"date": "2026-01-29", "open": 504, "high": 505, "low": 503, "close": 504, "volume": 1000},
            {"date": "2026-01-30", "open": 505, "high": 506, "low": 504, "close": 505, "volume": 1000},
        ],
        "news": [
            {"headline": "Company beats earnings expectations", "source": "Reuters", "provider": "yahoo", "published_at": "2026-03-14T13:00:00Z"},
        ],
    }
    sentiment_features = {
        "gemini_sentiment_avg": 0.6,
        "gemini_sentiment_max": 0.8,
        "gemini_sentiment_min": 0.2,
        "gemini_magnitude_avg": 0.7,
        "headline_count": 1,
        "has_earnings_news": 1,
        "has_analyst_news": 0,
        "sentiment_dispersion": 0.0,
        "sentiment_market_alignment": 1,
        LIVE_NEWS_SENTIMENT_ITEMS_KEY: [
            {
                "headline": "Company beats earnings expectations",
                "source": "Reuters",
                "provider": "yahoo",
                "url": None,
                "published_at": "2026-03-14T13:00:00+00:00",
                "sentiment_score": 0.6,
                "magnitude": 0.7,
                "category": "earnings",
                "scored_at": "2026-03-14T13:05:00+00:00",
                "sentiment_label": "positive",
            }
        ],
    }
    macro_features = {
        "yield_curve_spread": 0.8,
        "fed_funds_rate": 4.5,
        "fed_funds_change_3m": 0.25,
        "credit_spread": 3.2,
        "dxy_change_5d": -0.01,
        "universe_breadth_50d": 0.64,
        "_sources": {"universe_breadth_50d": "bigquery"},
    }

    candidate_frame, selected_frame, snapshot = build_live_feature_bundle(
        live_snapshot,
        sentiment_features=sentiment_features,
        macro_features=macro_features,
        feature_names=MODEL_FEATURE_NAMES,
    )

    assert list(candidate_frame.columns) == CANDIDATE_FEATURE_NAMES
    assert list(selected_frame.columns) == MODEL_FEATURE_NAMES
    assert snapshot["quote_currency"] == "USD"
    assert snapshot["feature_count"] == len(CANDIDATE_FEATURE_NAMES)
    assert snapshot["selected_feature_count"] == len(MODEL_FEATURE_NAMES)
    assert snapshot["news_items"][0]["sentiment_score"] == 0.6
    assert snapshot["news_items"][0]["sentiment_label"] == "positive"
    assert np.isfinite(candidate_frame["return_5d"].iloc[0])
