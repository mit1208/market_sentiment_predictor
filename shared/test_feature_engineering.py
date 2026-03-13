import pytest
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from shared.feature_engineering import (
    MODEL_FEATURE_NAMES,
    build_live_feature_bundle,
    normalize_headlines,
    extract_news_title,
    compute_headline_features,
    _safe_divide,
    _safe_float
)

def test_normalize_headlines():
    assert normalize_headlines(None) == []
    assert normalize_headlines("") == []
    assert normalize_headlines("  ") == []
    assert normalize_headlines("Headline 1") == ["Headline 1"]
    assert normalize_headlines(["Headline 1", "  ", None, "Headline 2"]) == ["Headline 1", "Headline 2"]

def test_extract_news_title():
    assert extract_news_title(None) is None
    assert extract_news_title({}) is None
    assert extract_news_title({"title": "Test Title"}) == "Test Title"
    assert extract_news_title({"headline": "Test Headline"}) == "Test Headline"
    assert extract_news_title({"content": {"title": "Nested Title"}}) == "Nested Title"
    assert extract_news_title({"title": "  ", "headline": "Real Title"}) == "Real Title"

def test_safe_divide():
    assert _safe_divide(10, 2) == 5
    assert np.isnan(_safe_divide(10, 0))
    assert np.isnan(_safe_divide(10, 0.0))
    
    s1 = pd.Series([10, 20])
    s2 = pd.Series([2, 0])
    result = _safe_divide(s1, s2)
    assert result[0] == 5
    assert np.isnan(result[1])

def test_safe_float():
    assert _safe_float(None) is None
    assert _safe_float(None, default=0.0) == 0.0
    assert _safe_float("1.23") == 1.23
    assert _safe_float("abc", default=0.0) == 0.0
    assert _safe_float(np.nan) is None
    assert _safe_float(float('inf')) is None

def test_compute_headline_features_empty():
    features = compute_headline_features([])
    assert features["has_news"] == 0
    assert features["headline_count"] == 0
    assert features["sentiment_score"] == 0.0


def _trace(value, provider="yahoo", raw_field="test.field"):
    return {
        "value": value,
        "provider": provider,
        "raw_field": raw_field,
        "fallback_used": False,
    }


def test_build_live_feature_bundle_includes_quote_currency():
    vectorizer = TfidfVectorizer(max_features=16)
    vectorizer.fit(["earnings beat expectations", "market volatility eases"])

    live_snapshot = {
        "as_of": "2026-03-13T21:15:00Z",
        "quote_currency": "cad",
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
        "news": [
            {
                "headline": "Company beats earnings expectations",
                "source": "Reuters",
                "provider": "yahoo",
                "published_at": "2026-03-13T20:30:00Z",
            }
        ],
    }

    frame, snapshot = build_live_feature_bundle(live_snapshot, vectorizer, MODEL_FEATURE_NAMES)

    assert list(frame.columns) == MODEL_FEATURE_NAMES
    assert snapshot["quote_currency"] == "CAD"
