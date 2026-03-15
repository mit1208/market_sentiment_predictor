import sys
import types

import pandas as pd

from backend.ml.shared.feature_engineering import LIVE_NEWS_SENTIMENT_ITEMS_KEY
from backend.ml.shared.gemini_sentiment import (
    headline_hash,
    score_headline_batch_with_gemini,
    score_headline_with_gemini,
    score_live_news_items,
)


def _install_fake_genai_modules():
    genai_module = types.ModuleType("google.genai")
    errors_module = types.ModuleType("google.genai.errors")
    types_module = types.ModuleType("google.genai.types")

    class FakeClientError(Exception):
        def __init__(self, status_code, message="client error"):
            super().__init__(message)
            self.status_code = status_code

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    setattr(errors_module, "ClientError", FakeClientError)
    setattr(types_module, "GenerateContentConfig", GenerateContentConfig)
    setattr(genai_module, "errors", errors_module)
    setattr(genai_module, "types", types_module)

    sys.modules["google.genai"] = genai_module
    sys.modules["google.genai.errors"] = errors_module
    sys.modules["google.genai.types"] = types_module
    return FakeClientError


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeModels:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def generate_content(self, **_kwargs):
        self.calls += 1
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeClient:
    def __init__(self, responses):
        self.models = _FakeModels(responses)


def test_score_live_news_items_returns_per_headline_sentiment_details(monkeypatch):
    def fake_load_or_score_headlines(headline_rows, **_kwargs):
        return (
            pd.DataFrame(
                [
                    {
                        "headline_hash": headline_hash("Apple rallies after earnings beat"),
                        "ticker": "AAPL",
                        "headline": "Apple rallies after earnings beat",
                        "sentiment_score": 0.72,
                        "magnitude": 0.81,
                        "category": "earnings",
                        "scored_at": "2026-03-14T13:05:00+00:00",
                    },
                    {
                        "headline_hash": headline_hash("Analyst cuts Apple estimates"),
                        "ticker": "AAPL",
                        "headline": "Analyst cuts Apple estimates",
                        "sentiment_score": -0.44,
                        "magnitude": 0.66,
                        "category": "analyst",
                        "scored_at": "2026-03-14T13:07:00+00:00",
                    },
                ]
            ),
            [],
        )

    monkeypatch.setattr(
        "backend.ml.shared.gemini_sentiment.load_or_score_headlines",
        fake_load_or_score_headlines,
    )

    payload = score_live_news_items(
        ticker="AAPL",
        news_items=[
            {
                "headline": "Apple rallies after earnings beat",
                "provider": "yahoo",
                "source": "Reuters",
                "published_at": "2026-03-14T13:00:00+00:00",
            },
            {
                "headline": "Analyst cuts Apple estimates",
                "provider": "finnhub",
                "source": "MarketWatch",
                "published_at": "2026-03-14T12:30:00+00:00",
            },
        ],
        project_id="proj",
        region="us-central1",
    )

    assert payload["gemini_sentiment_avg"] == 0.14
    assert payload["headline_count"] == 2
    assert payload["has_earnings_news"] == 1
    assert payload["has_analyst_news"] == 1
    assert payload[LIVE_NEWS_SENTIMENT_ITEMS_KEY][0]["sentiment_label"] == "positive"
    assert payload[LIVE_NEWS_SENTIMENT_ITEMS_KEY][1]["sentiment_label"] == "negative"
    assert payload[LIVE_NEWS_SENTIMENT_ITEMS_KEY][0]["category"] == "earnings"


def test_score_headline_with_gemini_retries_invalid_payload(monkeypatch):
    _install_fake_genai_modules()
    fake_client = _FakeClient(
        [
            _FakeResponse("{}"),
            _FakeResponse('{"sentiment_score": 0.34, "magnitude": 0.52, "category": "earnings"}'),
        ]
    )

    monkeypatch.setattr("backend.ml.shared.gemini_sentiment._gemini_client", lambda *_args, **_kwargs: fake_client)
    monkeypatch.setattr("backend.ml.shared.gemini_sentiment.time.sleep", lambda *_args, **_kwargs: None)

    payload = score_headline_with_gemini(
        ticker="AAPL",
        headline="Apple beats earnings estimates",
        project_id="proj",
        region="us-central1",
    )

    assert fake_client.models.calls == 2
    assert payload["sentiment_score"] == 0.34
    assert payload["magnitude"] == 0.52
    assert payload["category"] == "earnings"


def test_score_headline_batch_with_gemini_falls_back_for_invalid_batch_item(monkeypatch):
    _install_fake_genai_modules()
    fake_client = _FakeClient(
        [
            _FakeResponse(
                '{"items": ['
                '{"index": 0, "sentiment_score": 0.61, "magnitude": 0.7, "category": "product"}, '
                '{"index": 1, "sentiment_score": -0.2, "category": "analyst"}'
                "]}",
            ),
            _FakeResponse('{"sentiment_score": -0.2, "magnitude": 0.45, "category": "analyst"}'),
        ]
    )

    monkeypatch.setattr("backend.ml.shared.gemini_sentiment._gemini_client", lambda *_args, **_kwargs: fake_client)
    monkeypatch.setattr("backend.ml.shared.gemini_sentiment.time.sleep", lambda *_args, **_kwargs: None)

    rows = score_headline_batch_with_gemini(
        [
            {"ticker": "AAPL", "headline": "Apple unveils new product line"},
            {"ticker": "AAPL", "headline": "Analyst cuts Apple price target"},
        ],
        project_id="proj",
        region="us-central1",
    )

    assert fake_client.models.calls == 2
    assert len(rows) == 2
    assert rows[0]["category"] == "product"
    assert rows[1]["category"] == "analyst"
    assert rows[1]["magnitude"] == 0.45
