"""Live market and news fetchers with Finnhub primary quote data and Yahoo fallback/history."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from shared.feature_engineering import NEWS_LOOKBACK_DAYS, prepare_price_history_frame

log = logging.getLogger(__name__)

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
YAHOO_HISTORY_PERIOD = "2y"


def _trace(
    value: Any,
    provider: str,
    raw_field: str,
    *,
    fallback_used: bool = False,
) -> dict | None:
    if value is None or pd.isna(value):
        return None
    return {
        "value": value.item() if hasattr(value, "item") else value,
        "provider": provider,
        "raw_field": raw_field,
        "fallback_used": fallback_used,
    }


def _pick_trace(*candidates: dict | None) -> dict:
    for index, candidate in enumerate(candidates):
        if candidate is None:
            continue
        picked = dict(candidate)
        picked["fallback_used"] = bool(index > 0 or picked.get("fallback_used", False))
        return picked
    return {
        "value": 0.0,
        "provider": "default",
        "raw_field": "default",
        "fallback_used": False,
    }


def _request_json(path: str, *, token: str | None, params: dict[str, Any]) -> dict | list:
    if not token:
        return {}

    url = f"{FINNHUB_BASE_URL}/{path.lstrip('/')}"
    response = requests.get(
        url,
        params={**params, "token": token},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def _fetch_finnhub_quote(symbol: str, token: str | None) -> dict:
    try:
        payload = _request_json("quote", token=token, params={"symbol": symbol})
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        log.warning("Finnhub quote failed for %s: %s", symbol, exc)
        return {}


def _fetch_finnhub_company_news(symbol: str, token: str | None) -> list[dict]:
    if not token:
        return []

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=NEWS_LOOKBACK_DAYS)
    try:
        payload = _request_json(
            "company-news",
            token=token,
            params={
                "symbol": symbol,
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
            },
        )
        if not isinstance(payload, list):
            return []
        return [
            {
                "headline": item.get("headline"),
                "source": item.get("source"),
                "provider": "finnhub",
                "url": item.get("url"),
                "published_at": None
                if item.get("datetime") is None
                else datetime.fromtimestamp(int(item["datetime"]), tz=timezone.utc).isoformat(),
            }
            for item in payload
        ]
    except Exception as exc:
        log.warning("Finnhub company news failed for %s: %s", symbol, exc)
        return []


def _fetch_yahoo_history(symbol: str) -> pd.DataFrame:
    try:
        history = yf.Ticker(symbol).history(period=YAHOO_HISTORY_PERIOD, auto_adjust=False)
    except Exception as exc:
        log.warning("Yahoo history failed for %s: %s", symbol, exc)
        return pd.DataFrame()
    return prepare_price_history_frame(history)


def _fetch_yahoo_news(symbol: str) -> list[dict]:
    try:
        news_items = yf.Ticker(symbol).news or []
    except Exception as exc:
        log.warning("Yahoo news failed for %s: %s", symbol, exc)
        return []

    normalized: list[dict] = []
    for item in news_items:
        if not isinstance(item, dict):
            continue

        content = item.get("content")
        content_title = content.get("title") if isinstance(content, dict) else None
        title = item.get("title") or content_title
        if not isinstance(title, str) or not title.strip():
            continue

        published_at = item.get("providerPublishTime")
        if published_at is None and isinstance(content, dict):
            published_at = content.get("pubDate")

        provider_name = item.get("publisher")
        if provider_name is None and isinstance(content, dict):
            provider_name = content.get("provider", {}).get("displayName")

        url = item.get("link")
        if url is None and isinstance(content, dict):
            canonical = content.get("canonicalUrl")
            if isinstance(canonical, dict):
                url = canonical.get("url")

        normalized.append(
            {
                "headline": title.strip(),
                "source": provider_name,
                "provider": "yahoo",
                "url": url,
                "published_at": published_at,
            }
        )

    return normalized


def _normalize_currency_code(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    normalized = value.strip().upper()
    return normalized if normalized else None


def _normalize_company_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    normalized = value.strip()
    return normalized if normalized else None


def _fetch_yahoo_currency(symbol: str) -> str | None:
    try:
        fast_info = yf.Ticker(symbol).fast_info
        currency = fast_info.get("currency") if hasattr(fast_info, "get") else fast_info["currency"]
    except Exception as exc:
        log.warning("Yahoo currency lookup failed for %s: %s", symbol, exc)
        return None

    return _normalize_currency_code(currency)


def _fetch_yahoo_company_name(symbol: str) -> str | None:
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as exc:
        log.warning("Yahoo company name lookup failed for %s: %s", symbol, exc)
        return None

    return _normalize_company_name(
        info.get("longName") or info.get("shortName") or info.get("displayName")
    )


def _history_quote_traces(history: pd.DataFrame) -> dict[str, dict]:
    if history.empty:
        return {}

    working = history.sort_values("date").reset_index(drop=True)
    latest = working.iloc[-1]
    previous = working.iloc[-2] if len(working) > 1 else latest

    close = working["close"].astype(float)
    high = working["high"].astype(float)
    low = working["low"].astype(float)
    volume = working["volume"].astype(float)

    avg_volume_10d = volume.rolling(10, min_periods=3).mean().iloc[-1]
    avg_volume_3m = volume.rolling(63, min_periods=10).mean().iloc[-1]
    fifty_day_avg = close.rolling(50, min_periods=10).mean().iloc[-1]
    two_hundred_day_avg = close.rolling(200, min_periods=30).mean().iloc[-1]
    fifty_two_week_high = high.rolling(252, min_periods=30).max().iloc[-1]
    fifty_two_week_low = low.rolling(252, min_periods=30).min().iloc[-1]

    return {
        "current_price": _trace(latest["close"], "yahoo", "history.close"),
        "previous_close": _trace(previous["close"], "yahoo", "history.previous_close"),
        "day_open": _trace(latest["open"], "yahoo", "history.open"),
        "day_high": _trace(latest["high"], "yahoo", "history.high"),
        "day_low": _trace(latest["low"], "yahoo", "history.low"),
        "volume": _trace(latest["volume"], "yahoo", "history.volume"),
        "avg_volume_10d": _trace(avg_volume_10d, "yahoo", "history.avg_volume_10d"),
        "avg_volume_3m": _trace(avg_volume_3m, "yahoo", "history.avg_volume_3m"),
        "fifty_day_avg": _trace(fifty_day_avg, "yahoo", "history.fifty_day_avg"),
        "two_hundred_day_avg": _trace(two_hundred_day_avg, "yahoo", "history.two_hundred_day_avg"),
        "fifty_two_week_high": _trace(fifty_two_week_high, "yahoo", "history.fifty_two_week_high"),
        "fifty_two_week_low": _trace(fifty_two_week_low, "yahoo", "history.fifty_two_week_low"),
    }


def _finnhub_quote_traces(quote: dict) -> dict[str, dict]:
    if not quote:
        return {}

    return {
        "current_price": _trace(quote.get("c"), "finnhub", "quote.c"),
        "previous_close": _trace(quote.get("pc"), "finnhub", "quote.pc"),
        "day_open": _trace(quote.get("o"), "finnhub", "quote.o"),
        "day_high": _trace(quote.get("h"), "finnhub", "quote.h"),
        "day_low": _trace(quote.get("l"), "finnhub", "quote.l"),
    }


def build_symbol_snapshot(
    symbol: str,
    *,
    finnhub_key: str | None,
    include_news: bool = False,
    include_currency: bool = False,
    include_company_name: bool = False,
    prefer_finnhub_quote: bool = True,
    extra_headlines: list[str] | None = None,
) -> dict:
    finnhub_quote = _fetch_finnhub_quote(symbol, finnhub_key if prefer_finnhub_quote else None)
    yahoo_history = _fetch_yahoo_history(symbol)
    quote_currency = _fetch_yahoo_currency(symbol) if include_currency else None
    company_name = _fetch_yahoo_company_name(symbol) if include_company_name else None

    finnhub_traces = _finnhub_quote_traces(finnhub_quote)
    history_traces = _history_quote_traces(yahoo_history)

    quote = {
        "current_price": _pick_trace(finnhub_traces.get("current_price"), history_traces.get("current_price"))
        if prefer_finnhub_quote
        else _pick_trace(history_traces.get("current_price"), finnhub_traces.get("current_price")),
        "previous_close": _pick_trace(finnhub_traces.get("previous_close"), history_traces.get("previous_close"))
        if prefer_finnhub_quote
        else _pick_trace(history_traces.get("previous_close"), finnhub_traces.get("previous_close")),
        "day_open": _pick_trace(finnhub_traces.get("day_open"), history_traces.get("day_open"))
        if prefer_finnhub_quote
        else _pick_trace(history_traces.get("day_open"), finnhub_traces.get("day_open")),
        "day_high": _pick_trace(finnhub_traces.get("day_high"), history_traces.get("day_high"))
        if prefer_finnhub_quote
        else _pick_trace(history_traces.get("day_high"), finnhub_traces.get("day_high")),
        "day_low": _pick_trace(finnhub_traces.get("day_low"), history_traces.get("day_low"))
        if prefer_finnhub_quote
        else _pick_trace(history_traces.get("day_low"), finnhub_traces.get("day_low")),
        "volume": _pick_trace(history_traces.get("volume")),
        "avg_volume_10d": _pick_trace(history_traces.get("avg_volume_10d")),
        "avg_volume_3m": _pick_trace(history_traces.get("avg_volume_3m")),
        "fifty_day_avg": _pick_trace(history_traces.get("fifty_day_avg")),
        "two_hundred_day_avg": _pick_trace(history_traces.get("two_hundred_day_avg")),
        "fifty_two_week_high": _pick_trace(history_traces.get("fifty_two_week_high")),
        "fifty_two_week_low": _pick_trace(history_traces.get("fifty_two_week_low")),
    }

    news: list[dict] = []
    if include_news:
        news.extend(_fetch_finnhub_company_news(symbol, finnhub_key))
        news.extend(_fetch_yahoo_news(symbol))
        for headline in extra_headlines or []:
            news.append(
                {
                    "headline": headline,
                    "source": "User Input",
                    "provider": "request",
                    "url": None,
                    "published_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    return {
        "symbol": symbol,
        "company_name": company_name,
        "quote_currency": quote_currency,
        "quote": quote,
        "news": news,
        "history_rows": int(len(yahoo_history)),
    }


def fetch_live_snapshot(
    ticker: str,
    *,
    finnhub_key: str | None,
    extra_headlines: list[str] | None = None,
) -> dict:
    ticker_snapshot = build_symbol_snapshot(
        ticker,
        finnhub_key=finnhub_key,
        include_news=True,
        include_currency=True,
        include_company_name=True,
        prefer_finnhub_quote=True,
        extra_headlines=extra_headlines,
    )
    spy_snapshot = build_symbol_snapshot(
        "SPY",
        finnhub_key=finnhub_key,
        include_news=False,
        prefer_finnhub_quote=True,
    )
    vix_snapshot = build_symbol_snapshot(
        "^VIX",
        finnhub_key=None,
        include_news=False,
        prefer_finnhub_quote=False,
    )

    return {
        "ticker": ticker,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "company_name": ticker_snapshot.get("company_name"),
        "quote_currency": ticker_snapshot.get("quote_currency"),
        "quote": ticker_snapshot["quote"],
        "news": ticker_snapshot["news"],
        "market_context": {
            "spy": spy_snapshot["quote"],
            "vix": vix_snapshot["quote"],
        },
        "source_health": {
            "ticker_history_rows": ticker_snapshot["history_rows"],
            "spy_history_rows": spy_snapshot["history_rows"],
            "vix_history_rows": vix_snapshot["history_rows"],
        },
    }
