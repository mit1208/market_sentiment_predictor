#!/usr/bin/env python3
"""Send one live /predict request and print a compact response summary."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _resolve_predict_url(raw_url: str) -> str:
    url = raw_url.strip()
    if not url:
        raise ValueError("PREDICT_API_URL or --url is required.")
    if url.endswith("/predict"):
        return url
    return f"{url.rstrip('/')}/predict"


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("PREDICT_API_URL", ""))
    parser.add_argument("--ticker", default=os.getenv("TICKER", "AAPL"))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--raw", action="store_true", help="Print the full JSON response.")
    args = parser.parse_args()

    try:
        predict_url = _resolve_predict_url(args.url)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = json.dumps(
        {
            "instances": [
                {
                    "ticker": args.ticker.strip().upper(),
                }
            ]
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        predict_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}", file=sys.stderr)
        print(body, file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 1

    predictions = data.get("predictions") or []
    if not predictions:
        print(json.dumps(data, indent=2))
        return 0

    first = predictions[0]
    feature_snapshot = first.get("feature_snapshot") or {}
    groups = feature_snapshot.get("groups") or []
    news_items = feature_snapshot.get("news_items") or []
    provider_summary = feature_snapshot.get("provider_summary") or {}
    feature_count = feature_snapshot.get("feature_count")
    if feature_count is None:
        feature_count = sum(len(group.get("features") or []) for group in groups)

    print(f"url: {predict_url}")
    print(f"ticker: {first.get('ticker')}")
    print(f"prediction: {first.get('prediction')}")
    print(f"confidence: {first.get('confidence')}")
    print(f"prob_up: {first.get('prob_up')}")
    print(f"as_of: {first.get('as_of')}")
    print(f"model_trained_at: {first.get('model_trained_at')}")
    print(f"feature_count: {feature_count}")
    print(f"feature_groups: {len(groups)}")
    print(f"news_count: {len(news_items)}")
    print(f"provider_summary: {json.dumps(provider_summary, separators=(',', ':'))}")
    if news_items:
        print("sample_headlines:")
        for item in news_items[:3]:
            print(f"- {item.get('headline')}")

    if args.raw:
        print("response_json:")
        print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
