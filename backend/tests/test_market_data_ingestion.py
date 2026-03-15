import sys
import types

import pandas as pd

from backend.ingestion import main as ingestion_main


def test_run_ingestion_loads_bronze_tables_without_triggering_training(monkeypatch):
    calls = []
    market_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-03-13").date(),
                "ticker": "AAPL",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 123456,
            }
        ]
    )
    news_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-03-13").date(),
                "timestamp": pd.Timestamp("2026-03-13T20:30:00Z"),
                "ticker": "AAPL",
                "headline": "Apple launches new feature",
                "description": None,
                "source": "Reuters",
                "url": "https://example.com/news/apple",
            }
        ]
    )
    macro_df = pd.DataFrame(
        [
            {
                "observation_date": pd.Timestamp("2026-03-13").date(),
                "series_id": "FEDFUNDS",
                "value": 4.5,
                "source": "FRED",
                "fetched_at": pd.Timestamp("2026-03-14T00:00:00Z"),
            }
        ]
    )

    def fake_table_has_rows(table_name):
        return table_name == "raw_news_data"

    def fake_fetch_market_data_df(tickers, *, bootstrap=False, start_date_override=None):
        calls.append(("market", tuple(tickers), bootstrap, start_date_override))
        return market_df

    def fake_fetch_news_data_df(tickers, *, bootstrap=False):
        calls.append(("news", tuple(tickers), bootstrap))
        return news_df

    def fake_fetch_macro_data_df(*, bootstrap=False):
        calls.append(("macro", bootstrap))
        return macro_df

    def fake_load_with_merge(df, table_name, unique_keys):
        calls.append(("merge", table_name, tuple(unique_keys), len(df)))

    monkeypatch.setattr(ingestion_main, "table_has_rows", fake_table_has_rows)
    monkeypatch.setattr(ingestion_main, "fetch_market_data_df", fake_fetch_market_data_df)
    monkeypatch.setattr(ingestion_main, "fetch_news_data_df", fake_fetch_news_data_df)
    monkeypatch.setattr(ingestion_main, "fetch_macro_data_df", fake_fetch_macro_data_df)
    monkeypatch.setattr(ingestion_main, "load_with_merge", fake_load_with_merge)
    monkeypatch.setattr(ingestion_main, "TICKERS", ["AAPL"])

    response, status_code = ingestion_main.run_ingestion()

    assert status_code == 200
    assert response == {
        "status": "success",
        "market_rows_loaded": 1,
        "news_rows_loaded": 1,
        "macro_rows_loaded": 1,
        "training_tickers": ["AAPL"],
        "market_universe_tickers": ["AAPL", "SPY", "^VIX"],
    }
    assert calls == [
        ("market", ("AAPL", "SPY", "^VIX"), True, None),
        ("merge", "raw_market_data", ("date", "ticker"), 1),
        ("news", tuple(ingestion_main.TICKERS), False),
        ("merge", "raw_news_data", ("url", "ticker"), 1),
        ("macro", True),
        ("merge", "raw_macro_data", ("observation_date", "series_id"), 1),
    ]


def test_run_ingestion_backfills_context_history_when_incomplete(monkeypatch):
    calls = []
    market_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-03-13").date(),
                "ticker": "AAPL",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 123456,
            }
        ]
    )
    context_backfill_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-02").date(),
                "ticker": "SPY",
                "open": 320.0,
                "high": 321.0,
                "low": 319.0,
                "close": 320.5,
                "volume": 987654,
            },
            {
                "date": pd.Timestamp("2020-01-02").date(),
                "ticker": "^VIX",
                "open": 14.0,
                "high": 14.5,
                "low": 13.5,
                "close": 14.2,
                "volume": 0,
            },
        ]
    )

    def fake_table_has_rows(_table_name):
        return True

    def fake_fetch_market_data_df(tickers, *, bootstrap=False, start_date_override=None):
        calls.append(("market", tuple(tickers), bootstrap, start_date_override))
        return context_backfill_df if bootstrap else market_df

    def fake_fetch_news_data_df(tickers, *, bootstrap=False):
        calls.append(("news", tuple(tickers), bootstrap))
        return pd.DataFrame()

    def fake_fetch_macro_data_df(*, bootstrap=False):
        calls.append(("macro", bootstrap))
        return pd.DataFrame()

    def fake_load_with_merge(df, table_name, unique_keys):
        calls.append(("merge", table_name, tuple(unique_keys), len(df)))

    monkeypatch.setattr(ingestion_main, "table_has_rows", fake_table_has_rows)
    monkeypatch.setattr(ingestion_main, "fetch_market_data_df", fake_fetch_market_data_df)
    monkeypatch.setattr(ingestion_main, "fetch_news_data_df", fake_fetch_news_data_df)
    monkeypatch.setattr(ingestion_main, "fetch_macro_data_df", fake_fetch_macro_data_df)
    monkeypatch.setattr(ingestion_main, "load_with_merge", fake_load_with_merge)
    monkeypatch.setattr(ingestion_main, "_resolve_context_backfill_start", lambda tickers: pd.Timestamp("2020-01-02"))
    monkeypatch.setattr(ingestion_main, "_context_history_requires_backfill", lambda required_start: True)
    monkeypatch.setattr(ingestion_main, "TICKERS", ["AAPL"])

    response, status_code = ingestion_main.run_ingestion()

    assert status_code == 200
    assert response["market_rows_loaded"] == 3
    assert calls == [
        ("market", ("AAPL", "SPY", "^VIX"), False, None),
        ("merge", "raw_market_data", ("date", "ticker"), 1),
        ("market", ("SPY", "^VIX"), True, pd.Timestamp("2020-01-02")),
        ("merge", "raw_market_data", ("date", "ticker"), 2),
        ("news", ("AAPL",), False),
        ("macro", False),
    ]


def test_trigger_training_job_builds_expected_cloud_run_request(monkeypatch):
    captured = {}

    google_module = types.ModuleType("google")
    api_core_module = types.ModuleType("google.api_core")
    client_options_module = types.ModuleType("google.api_core.client_options")
    cloud_module = types.ModuleType("google.cloud")
    run_v2_module = types.ModuleType("google.cloud.run_v2")

    class FakeClientOptions:
        def __init__(self, api_endpoint):
            captured["api_endpoint"] = api_endpoint

    class FakeRunJobRequest:
        def __init__(self, name):
            self.name = name

    class FakeOperation:
        operation = types.SimpleNamespace(name="operations/run-456")

    class FakeJobsClient:
        def __init__(self, client_options):
            captured["client_options_seen"] = "api_endpoint" in captured
            captured["client_options_type"] = type(client_options).__name__

        def run_job(self, request):
            captured["request_name"] = request.name
            return FakeOperation()

    client_options_module.ClientOptions = FakeClientOptions
    run_v2_module.JobsClient = FakeJobsClient
    run_v2_module.RunJobRequest = FakeRunJobRequest

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.api_core", api_core_module)
    monkeypatch.setitem(sys.modules, "google.api_core.client_options", client_options_module)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_module)
    monkeypatch.setitem(sys.modules, "google.cloud.run_v2", run_v2_module)
    monkeypatch.setattr(ingestion_main, "PROJECT_ID", "demo-project")
    monkeypatch.setattr(ingestion_main, "REGION", "us-central1")
    monkeypatch.setenv("TRAINING_JOB_NAME", "market-sentiment-train")

    operation_name = ingestion_main.trigger_training_job()

    assert operation_name == "operations/run-456"
    assert captured["api_endpoint"] == "us-central1-run.googleapis.com"
    assert captured["client_options_seen"] is True
    assert captured["client_options_type"] == "FakeClientOptions"
    assert captured["request_name"] == (
        "projects/demo-project/locations/us-central1/jobs/market-sentiment-train"
    )
