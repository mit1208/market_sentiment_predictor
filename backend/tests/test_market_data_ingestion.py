import sys
import types

import pandas as pd

from backend.ingestion import main as ingestion_main


def test_run_ingestion_loads_data_and_triggers_training(monkeypatch):
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

    def fake_table_has_rows(table_name):
        return table_name == "raw_news_data"

    def fake_fetch_market_data_df(tickers, *, bootstrap=False):
        calls.append(("market", tuple(tickers), bootstrap))
        return market_df

    def fake_fetch_news_data_df(tickers, *, bootstrap=False):
        calls.append(("news", tuple(tickers), bootstrap))
        return news_df

    def fake_load_with_merge(df, table_name, unique_keys):
        calls.append(("merge", table_name, tuple(unique_keys), len(df)))

    monkeypatch.setattr(ingestion_main, "table_has_rows", fake_table_has_rows)
    monkeypatch.setattr(ingestion_main, "fetch_market_data_df", fake_fetch_market_data_df)
    monkeypatch.setattr(ingestion_main, "fetch_news_data_df", fake_fetch_news_data_df)
    monkeypatch.setattr(ingestion_main, "load_with_merge", fake_load_with_merge)
    monkeypatch.setattr(ingestion_main, "trigger_training_job", lambda: "operations/train-123")
    monkeypatch.setattr(ingestion_main, "TICKERS", ["AAPL"])

    response, status_code = ingestion_main.run_ingestion()

    assert status_code == 200
    assert response == {"status": "success", "training_execution": "operations/train-123"}
    assert calls == [
        ("market", tuple(ingestion_main.TICKERS), True),
        ("merge", "raw_market_data", ("date", "ticker"), 1),
        ("news", tuple(ingestion_main.TICKERS), False),
        ("merge", "raw_news_data", ("url", "ticker"), 1),
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
