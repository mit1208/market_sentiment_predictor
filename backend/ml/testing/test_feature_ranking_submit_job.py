import importlib
import sys
import types

import pandas as pd
import pytest


def _load_submit_job_module():
    fake_automl = types.ModuleType("backend.ml.training.automl_selection")
    fake_automl.submit_or_reuse_automl_feature_ranking = lambda *args, **kwargs: {}

    fake_train = types.ModuleType("backend.ml.training.components.train")
    fake_train.MODEL_NAME = "market_sentiment_lgbm"
    fake_train.prepare_candidate_dataset = lambda rebuild_snapshots=True: (pd.DataFrame(), {})

    sys.modules["backend.ml.training.automl_selection"] = fake_automl
    sys.modules["backend.ml.training.components.train"] = fake_train

    module = importlib.import_module("backend.ml.training.feature_ranking_submit_job")
    return importlib.reload(module)


def test_apply_automl_sample_limit_keeps_latest_rows():
    submit_job = _load_submit_job_module()
    dataset = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03", "2026-01-03"]
            ),
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
            "next_day_return": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        }
    )

    sampled = submit_job._apply_automl_sample_limit(dataset, sample_rows=4, min_rows=4)

    assert len(sampled) == 4
    assert sampled["date"].min() == pd.Timestamp("2026-01-02")
    assert sampled["date"].max() == pd.Timestamp("2026-01-03")


def test_apply_automl_sample_limit_rejects_invalid_small_sample():
    submit_job = _load_submit_job_module()
    dataset = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "ticker": ["AAPL", "MSFT"],
            "next_day_return": [0.01, 0.02],
        }
    )

    with pytest.raises(RuntimeError, match="AUTO_ML_SAMPLE_ROWS must be at least 1000"):
        submit_job._apply_automl_sample_limit(dataset, sample_rows=100, min_rows=1000)
