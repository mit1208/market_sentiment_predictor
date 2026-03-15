import pandas as pd
import pytest

from backend.ml.training.context_validation import validate_context_history_coverage


def test_validate_context_history_coverage_raises_when_spy_history_is_too_short():
    training_market_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "ticker": ["AAPL", "AAPL", "AAPL"],
        }
    )
    short_spy_history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-10", "2026-03-11", "2026-03-12"]),
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    full_vix_history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )

    with pytest.raises(RuntimeError) as exc_info:
        validate_context_history_coverage(
            training_market_df,
            spy_history=short_spy_history,
            vix_history=full_vix_history,
        )

    assert "context coverage is incomplete for SPY" in str(exc_info.value)
