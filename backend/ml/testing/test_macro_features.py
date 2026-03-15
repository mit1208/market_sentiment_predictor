import pandas as pd

from backend.ml.shared.feature_engineering import MACRO_FEATURE_NAMES
from backend.ml.shared.macro_features import build_macro_history_from_raw, build_training_macro_feature_frame


def test_build_macro_history_from_raw_pivots_bronze_macro_rows():
    dates = pd.date_range("2026-01-01", periods=70, freq="B")
    rows = []
    for index, date in enumerate(dates):
        rows.extend(
            [
                {"observation_date": date, "series_id": "T10Y2Y", "value": 1.0 + index * 0.01},
                {"observation_date": date, "series_id": "FEDFUNDS", "value": 4.0 + index * 0.005},
                {"observation_date": date, "series_id": "BAMLH0A0HYM2", "value": 3.0 + index * 0.02},
                {"observation_date": date, "series_id": "DTWEXBGS", "value": 100.0 + index},
            ]
        )

    history = build_macro_history_from_raw(pd.DataFrame(rows))

    assert list(history.columns) == ["date", "yield_curve_spread", "fed_funds_rate", "fed_funds_change_3m", "credit_spread", "dxy_change_5d"]
    assert history["yield_curve_spread"].iloc[-1] > history["yield_curve_spread"].iloc[0]
    assert history["fed_funds_change_3m"].notna().any()
    assert history["dxy_change_5d"].notna().any()


def test_build_training_macro_feature_frame_uses_bronze_macro_and_breadth():
    dates = pd.date_range("2026-01-01", periods=70, freq="B")
    macro_rows = []
    market_rows = []
    for index, date in enumerate(dates):
        macro_rows.extend(
            [
                {"observation_date": date, "series_id": "T10Y2Y", "value": 1.0 + index * 0.01},
                {"observation_date": date, "series_id": "FEDFUNDS", "value": 4.0 + index * 0.005},
                {"observation_date": date, "series_id": "BAMLH0A0HYM2", "value": 3.0 + index * 0.02},
                {"observation_date": date, "series_id": "DTWEXBGS", "value": 100.0 + index},
            ]
        )
        market_rows.extend(
            [
                {
                    "date": date,
                    "ticker": "AAPL",
                    "open": 100 + index,
                    "high": 101 + index,
                    "low": 99 + index,
                    "close": 100 + index,
                    "volume": 1_000_000 + index,
                },
                {
                    "date": date,
                    "ticker": "MSFT",
                    "open": 200 - index,
                    "high": 201 - index,
                    "low": 199 - index,
                    "close": 200 - index,
                    "volume": 900_000 + index,
                },
            ]
        )

    feature_frame = build_training_macro_feature_frame(
        dates[-3:],
        market_df=pd.DataFrame(market_rows),
        raw_macro_df=pd.DataFrame(macro_rows),
        active_tickers=["AAPL", "MSFT"],
    )

    assert list(feature_frame.columns) == MACRO_FEATURE_NAMES
    assert len(feature_frame) == 3
    assert feature_frame["yield_curve_spread"].notna().all()
    assert feature_frame["universe_breadth_50d"].between(0.0, 1.0).all()
