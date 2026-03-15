import numpy as np

from backend.ml.shared.snapshot_storage import build_live_feature_snapshot_row


def test_build_live_feature_snapshot_row_includes_quote_currency():
    row = build_live_feature_snapshot_row(
        snapshot_id="online:SHOP.TO:20260313T211500Z",
        request_id="req:SHOP.TO:20260313T211609Z",
        ticker="SHOP.TO",
        snapshot_timestamp="2026-03-13T21:15:00Z",
        headlines_text="Shopify shares rise on earnings beat",
        base_feature_values={},
        snapshot_source="endpoint_live_api",
        next_day_return=0.01,
        quote_currency="cad",
    )

    assert row["quote_currency"] == "CAD"
    assert row["next_day_return"] == 0.01


def test_build_live_feature_snapshot_row_coerces_numpy_scalars_and_drops_array_values():
    row = build_live_feature_snapshot_row(
        snapshot_id="online:AAPL:20260314T153000Z",
        ticker="AAPL",
        snapshot_timestamp="2026-03-14T15:30:00Z",
        headlines_text="Apple shares rise",
        base_feature_values={
            "current_price": np.array([101.5]),
            "volume": np.array([1_000_000, 1_100_000]),
        },
        snapshot_source="endpoint_live_api",
        next_day_return=np.array([0.02]),
    )

    assert row["current_price"] == 101.5
    assert row["volume"] is None
    assert row["next_day_return"] == 0.02
