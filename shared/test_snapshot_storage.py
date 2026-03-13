from shared.snapshot_storage import build_live_feature_snapshot_row


def test_build_live_feature_snapshot_row_includes_quote_currency():
    row = build_live_feature_snapshot_row(
        snapshot_id="online:SHOP.TO:20260313T211500Z",
        request_id="req:SHOP.TO:20260313T211609Z",
        ticker="SHOP.TO",
        snapshot_timestamp="2026-03-13T21:15:00Z",
        headlines_text="Shopify shares rise on earnings beat",
        base_feature_values={},
        snapshot_source="endpoint_live_api",
        quote_currency="cad",
    )

    assert row["quote_currency"] == "CAD"
