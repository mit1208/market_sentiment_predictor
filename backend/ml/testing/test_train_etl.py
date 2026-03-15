import pandas as pd
import pytest

from backend.ml.training.components import train as train_module


def test_date_split_indices_use_positional_offsets_after_holdout_split():
    gold = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "ticker": ["AAPL"] * 20,
            "target": [0, 1] * 10,
            "next_day_return": [0.01] * 20,
            "current_price": [100.0 + idx for idx in range(20)],
        },
        index=range(100, 120),
    )

    train_df, _holdout_df, _holdout_start = train_module._chronological_holdout_split(gold)

    for _fold, train_idx, val_idx, _train_dates, _val_dates in train_module._date_split_indices(train_df, 3):
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert max(train_idx) < len(train_df)
        assert max(val_idx) < len(train_df)
        train_df.iloc[train_idx]
        train_df.iloc[val_idx]


def test_run_gold_training_etl_materializes_gold_before_training(monkeypatch):
    silver = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["AAPL", "AAPL"],
            "target": [1, 0],
            "next_day_return": [0.02, -0.01],
            "current_price": [101.0, 100.0],
        }
    )
    gold = silver.copy()
    feature_contract = {
        "run_id": "fr-20260314222328-approved",
        "candidate_feature_version": "2026-03-14-fast-accuracy-pass-v1",
        "selected_feature_count": 1,
        "selected_feature_names": ["current_price"],
        "selection_source": "vertex_automl_tabular_async_finalize",
        "feature_selection_mode": "automl_top_features",
        "review_status": "APPROVED",
        "is_temporary": False,
    }
    materialize_calls = []
    load_contract_calls = []

    def fake_load_selected_feature_contract(**kwargs):
        load_contract_calls.append(dict(kwargs))
        return feature_contract

    monkeypatch.setattr(train_module, "load_selected_feature_contract", fake_load_selected_feature_contract)
    monkeypatch.setattr(
        train_module,
        "prepare_candidate_dataset",
        lambda rebuild_snapshots=False: (silver, {"snapshot_rows": len(silver), "snapshot_table": "silver_table"}),
    )

    def fake_materialize(silver_df, *, feature_contract):
        materialize_calls.append((silver_df.copy(), dict(feature_contract)))
        return {
            "gold_materialization_id": "gold:test",
            "gold_rows": len(silver_df),
            "gold_table": "gold_table",
            "feature_selection_run_id": feature_contract["run_id"],
            "selected_feature_count": len(feature_contract["selected_feature_names"]),
        }

    monkeypatch.setattr(train_module, "materialize_gold_training_features", fake_materialize)
    monkeypatch.setattr(train_module, "read_gold_training_features", lambda contract: gold[["date", "ticker", "target", "next_day_return", "current_price"]].copy())
    monkeypatch.setattr(
        train_module,
        "build_gold_metadata",
        lambda gold_df, *, feature_contract, silver_metadata=None: {
            "training_start_date": str(pd.Timestamp(gold_df["date"].min()).date()),
            "training_end_date": str(pd.Timestamp(gold_df["date"].max()).date()),
            "snapshot_rows": len(gold_df),
            "snapshot_table": "silver_table",
            "gold_table": "gold_table",
            "selected_feature_count": len(feature_contract["selected_feature_names"]),
            "feature_selection_run_id": feature_contract["run_id"],
            "feature_selection_source": feature_contract["selection_source"],
            "feature_selection_is_temporary": feature_contract["is_temporary"],
        },
    )

    gold_df, metadata = train_module.run_gold_training_etl(rebuild_gold=True)

    assert len(materialize_calls) == 1
    assert load_contract_calls == [
        {
            "project_id": train_module.PROJECT_ID,
            "dataset_id": train_module.DATASET_ID,
            "require_approved": True,
            "allow_temporary_fallback": False,
        }
    ]
    pd.testing.assert_frame_equal(materialize_calls[0][0], silver)
    assert materialize_calls[0][1]["run_id"] == feature_contract["run_id"]
    pd.testing.assert_frame_equal(gold_df.reset_index(drop=True), gold.reset_index(drop=True))
    assert metadata["training_rows"] == len(gold)
    assert metadata["data_layer"] == "gold"
    assert metadata["feature_selection_source"] == feature_contract["selection_source"]
    assert metadata["feature_selection_is_temporary"] is False


def test_load_prepared_gold_training_dataset_reads_existing_gold_only(monkeypatch):
    gold = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["AAPL", "AAPL"],
            "target": [1, 0],
            "next_day_return": [0.02, -0.01],
            "current_price": [101.0, 100.0],
        }
    )
    feature_contract = {
        "run_id": "fr-20260314222328-approved",
        "candidate_feature_version": "2026-03-14-fast-accuracy-pass-v1",
        "selected_feature_count": 1,
        "selected_feature_names": ["current_price"],
        "selection_source": "vertex_automl_tabular_async_finalize",
        "feature_selection_mode": "automl_top_features",
        "review_status": "APPROVED",
        "is_temporary": False,
    }
    materialize_calls = []
    load_contract_calls = []

    def fake_load_selected_feature_contract(**kwargs):
        load_contract_calls.append(dict(kwargs))
        return feature_contract

    monkeypatch.setattr(train_module, "load_selected_feature_contract", fake_load_selected_feature_contract)
    monkeypatch.setattr(
        train_module,
        "prepare_candidate_dataset",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("prepare_candidate_dataset should not be called")),
    )
    monkeypatch.setattr(
        train_module,
        "materialize_gold_training_features",
        lambda *_args, **_kwargs: materialize_calls.append("called"),
    )
    monkeypatch.setattr(
        train_module,
        "read_gold_training_features",
        lambda contract: gold[["date", "ticker", "target", "next_day_return", "current_price"]].copy(),
    )
    monkeypatch.setattr(
        train_module,
        "build_gold_metadata",
        lambda gold_df, *, feature_contract, silver_metadata=None: {
            "training_start_date": str(pd.Timestamp(gold_df["date"].min()).date()),
            "training_end_date": str(pd.Timestamp(gold_df["date"].max()).date()),
            "snapshot_rows": len(gold_df),
            "snapshot_table": "silver_table",
            "gold_table": "gold_table",
            "selected_feature_count": len(feature_contract["selected_feature_names"]),
            "feature_selection_run_id": feature_contract["run_id"],
            "feature_selection_source": feature_contract["selection_source"],
            "feature_selection_is_temporary": feature_contract["is_temporary"],
        },
    )

    gold_df, metadata = train_module.load_prepared_gold_training_dataset()

    assert materialize_calls == []
    assert load_contract_calls == [
        {
            "project_id": train_module.PROJECT_ID,
            "dataset_id": train_module.DATASET_ID,
            "require_approved": True,
            "allow_temporary_fallback": False,
        }
    ]
    pd.testing.assert_frame_equal(gold_df.reset_index(drop=True), gold.reset_index(drop=True))
    assert metadata["training_rows"] == len(gold)
    assert metadata["data_layer"] == "gold"
    assert metadata["feature_selection_source"] == feature_contract["selection_source"]


def test_materialize_gold_training_features_validates_selected_feature_contract():
    silver = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "ticker": ["AAPL"],
            "target": [1],
            "next_day_return": [0.02],
            "current_price": [101.0],
        }
    )
    feature_contract = {
        "run_id": "fr-20260314222328-approved",
        "candidate_feature_version": "2026-03-14-fast-accuracy-pass-v1",
        "selected_feature_count": 2,
        "selected_feature_names": ["current_price", "unknown_feature"],
    }

    with pytest.raises(ValueError) as exc_info:
        train_module.materialize_gold_training_features(silver, feature_contract=feature_contract)

    assert str(exc_info.value) == "Unknown feature names in contract: ['unknown_feature']"


def test_select_prediction_strategy_prefers_stronger_lgb_signal():
    strategy = train_module._select_prediction_strategy(
        y_true=pd.Series([1, 1, 0, 0]),
        lgb_proba=[0.81, 0.76, 0.22, 0.18],
        xgb_proba=[0.58, 0.57, 0.53, 0.52],
    )

    assert strategy["prediction_strategy"] == "lightgbm_only"
    assert strategy["ensemble_lgb_weight"] == 1.0
    assert strategy["ensemble_xgb_weight"] == 0.0
    assert 0.4 <= strategy["classification_threshold"] <= 0.7


def test_select_prediction_strategy_tolerates_initial_uncovered_oof_rows():
    strategy = train_module._select_prediction_strategy(
        y_true=pd.Series([1, 1, 0, 0, 1]),
        lgb_proba=[float("nan"), 0.82, 0.24, 0.19, 0.79],
        xgb_proba=[float("nan"), 0.57, 0.53, 0.52, 0.56],
    )

    assert strategy["prediction_strategy"] == "lightgbm_only"
    assert strategy["oof_rows"] == 4
    assert strategy["oof_total_rows"] == 5
    assert strategy["oof_coverage"] == 0.8
