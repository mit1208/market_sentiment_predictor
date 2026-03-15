import sys
import types

google_module = sys.modules.setdefault("google", types.ModuleType("google"))
cloud_module = sys.modules.setdefault("google.cloud", types.ModuleType("google.cloud"))
bigquery_module = sys.modules.setdefault("google.cloud.bigquery", types.ModuleType("google.cloud.bigquery"))
storage_module = sys.modules.setdefault("google.cloud.storage", types.ModuleType("google.cloud.storage"))
setattr(bigquery_module, "Client", object)
setattr(storage_module, "Client", object)
setattr(google_module, "cloud", cloud_module)
setattr(cloud_module, "bigquery", bigquery_module)
setattr(cloud_module, "storage", storage_module)

from backend.ml.shared.feature_engineering import GEMINI_FEATURE_NAMES, TEMPORARY_SELECTED_FEATURE_COUNT
from backend.ml.training import automl_selection


def test_load_selected_feature_contract_can_fall_back_to_temporary_subset(monkeypatch):
    monkeypatch.setattr(automl_selection.bigquery, "Client", lambda project=None: object())
    monkeypatch.setattr(automl_selection, "_feature_selection_table", lambda **_: "proj.ds.feature_selection_runs")
    monkeypatch.setattr(automl_selection, "_load_bigquery_selected_feature_contract", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        automl_selection,
        "_load_local_selected_feature_contract",
        lambda include_temporary=False: (
            automl_selection.build_temporary_selected_feature_contract() if include_temporary else None
        ),
    )

    contract = automl_selection.load_selected_feature_contract(
        project_id="proj",
        dataset_id="ds",
        require_approved=True,
        allow_temporary_fallback=True,
    )

    assert contract is not None
    assert contract["is_temporary"] is True
    assert contract["selected_feature_count"] == TEMPORARY_SELECTED_FEATURE_COUNT
    assert contract["selected_feature_names"][: len(GEMINI_FEATURE_NAMES)] == GEMINI_FEATURE_NAMES


def test_load_selected_feature_contract_does_not_use_temporary_subset_by_default(monkeypatch):
    monkeypatch.setattr(automl_selection.bigquery, "Client", lambda project=None: object())
    monkeypatch.setattr(automl_selection, "_feature_selection_table", lambda **_: "proj.ds.feature_selection_runs")
    monkeypatch.setattr(automl_selection, "_load_bigquery_selected_feature_contract", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        automl_selection,
        "_load_local_selected_feature_contract",
        lambda include_temporary=False: (
            automl_selection.build_temporary_selected_feature_contract() if include_temporary else None
        ),
    )

    contract = automl_selection.load_selected_feature_contract(
        project_id="proj",
        dataset_id="ds",
        require_approved=True,
    )

    assert contract is None


def test_load_selected_feature_contract_prefers_bigquery_contract_over_temporary_subset(monkeypatch):
    approved_contract = {
        "run_id": "automl-approval-1",
        "candidate_feature_version": "cand-v1",
        "candidate_feature_count": 54,
        "selected_feature_names": ["gap_pct", "gemini_sentiment_avg"],
        "selected_feature_count": 2,
        "selection_source": "vertex_automl_tabular_async_finalize",
        "review_status": automl_selection.REVIEW_STATUS_APPROVED,
        "execution_status": automl_selection.EXECUTION_STATUS_COMPLETED,
    }
    mirrored = []

    monkeypatch.setattr(automl_selection.bigquery, "Client", lambda project=None: object())
    monkeypatch.setattr(automl_selection, "_feature_selection_table", lambda **_: "proj.ds.feature_selection_runs")
    monkeypatch.setattr(
        automl_selection,
        "_load_bigquery_selected_feature_contract",
        lambda *args, **kwargs: dict(approved_contract),
    )
    monkeypatch.setattr(
        automl_selection,
        "_load_local_selected_feature_contract",
        lambda include_temporary=False: (
            automl_selection.build_temporary_selected_feature_contract() if include_temporary else None
        ),
    )
    monkeypatch.setattr(
        automl_selection,
        "_write_local_selected_feature_contract",
        lambda contract: mirrored.append(contract),
    )

    contract = automl_selection.load_selected_feature_contract(
        project_id="proj",
        dataset_id="ds",
        require_approved=True,
        allow_temporary_fallback=True,
    )

    assert contract["run_id"] == approved_contract["run_id"]
    assert contract["selected_feature_names"] == approved_contract["selected_feature_names"]
    assert contract["is_temporary"] is False
    assert mirrored and mirrored[0]["run_id"] == approved_contract["run_id"]


def test_load_selected_feature_contract_requires_approved_local_contract(monkeypatch):
    pending_local_contract = {
        "run_id": "automl-pending-1",
        "candidate_feature_version": "cand-v1",
        "candidate_feature_count": 54,
        "selected_feature_names": ["gap_pct", "day_change_pct"],
        "selected_feature_count": 2,
        "selection_source": "vertex_automl_tabular_async_finalize",
        "review_status": automl_selection.REVIEW_STATUS_PENDING,
        "execution_status": automl_selection.EXECUTION_STATUS_COMPLETED,
    }

    monkeypatch.setattr(automl_selection.bigquery, "Client", lambda project=None: object())
    monkeypatch.setattr(automl_selection, "_feature_selection_table", lambda **_: "proj.ds.feature_selection_runs")
    monkeypatch.setattr(automl_selection, "_load_bigquery_selected_feature_contract", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        automl_selection,
        "_load_local_selected_feature_contract",
        lambda include_temporary=False: dict(pending_local_contract),
    )

    contract = automl_selection.load_selected_feature_contract(
        project_id="proj",
        dataset_id="ds",
        require_approved=True,
        allow_temporary_fallback=False,
    )

    assert contract is None


def test_extract_feature_importance_supports_vertex_model_explanation_on_evaluation_resource():
    class _Attribution:
        def __init__(self, mapping):
            self.feature_attributions = mapping

    class _Explanation:
        def __init__(self, mapping):
            self.mean_attributions = [_Attribution(mapping)]

    class _Resource:
        def __init__(self, mapping):
            self.metrics = types.SimpleNamespace(model_explanation=None)
            self.model_explanation = _Explanation(mapping)

    class _Evaluation:
        def __init__(self, mapping):
            self._gca_resource = _Resource(mapping)

    class _Model:
        def list_model_evaluations(self):
            return [
                _Evaluation(
                    {
                        "credit_spread": 0.057,
                        "yield_curve_spread": -0.051,
                        "not_a_candidate": 0.99,
                    }
                )
            ]

    feature_importance = automl_selection._extract_feature_importance(_Model())

    assert feature_importance[:2] == [
        {"feature": "credit_spread", "importance": 0.057},
        {"feature": "yield_curve_spread", "importance": 0.051},
    ]
