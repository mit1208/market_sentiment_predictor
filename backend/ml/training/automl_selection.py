"""Vertex AI AutoML feature ranking helpers with async submit/finalize flow."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import bigquery, storage

from backend.config import qualify_table_name
from backend.ml.shared.feature_engineering import (
    CANDIDATE_FEATURE_NAMES,
    TEMPORARY_FEATURE_SELECTION_SOURCE,
    build_temporary_selected_feature_contract,
    candidate_feature_version,
)

log = logging.getLogger(__name__)

SELECTED_FEATURES_PATH = Path(__file__).resolve().parents[1] / "shared" / "selected_features.json"
DEFAULT_MAX_SELECTED_FEATURES = 30
DEFAULT_BUDGET_MILLI_NODE_HOURS = 1000

REVIEW_STATUS_NOT_READY = "NOT_READY"
REVIEW_STATUS_PENDING = "PENDING_REVIEW"
REVIEW_STATUS_APPROVED = "APPROVED"
REVIEW_STATUS_REJECTED = "REJECTED"

EXECUTION_STATUS_SUBMITTED = "SUBMITTED"
EXECUTION_STATUS_RUNNING = "RUNNING"
EXECUTION_STATUS_COMPLETED = "COMPLETED"
EXECUTION_STATUS_FAILED = "FAILED"
VERTEX_FAILURE_TOKENS = ("FAILED", "CANCELLED", "PAUSED")


def _feature_selection_table(
    *,
    project_id: str,
    dataset_id: str,
) -> str:
    return qualify_table_name(
        "feature_selection_runs",
        project_id=project_id or None,
        dataset_id=dataset_id or None,
    )


def _normalize_contract(payload: dict | None, *, allow_empty: bool = False) -> dict | None:
    if not payload:
        return None

    feature_names = [
        str(name)
        for name in (payload.get("selected_feature_names") or [])
        if str(name) in CANDIDATE_FEATURE_NAMES
    ]
    if not feature_names and not allow_empty:
        return None

    return {
        **payload,
        "candidate_feature_version": payload.get("candidate_feature_version") or candidate_feature_version(),
        "candidate_feature_count": int(payload.get("candidate_feature_count") or len(CANDIDATE_FEATURE_NAMES)),
        "selected_feature_names": feature_names,
        "selected_feature_count": int(payload.get("selected_feature_count") or len(feature_names)),
        "review_status": payload.get("review_status") or REVIEW_STATUS_NOT_READY,
        "execution_status": payload.get("execution_status") or EXECUTION_STATUS_SUBMITTED,
    }


def _is_temporary_contract(contract: dict | None) -> bool:
    if not contract:
        return False
    return bool(contract.get("is_temporary")) or (
        contract.get("selection_source") == TEMPORARY_FEATURE_SELECTION_SOURCE
    )


def _load_local_selected_feature_contract(*, include_temporary: bool = False) -> dict | None:
    if not SELECTED_FEATURES_PATH.exists():
        if include_temporary:
            return build_temporary_selected_feature_contract()
        return None
    try:
        contract = _normalize_contract(json.loads(SELECTED_FEATURES_PATH.read_text()))
    except Exception as exc:
        log.warning("Failed to load selected feature contract from %s: %s", SELECTED_FEATURES_PATH, exc)
        if include_temporary:
            return build_temporary_selected_feature_contract()
        return None
    if not contract:
        if include_temporary:
            return build_temporary_selected_feature_contract()
        return None
    contract["is_temporary"] = _is_temporary_contract(contract)
    if contract["is_temporary"] and not include_temporary:
        return None
    return contract


def _write_local_selected_feature_contract(contract: dict) -> None:
    try:
        SELECTED_FEATURES_PATH.write_text(json.dumps(contract, indent=2) + "\n")
    except Exception as exc:
        log.warning("Failed to write local selected feature contract mirror: %s", exc)


def _row_to_contract(row) -> dict | None:
    try:
        payload = json.loads(row.contract_json)
    except Exception as exc:
        log.warning("Failed to parse BigQuery feature contract payload: %s", exc)
        return None

    payload.update(
        {
            "run_id": row.run_id,
            "candidate_feature_version": row.candidate_feature_version,
            "candidate_feature_count": row.candidate_feature_count,
            "selected_feature_names": list(row.selected_feature_names or []),
            "selected_feature_count": row.selected_feature_count,
            "selection_source": row.selection_source,
            "review_status": row.review_status,
            "execution_status": getattr(row, "execution_status", None) or payload.get("execution_status"),
            "report_uri": getattr(row, "report_uri", None) or payload.get("report_uri"),
            "contract_uri": getattr(row, "contract_uri", None) or payload.get("contract_uri"),
            "generated_at": row.generated_at.isoformat() if getattr(row, "generated_at", None) else payload.get("generated_at"),
            "started_at": (
                row.started_at.isoformat()
                if getattr(row, "started_at", None)
                else payload.get("started_at")
            ),
            "completed_at": (
                row.completed_at.isoformat()
                if getattr(row, "completed_at", None)
                else payload.get("completed_at")
            ),
            "notes": getattr(row, "notes", None) or payload.get("notes"),
            "error_message": getattr(row, "error_message", None) or payload.get("error_message"),
        }
    )
    return _normalize_contract(payload, allow_empty=True)


def _load_bigquery_selected_feature_contract(
    client: bigquery.Client,
    *,
    table_name: str,
    require_approved: bool,
) -> dict | None:
    statuses = [REVIEW_STATUS_APPROVED] if require_approved else [REVIEW_STATUS_APPROVED, REVIEW_STATUS_PENDING]
    query = f"""
        SELECT
          run_id,
          candidate_feature_version,
          candidate_feature_count,
          selected_feature_names,
          selected_feature_count,
          selection_source,
          review_status,
          execution_status,
          report_uri,
          contract_uri,
          generated_at,
          started_at,
          completed_at,
          notes,
          error_message,
          contract_json
        FROM `{table_name}`
        WHERE candidate_feature_version = @candidate_feature_version
          AND execution_status = @execution_status
          AND review_status IN UNNEST(@review_statuses)
        ORDER BY generated_at DESC
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("candidate_feature_version", "STRING", candidate_feature_version()),
            bigquery.ScalarQueryParameter("execution_status", "STRING", EXECUTION_STATUS_COMPLETED),
            bigquery.ArrayQueryParameter("review_statuses", "STRING", statuses),
        ]
    )
    rows = list(client.query(query, job_config=job_config).result())
    if not rows:
        return None
    return _row_to_contract(rows[0])


def _load_latest_active_feature_ranking_run(
    client: bigquery.Client,
    *,
    table_name: str,
) -> dict | None:
    query = f"""
        SELECT
          run_id,
          candidate_feature_version,
          candidate_feature_count,
          selected_feature_names,
          selected_feature_count,
          selection_source,
          review_status,
          execution_status,
          report_uri,
          contract_uri,
          generated_at,
          started_at,
          completed_at,
          notes,
          error_message,
          contract_json
        FROM `{table_name}`
        WHERE candidate_feature_version = @candidate_feature_version
          AND execution_status IN UNNEST(@execution_statuses)
        ORDER BY generated_at DESC
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("candidate_feature_version", "STRING", candidate_feature_version()),
            bigquery.ArrayQueryParameter(
                "execution_statuses",
                "STRING",
                [EXECUTION_STATUS_SUBMITTED, EXECUTION_STATUS_RUNNING],
            ),
        ]
    )
    rows = list(client.query(query, job_config=job_config).result())
    if not rows:
        return None
    return _row_to_contract(rows[0])


def _load_feature_ranking_run_by_id(
    client: bigquery.Client,
    *,
    table_name: str,
    run_id: str,
) -> dict | None:
    query = f"""
        SELECT
          run_id,
          candidate_feature_version,
          candidate_feature_count,
          selected_feature_names,
          selected_feature_count,
          selection_source,
          review_status,
          execution_status,
          report_uri,
          contract_uri,
          generated_at,
          started_at,
          completed_at,
          notes,
          error_message,
          contract_json
        FROM `{table_name}`
        WHERE run_id = @run_id
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
    )
    rows = list(client.query(query, job_config=job_config).result())
    if not rows:
        return None
    return _row_to_contract(rows[0])


def load_selected_feature_contract(
    *,
    project_id: str,
    dataset_id: str,
    require_approved: bool,
    allow_temporary_fallback: bool = False,
) -> dict | None:
    client = bigquery.Client(project=project_id or None)
    table_name = _feature_selection_table(project_id=project_id, dataset_id=dataset_id)
    try:
        contract = _load_bigquery_selected_feature_contract(
            client,
            table_name=table_name,
            require_approved=require_approved,
        )
    except Exception as exc:
        log.warning("Failed to read selected feature contract from %s: %s", table_name, exc)
        contract = None

    if contract and contract.get("selected_feature_names"):
        contract["is_temporary"] = _is_temporary_contract(contract)
        _write_local_selected_feature_contract(contract)
        return contract

    local_contract = _load_local_selected_feature_contract(include_temporary=allow_temporary_fallback)
    if require_approved and local_contract:
        if local_contract.get("is_temporary"):
            return local_contract if allow_temporary_fallback else None
        if local_contract.get("review_status") != REVIEW_STATUS_APPROVED:
            return None
        return local_contract

    return local_contract


def _temporary_automl_table_name(
    *,
    project_id: str,
    dataset_id: str,
    suffix: str,
) -> str:
    return f"{project_id}.{dataset_id}.automl_feature_selection_{suffix}"


def _load_training_frame_to_bigquery(
    client: bigquery.Client,
    *,
    table_name: str,
    dataset: pd.DataFrame,
) -> None:
    load_df = dataset.copy()
    if "date" in load_df.columns:
        load_df["date"] = pd.to_datetime(load_df["date"]).dt.date
    client.load_table_from_dataframe(
        load_df,
        table_name,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    ).result()


def _normalize_feature_attribution_entries(values: Any) -> dict[str, float]:
    if values is None:
        return {}

    items = None
    if isinstance(values, dict):
        items = values.items()
    elif hasattr(values, "items"):
        try:
            items = values.items()
        except Exception:
            items = None

    if items is None:
        raw = getattr(values, "_pb", None)
        fields = getattr(raw, "fields", None)
        if fields:
            normalized: dict[str, float] = {}
            for key, value in fields.items():
                number_value = getattr(value, "number_value", None)
                if number_value is not None:
                    normalized[str(key)] = float(number_value)
            return normalized
        return {}

    normalized = {}
    for key, value in items:
        try:
            normalized[str(key)] = float(value)
            continue
        except Exception:
            number_value = getattr(value, "number_value", None)
            if number_value is not None:
                normalized[str(key)] = float(number_value)
    return normalized


def _extract_explanation_feature_importance(explanation: Any) -> dict[str, float]:
    if explanation is None:
        return {}

    direct_attributions = getattr(explanation, "mean_attribution", None) or getattr(explanation, "meanAttribution", None)
    direct_entries = _normalize_feature_attribution_entries(direct_attributions)
    if direct_entries:
        return direct_entries

    repeated_attributions = getattr(explanation, "mean_attributions", None) or getattr(explanation, "meanAttributions", None)
    if repeated_attributions is None:
        raw = getattr(explanation, "_pb", None)
        if raw is not None:
            repeated_attributions = getattr(raw, "mean_attributions", None) or getattr(raw, "meanAttributions", None)

    if not repeated_attributions:
        return {}

    aggregated: dict[str, list[float]] = {}
    for attribution in repeated_attributions:
        feature_values = (
            getattr(attribution, "feature_attributions", None)
            or getattr(attribution, "featureAttributions", None)
        )
        entries = _normalize_feature_attribution_entries(feature_values)
        for feature_name, value in entries.items():
            aggregated.setdefault(feature_name, []).append(abs(float(value)))

    return {
        feature_name: sum(values) / len(values)
        for feature_name, values in aggregated.items()
        if values
    }


def _extract_feature_importance(model) -> list[dict[str, float]]:
    evaluations = model.list_model_evaluations()
    if not evaluations:
        raise RuntimeError("AutoML model returned no evaluations to read feature importance from.")

    evaluation = evaluations[0]
    resource = getattr(evaluation, "_gca_resource", None)
    metrics = getattr(resource, "metrics", None) if resource is not None else None
    explanations = [
        getattr(metrics, "model_explanation", None) if metrics is not None else None,
        getattr(metrics, "modelExplanation", None) if metrics is not None else None,
        getattr(resource, "model_explanation", None) if resource is not None else None,
        getattr(resource, "modelExplanation", None) if resource is not None else None,
    ]

    feature_values: dict[str, float] = {}
    for explanation in explanations:
        feature_values = _extract_explanation_feature_importance(explanation)
        if feature_values:
            break
    if not feature_values:
        raise RuntimeError("AutoML evaluation did not include usable model explanation attributions.")

    feature_importance: list[dict[str, float]] = []
    for feature_name, value in feature_values.items():
        if feature_name in CANDIDATE_FEATURE_NAMES:
            feature_importance.append({"feature": feature_name, "importance": abs(float(value))})
    return sorted(feature_importance, key=lambda item: item["importance"], reverse=True)


def _build_feature_selection_report(contract: dict) -> str:
    selected_lines = "\n".join(f"- {feature}" for feature in contract["selected_feature_names"])
    importance_lines = "\n".join(
        f"- {row['feature']}: {row['importance']:.6f}"
        for row in (contract.get("automl") or {}).get("top_feature_importance", [])
    )
    return f"""# Feature Ranking Report

## Summary

- Candidate feature version: {contract['candidate_feature_version']}
- Candidate feature count: {contract['candidate_feature_count']}
- Selected feature count: {contract['selected_feature_count']}
- Execution status: {contract.get('execution_status')}
- Review status: {contract.get('review_status')}
- Generated at: {contract.get('generated_at')}

## Selected Features

{selected_lines or "- Pending finalization"}

## Top AutoML Feature Importance

{importance_lines or "- Pending finalization"}
"""


def _upload_feature_selection_report(
    *,
    project_id: str,
    gcs_bucket: str,
    model_name: str,
    run_id: str,
    report_body: str,
    contract: dict,
) -> tuple[str, str]:
    storage_client = storage.Client(project=project_id or None)
    prefix = f"models/{model_name}/feature_ranking/{run_id}"
    bucket = storage_client.bucket(gcs_bucket)

    report_blob = bucket.blob(f"{prefix}/feature_ranking_report.md")
    report_blob.upload_from_string(report_body, content_type="text/markdown")

    contract_blob = bucket.blob(f"{prefix}/selected_features.json")
    contract_blob.upload_from_string(json.dumps(contract, indent=2), content_type="application/json")

    return (
        f"gs://{gcs_bucket}/{prefix}/feature_ranking_report.md",
        f"gs://{gcs_bucket}/{prefix}/selected_features.json",
    )


def _replace_feature_selection_run(
    client: bigquery.Client,
    *,
    table_name: str,
    contract: dict,
) -> None:
    delete_query = f"DELETE FROM `{table_name}` WHERE run_id = @run_id"
    delete_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", contract["run_id"])]
    )
    client.query(delete_query, job_config=delete_config).result()

    row = {
        "run_id": contract["run_id"],
        "candidate_feature_version": contract["candidate_feature_version"],
        "candidate_feature_count": contract["candidate_feature_count"],
        "selected_feature_names": contract["selected_feature_names"],
        "selected_feature_count": contract["selected_feature_count"],
        "selection_source": contract["selection_source"],
        "review_status": contract["review_status"],
        "execution_status": contract["execution_status"],
        "report_uri": contract.get("report_uri"),
        "contract_uri": contract.get("contract_uri"),
        "generated_at": contract["generated_at"],
        "started_at": contract.get("started_at"),
        "completed_at": contract.get("completed_at"),
        "reviewed_at": None,
        "reviewed_by": None,
        "notes": contract.get("notes"),
        "error_message": contract.get("error_message"),
        "contract_json": json.dumps(contract, separators=(",", ":")),
    }
    errors = client.insert_rows_json(table_name, [row])
    if errors:
        raise RuntimeError(f"Failed to write feature ranking run: {errors[:3]}")


def persist_feature_ranking_contract(
    *,
    project_id: str,
    dataset_id: str,
    contract: dict,
    write_local_mirror: bool = True,
) -> None:
    client = bigquery.Client(project=project_id or None)
    table_name = _feature_selection_table(project_id=project_id, dataset_id=dataset_id)
    _replace_feature_selection_run(client, table_name=table_name, contract=contract)
    if write_local_mirror:
        _write_local_selected_feature_contract(contract)


def _parse_training_job_state(training_job) -> str:
    state = getattr(training_job, "state", None)
    if state is None:
        return "UNKNOWN"
    if hasattr(state, "name"):
        return str(state.name)
    normalized = str(state)
    if "." in normalized:
        normalized = normalized.split(".")[-1]
    return normalized.upper()


def _refresh_terminal_active_run(
    *,
    project_id: str,
    region: str,
    gcs_bucket: str,
    client: bigquery.Client,
    table_name: str,
    contract: dict,
) -> dict | None:
    automl_meta = dict(contract.get("automl") or {})
    training_job_resource_name = automl_meta.get("training_job_resource_name")
    if not training_job_resource_name:
        return contract

    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region, staging_bucket=f"gs://{gcs_bucket}")
    training_job = aiplatform.AutoMLTabularTrainingJob.get(training_job_resource_name)
    state = _parse_training_job_state(training_job)
    if not training_job.done():
        return contract

    if any(token in state for token in VERTEX_FAILURE_TOKENS):
        contract["execution_status"] = EXECUTION_STATUS_FAILED
        contract["review_status"] = REVIEW_STATUS_NOT_READY
        contract["completed_at"] = datetime.now(timezone.utc).isoformat()
        contract["error_message"] = f"Vertex AutoML feature ranking ended in state {state}."
        contract["notes"] = "Previous AutoML feature ranking failed. Submit a new run after fixing the dataset."
        _replace_feature_selection_run(client, table_name=table_name, contract=contract)
        return None

    return contract


def refresh_automl_feature_ranking_status(
    *,
    project_id: str,
    region: str,
    dataset_id: str,
    gcs_bucket: str,
    run_id: str | None = None,
    write_local_mirror: bool = True,
) -> dict:
    client = bigquery.Client(project=project_id or None)
    table_name = _feature_selection_table(project_id=project_id, dataset_id=dataset_id)
    contract = (
        _load_feature_ranking_run_by_id(client, table_name=table_name, run_id=run_id)
        if run_id
        else _load_latest_active_feature_ranking_run(client, table_name=table_name)
    )
    if not contract:
        completed = load_selected_feature_contract(
            project_id=project_id,
            dataset_id=dataset_id,
            require_approved=False,
        )
        if completed:
            return completed
        raise RuntimeError("No feature ranking run was found to inspect.")

    automl_meta = dict(contract.get("automl") or {})
    training_job_resource_name = automl_meta.get("training_job_resource_name")
    if not training_job_resource_name:
        return contract

    from google.cloud import aiplatform

    now_iso = datetime.now(timezone.utc).isoformat()
    aiplatform.init(project=project_id, location=region, staging_bucket=f"gs://{gcs_bucket}")
    training_job = aiplatform.AutoMLTabularTrainingJob.get(training_job_resource_name)
    state = _parse_training_job_state(training_job)
    automl_meta["vertex_training_job_state"] = state
    automl_meta["last_polled_at"] = now_iso
    contract["automl"] = automl_meta

    if not training_job.done():
        contract["execution_status"] = EXECUTION_STATUS_RUNNING
        contract["notes"] = f"AutoML feature ranking is still running in Vertex AI (state={state})."
    elif any(token in state for token in VERTEX_FAILURE_TOKENS):
        contract["execution_status"] = EXECUTION_STATUS_FAILED
        contract["review_status"] = REVIEW_STATUS_NOT_READY
        contract["completed_at"] = now_iso
        contract["error_message"] = f"Vertex AutoML feature ranking ended in state {state}."
        contract["notes"] = "Feature ranking failed. Inspect the Vertex AI training job before retrying."
    else:
        try:
            model = training_job.get_model()
        except Exception:
            model = None
        if model is not None:
            automl_meta["model_resource_name"] = model.resource_name
        contract["notes"] = (
            f"AutoML feature ranking finished in Vertex AI (state={state}). "
            "Run the finalize step to extract selected features and create the review contract."
        )

    _replace_feature_selection_run(client, table_name=table_name, contract=contract)
    if write_local_mirror:
        _write_local_selected_feature_contract(contract)
    return contract


def submit_or_reuse_automl_feature_ranking(
    dataset: pd.DataFrame,
    *,
    project_id: str,
    region: str,
    dataset_id: str,
    gcs_bucket: str,
    model_name: str,
    refresh: bool = False,
    max_selected_features: int = DEFAULT_MAX_SELECTED_FEATURES,
    budget_milli_node_hours: int = DEFAULT_BUDGET_MILLI_NODE_HOURS,
) -> dict:
    if dataset.empty:
        raise RuntimeError("AutoML feature ranking received an empty training dataset.")

    client = bigquery.Client(project=project_id or None)
    table_name = _feature_selection_table(project_id=project_id, dataset_id=dataset_id)

    if not refresh:
        existing_contract = load_selected_feature_contract(
            project_id=project_id,
            dataset_id=dataset_id,
            require_approved=False,
        )
        if existing_contract and existing_contract.get("selected_feature_names"):
            return existing_contract

        active_run = _load_latest_active_feature_ranking_run(client, table_name=table_name)
        if active_run:
            refreshed_active_run = _refresh_terminal_active_run(
                project_id=project_id,
                region=region,
                gcs_bucket=gcs_bucket,
                client=client,
                table_name=table_name,
                contract=active_run,
            )
            if refreshed_active_run is not None:
                return refreshed_active_run

    from google.cloud import aiplatform

    suffix = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    run_id = f"fr-{suffix}"
    temp_table_name = _temporary_automl_table_name(
        project_id=project_id,
        dataset_id=dataset_id,
        suffix=suffix,
    )
    automl_frame = dataset[["date", "ticker", "target", *CANDIDATE_FEATURE_NAMES]].copy()
    _load_training_frame_to_bigquery(client, table_name=temp_table_name, dataset=automl_frame)

    aiplatform.init(project=project_id, location=region, staging_bucket=f"gs://{gcs_bucket}")
    dataset_resource = aiplatform.TabularDataset.create(
        display_name=f"market-sentiment-feature-ranking-{suffix}",
        bq_source=[f"bq://{temp_table_name}"],
        sync=True,
    )
    training_job = aiplatform.AutoMLTabularTrainingJob(
        display_name=f"market-sentiment-feature-ranking-{suffix}",
        optimization_prediction_type="classification",
        optimization_objective="maximize-au-roc",
    )
    training_job.run(
        dataset=dataset_resource,
        target_column="target",
        model_display_name=f"market-sentiment-feature-ranker-{suffix}",
        budget_milli_node_hours=budget_milli_node_hours,
        disable_early_stopping=False,
        sync=False,
    )
    training_job.wait_for_resource_creation()

    now_iso = datetime.now(timezone.utc).isoformat()
    contract = {
        "run_id": run_id,
        "candidate_feature_version": candidate_feature_version(),
        "candidate_feature_count": len(CANDIDATE_FEATURE_NAMES),
        "selected_feature_names": [],
        "selected_feature_count": 0,
        "selection_source": "vertex_automl_tabular_async",
        "review_status": REVIEW_STATUS_NOT_READY,
        "execution_status": EXECUTION_STATUS_RUNNING,
        "generated_at": now_iso,
        "started_at": now_iso,
        "completed_at": None,
        "notes": "AutoML feature ranking submitted. Run the finalize step after Vertex AutoML completes.",
        "error_message": None,
        "automl": {
            "dataset_bq_source": temp_table_name,
            "dataset_resource_name": dataset_resource.resource_name,
            "training_job_resource_name": training_job.resource_name,
            "budget_milli_node_hours": budget_milli_node_hours,
            "max_selected_features": max_selected_features,
        },
        "report_uri": None,
        "contract_uri": None,
    }
    _replace_feature_selection_run(client, table_name=table_name, contract=contract)
    return contract


def finalize_automl_feature_ranking(
    *,
    project_id: str,
    region: str,
    dataset_id: str,
    gcs_bucket: str,
    model_name: str,
    run_id: str | None = None,
    max_selected_features: int = DEFAULT_MAX_SELECTED_FEATURES,
) -> dict:
    client = bigquery.Client(project=project_id or None)
    table_name = _feature_selection_table(project_id=project_id, dataset_id=dataset_id)
    contract = (
        _load_feature_ranking_run_by_id(client, table_name=table_name, run_id=run_id)
        if run_id
        else _load_latest_active_feature_ranking_run(client, table_name=table_name)
    )
    if not contract:
        completed = load_selected_feature_contract(
            project_id=project_id,
            dataset_id=dataset_id,
            require_approved=False,
        )
        if completed:
            return completed
        raise RuntimeError("No active feature ranking run was found to finalize.")

    if contract.get("execution_status") == EXECUTION_STATUS_COMPLETED and contract.get("selected_feature_names"):
        return contract

    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region, staging_bucket=f"gs://{gcs_bucket}")
    automl_meta = dict(contract.get("automl") or {})
    training_job_resource_name = automl_meta.get("training_job_resource_name")
    if not training_job_resource_name:
        raise RuntimeError(f"Feature ranking run {contract['run_id']} is missing the AutoML training job resource name.")

    training_job = aiplatform.AutoMLTabularTrainingJob.get(training_job_resource_name)
    state = _parse_training_job_state(training_job)
    if not training_job.done():
        contract["execution_status"] = EXECUTION_STATUS_RUNNING
        contract["notes"] = f"AutoML feature ranking is still running in Vertex AI (state={state})."
        _replace_feature_selection_run(client, table_name=table_name, contract=contract)
        return contract

    if any(token in state for token in VERTEX_FAILURE_TOKENS):
        contract["execution_status"] = EXECUTION_STATUS_FAILED
        contract["review_status"] = REVIEW_STATUS_NOT_READY
        contract["completed_at"] = datetime.now(timezone.utc).isoformat()
        contract["error_message"] = f"Vertex AutoML feature ranking ended in state {state}."
        contract["notes"] = "Feature ranking failed. Inspect the Vertex AI training job before retrying."
        _replace_feature_selection_run(client, table_name=table_name, contract=contract)
        raise RuntimeError(contract["error_message"])

    model = training_job.get_model()
    feature_importance = _extract_feature_importance(model)
    selected = [row["feature"] for row in feature_importance[:max_selected_features] if row["importance"] > 0]
    if not selected:
        selected = list(CANDIDATE_FEATURE_NAMES)

    contract["selected_feature_names"] = selected
    contract["selected_feature_count"] = len(selected)
    contract["selection_source"] = "vertex_automl_tabular_async_finalize"
    contract["review_status"] = REVIEW_STATUS_PENDING
    contract["execution_status"] = EXECUTION_STATUS_COMPLETED
    contract["completed_at"] = datetime.now(timezone.utc).isoformat()
    contract["error_message"] = None
    contract["notes"] = "Pending review. Approve this contract in BigQuery before running the training job."
    contract["automl"] = {
        **automl_meta,
        "model_resource_name": model.resource_name,
        "top_feature_importance": feature_importance[:20],
        "vertex_training_job_state": state,
    }

    report_body = _build_feature_selection_report(contract)
    report_uri, contract_uri = _upload_feature_selection_report(
        project_id=project_id,
        gcs_bucket=gcs_bucket,
        model_name=model_name,
        run_id=contract["run_id"],
        report_body=report_body,
        contract=contract,
    )
    contract["report_uri"] = report_uri
    contract["contract_uri"] = contract_uri
    _replace_feature_selection_run(client, table_name=table_name, contract=contract)
    _write_local_selected_feature_contract(contract)
    return contract


def run_or_reuse_automl_feature_selection(*args, **kwargs) -> dict:
    """Backward-compatible alias for the async submit step."""

    return submit_or_reuse_automl_feature_ranking(*args, **kwargs)
