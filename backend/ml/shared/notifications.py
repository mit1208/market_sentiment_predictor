"""Lightweight webhook notifications for long-running ML jobs."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import requests

log = logging.getLogger(__name__)

EVENT_TYPE_FEATURE_RANKING_READY = "feature_ranking_ready_for_review"
EVENT_TYPE_MODEL_TRAINED = "model_trained_and_evaluated"


def build_feature_ranking_notification_payload(contract: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON payload for feature-ranking completion."""
    selected_count = int(contract.get("selected_feature_count") or 0)
    selected_preview = ", ".join((contract.get("selected_feature_names") or [])[:10]) or "none"
    text = (
        f"AutoML feature ranking completed for run {contract.get('run_id')} "
        f"with {selected_count} selected features. Review status: {contract.get('review_status')}. "
        f"Top selections: {selected_preview}. "
        f"Report: {contract.get('report_uri') or 'n/a'}"
    )
    return {
        "event_type": EVENT_TYPE_FEATURE_RANKING_READY,
        "text": text,
        "run_id": contract.get("run_id"),
        "candidate_feature_version": contract.get("candidate_feature_version"),
        "candidate_feature_count": contract.get("candidate_feature_count"),
        "selected_feature_count": selected_count,
        "selected_feature_names": contract.get("selected_feature_names") or [],
        "review_status": contract.get("review_status"),
        "execution_status": contract.get("execution_status"),
        "report_uri": contract.get("report_uri"),
        "contract_uri": contract.get("contract_uri"),
        "automl": contract.get("automl") or {},
        "generated_at": contract.get("generated_at"),
        "completed_at": contract.get("completed_at"),
    }


def build_model_training_notification_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON payload for model training completion/promotion."""
    promoted = manifest.get("promoted", True)
    status_str = "PROMOTED to live" if promoted else "NOT PROMOTED (accuracy below threshold)"
    text = (
        f"Model training completed for {manifest.get('model_name', 'market_sentiment')}. "
        f"Status: {status_str}. "
        f"Accuracy: {manifest.get('accuracy', 0):.4f} (Holdout: {manifest.get('holdout_accuracy', 0):.4f}). "
        f"Artifact: {manifest.get('artifact_uri', 'n/a')}"
    )
    return {
        "event_type": EVENT_TYPE_MODEL_TRAINED,
        "text": text,
        "model_name": manifest.get("model_name"),
        "promoted": promoted,
        "accuracy": manifest.get("accuracy"),
        "auc": manifest.get("auc"),
        "holdout_accuracy": manifest.get("holdout_accuracy"),
        "holdout_auc": manifest.get("holdout_auc"),
        "artifact_uri": manifest.get("artifact_uri"),
        "trained_at": manifest.get("trained_at"),
        "feature_selection_run_id": manifest.get("feature_selection_run_id"),
    }


def maybe_send_feature_ranking_webhook(
    *,
    webhook_url: str,
    contract: dict[str, Any],
    timeout_seconds: int = 15,
) -> tuple[bool, dict[str, Any]]:
    """
    Send a single completion notification when a feature-ranking run is ready for review.
    """
    notification = dict(contract.get("notification") or {})
    if not webhook_url:
        return False, contract

    if contract.get("execution_status") != "COMPLETED" or contract.get("review_status") != "PENDING_REVIEW":
        return False, contract

    if notification.get("status") == "SENT":
        return False, contract

    payload = build_feature_ranking_notification_payload(contract)
    sent, metadata = _send_webhook_payload(webhook_url, payload, timeout_seconds)
    contract["notification"] = metadata
    return sent, contract


def maybe_send_model_training_notification(
    *,
    webhook_url: str,
    manifest: dict[str, Any],
    timeout_seconds: int = 15,
) -> bool:
    """Send a notification when model training is complete."""
    if not webhook_url:
        return False

    payload = build_model_training_notification_payload(manifest)
    sent, _ = _send_webhook_payload(webhook_url, payload, timeout_seconds)
    return sent


def _send_webhook_payload(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> tuple[bool, dict[str, Any]]:
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        response = requests.post(url, json=payload, timeout=timeout_seconds)
        response.raise_for_status()
    except Exception as exc:
        log.warning("Webhook notification failed: %s", exc)
        return False, {
            "status": "FAILED",
            "last_attempt_at": now_iso,
            "error_message": str(exc),
        }

    return True, {
        "status": "SENT",
        "last_attempt_at": now_iso,
        "sent_at": now_iso,
        "http_status_code": response.status_code,
        "error_message": None,
    }
