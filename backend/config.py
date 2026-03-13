"""Shared configuration helpers for backend services."""

from __future__ import annotations

import os


def get_env(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable from the provided names."""
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        normalized = value.strip()
        if normalized:
            return normalized
    return default


def get_required_env(*names: str) -> str:
    value = get_env(*names)
    if value:
        return value
    readable_names = ", ".join(names)
    raise RuntimeError(f"Missing required configuration. Set one of: {readable_names}.")


def get_int_env(*names: str, default: int | None = None) -> int | None:
    value = get_env(*names)
    if not value:
        return default
    return int(value)


def get_float_env(*names: str, default: float | None = None) -> float | None:
    value = get_env(*names)
    if not value:
        return default
    return float(value)


def get_csv_env(*names: str) -> list[str]:
    value = get_env(*names)
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def qualify_table_name(
    table_name: str,
    *,
    project_id: str | None = None,
    dataset_id: str | None = None,
) -> str:
    resolved_project_id = project_id or get_required_env("GCP_PROJECT_ID")
    resolved_dataset_id = dataset_id or get_required_env("BQ_DATASET", "BQ_DATASET_ID")
    return f"{resolved_project_id}.{resolved_dataset_id}.{table_name}"
