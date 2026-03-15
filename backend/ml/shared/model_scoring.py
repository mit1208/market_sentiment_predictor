"""Shared helpers for model blending and probability-to-label decisions."""

from __future__ import annotations


DEFAULT_CLASSIFICATION_THRESHOLD = 0.5
DEFAULT_LGB_WEIGHT = 0.5


def _clamp(value: float, *, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def resolve_ensemble_lgb_weight(metadata: dict | None) -> float:
    payload = metadata or {}
    strategy = str(payload.get("prediction_strategy") or "").strip().lower()
    if strategy == "lightgbm_only":
        return 1.0
    if strategy == "xgboost_only":
        return 0.0
    weight = _coerce_float(payload.get("ensemble_lgb_weight"), DEFAULT_LGB_WEIGHT)
    return _clamp(weight, minimum=0.0, maximum=1.0)


def resolve_classification_threshold(metadata: dict | None) -> float:
    payload = metadata or {}
    threshold = _coerce_float(payload.get("classification_threshold"), DEFAULT_CLASSIFICATION_THRESHOLD)
    return _clamp(threshold, minimum=0.0, maximum=1.0)


def blend_positive_class_scores(
    lgb_probability: float,
    xgb_probability: float,
    *,
    metadata: dict | None = None,
) -> float:
    lgb_weight = resolve_ensemble_lgb_weight(metadata)
    probability = (lgb_weight * float(lgb_probability)) + ((1.0 - lgb_weight) * float(xgb_probability))
    return float(_clamp(probability, minimum=0.0, maximum=1.0))


def classify_probability(
    probability: float,
    *,
    metadata: dict | None = None,
) -> tuple[str, float, float]:
    threshold = resolve_classification_threshold(metadata)
    probability = float(_clamp(float(probability), minimum=0.0, maximum=1.0))
    # Use Python float for comparisons to avoid "ambiguous truth value" with numpy scalars
    label = "Up" if probability >= threshold else "Down"
    confidence = float(probability if label == "Up" else 1.0 - probability)
    return label, confidence, float(threshold)
