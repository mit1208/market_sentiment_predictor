"""Deterministic moderation rules for swarm verdicts."""

from __future__ import annotations

from backend.swarm.config import (
    CONFIDENCE_CEILING,
    CONFIDENCE_FLOOR,
    HIGH_CONVICTION_THRESHOLD,
    SUPERMAJORITY_THRESHOLD,
)
from backend.swarm.models.schemas import AgentVerdict, MlPrediction, ModeratorVerdict

WEIGHTS = {"low": 0.5, "medium": 1.0, "high": 1.5}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def opposite(direction: str) -> str:
    return "Down" if direction == "Up" else "Up"


def moderate(
    *,
    deliberation_id: str,
    request_id: str,
    ticker: str,
    ml_prediction: MlPrediction,
    agent_verdicts: list[AgentVerdict],
) -> ModeratorVerdict:
    dissenters = [verdict for verdict in agent_verdicts if not verdict.agrees_with_model]
    high_conviction_dissenters = [
        verdict for verdict in dissenters if verdict.conviction == "high"
    ]
    direction_overridden = (
        len(dissenters) >= SUPERMAJORITY_THRESHOLD
        and len(high_conviction_dissenters) >= HIGH_CONVICTION_THRESHOLD
    )

    total_weight = sum(WEIGHTS[verdict.conviction] for verdict in agent_verdicts) or 1.0
    adjusted_delta = (
        sum(verdict.confidence_delta * WEIGHTS[verdict.conviction] for verdict in agent_verdicts)
        / total_weight
    )
    final_confidence = clamp(
        ml_prediction.confidence + adjusted_delta,
        CONFIDENCE_FLOOR,
        CONFIDENCE_CEILING,
    )
    final_direction = (
        opposite(ml_prediction.direction) if direction_overridden else ml_prediction.direction
    )
    strongest_dissent = (
        max(dissenters, key=lambda verdict: WEIGHTS[verdict.conviction]) if dissenters else None
    )
    return ModeratorVerdict(
        deliberation_id=deliberation_id,
        request_id=request_id,
        ticker=ticker,
        final_direction=final_direction,
        final_confidence=round(float(final_confidence), 4),
        direction_overridden=direction_overridden,
        consensus=f"{4 - len(dissenters)}/4 agree",
        dissent=(
            f"{strongest_dissent.display_name}: {strongest_dissent.thesis}"
            if strongest_dissent
            else None
        ),
    )
