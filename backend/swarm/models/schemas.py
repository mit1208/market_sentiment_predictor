"""Pydantic schemas for the swarm service."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MlPrediction(BaseModel):
    direction: Literal["Up", "Down"]
    confidence: float
    probability: float


class SwarmAnalyzeRequest(BaseModel):
    ticker: str
    request_id: str
    ml_prediction: MlPrediction
    feature_snapshot: dict[str, Any]
    news_items: list[dict[str, Any]] = Field(default_factory=list)


class AgentPersona(BaseModel):
    agent_id: str
    display_name: str
    system_prompt: str
    personality_traits: dict[str, Any] = Field(default_factory=dict)
    calibration_stats: dict[str, Any] = Field(default_factory=dict)
    feature_subset: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class AgentVerdict(BaseModel):
    deliberation_id: str
    request_id: str
    ticker: str
    agent_id: str
    display_name: str
    round: Literal[1, 2]
    direction: Literal["Up", "Down"]
    confidence_delta: float
    conviction: Literal["low", "medium", "high"]
    thesis: str
    key_signals: list[str] = Field(default_factory=list)
    rebuttals: dict[str, str] = Field(default_factory=dict)
    agrees_with_model: bool
    revised: bool = False
    ml_direction: Literal["Up", "Down"]
    ml_confidence: float
    actual_outcome: Optional[str] = None
    agent_was_correct: Optional[bool] = None
    created_at: datetime = Field(default_factory=utc_now)


class ModeratorVerdict(BaseModel):
    deliberation_id: str
    request_id: str
    ticker: str
    final_direction: Literal["Up", "Down"]
    final_confidence: float
    direction_overridden: bool
    consensus: str
    dissent: Optional[str] = None
    actual_outcome: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class SwarmInterviewRequest(BaseModel):
    deliberation_id: str
    agent: str
    question: str
    history: list[dict[str, str]] = Field(default_factory=list)


class SwarmInterviewResponse(BaseModel):
    agent: str
    answer: str
    turn: int
