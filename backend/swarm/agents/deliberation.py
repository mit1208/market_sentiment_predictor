"""Swarm deliberation orchestration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from backend.swarm.config import AGENT_ORDER
from backend.swarm.gemini.client import GeminiSwarmClient
from backend.swarm.models.schemas import AgentPersona, AgentVerdict, MlPrediction, SwarmAnalyzeRequest
from backend.swarm.storage.bigquery import SwarmRepository

ROUND1_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "direction": {"type": "STRING", "enum": ["Up", "Down"]},
        "confidence_delta": {"type": "NUMBER"},
        "conviction": {"type": "STRING", "enum": ["low", "medium", "high"]},
        "thesis": {"type": "STRING"},
        "key_signals": {"type": "ARRAY", "items": {"type": "STRING"}},
        "agrees_with_model": {"type": "BOOLEAN"},
    },
    "required": [
        "direction",
        "confidence_delta",
        "conviction",
        "thesis",
        "key_signals",
        "agrees_with_model",
    ],
}

ROUND2_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "revised": {"type": "BOOLEAN"},
        "direction": {"type": "STRING", "enum": ["Up", "Down"]},
        "confidence_delta": {"type": "NUMBER"},
        "conviction": {"type": "STRING", "enum": ["low", "medium", "high"]},
        "thesis": {"type": "STRING"},
        "rebuttals": {"type": "OBJECT"},
        "agrees_with_model": {"type": "BOOLEAN"},
    },
    "required": [
        "revised",
        "direction",
        "confidence_delta",
        "conviction",
        "thesis",
        "rebuttals",
        "agrees_with_model",
    ],
}


def build_deliberation_id(ticker: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"delib:{ticker}:{now}"


def _feature_map(feature_snapshot: dict[str, Any]) -> dict[str, Any]:
    groups = feature_snapshot.get("groups") or []
    features: dict[str, Any] = {}
    for group in groups:
        for feature in group.get("features", []):
            name = feature.get("name")
            if name:
                features[str(name)] = feature.get("value")
    return features


def _subset_for_agent(persona: AgentPersona, feature_snapshot: dict[str, Any]) -> dict[str, Any]:
    features = _feature_map(feature_snapshot)
    if persona.feature_subset == ["*"]:
        return features
    return {name: features.get(name) for name in persona.feature_subset}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _heuristic_round1(
    *,
    persona: AgentPersona,
    ml_prediction: MlPrediction,
    feature_snapshot: dict[str, Any],
    news_items: list[dict[str, Any]],
) -> dict[str, Any]:
    features = _feature_map(feature_snapshot)
    sentiment = _safe_float(features.get("gemini_sentiment_avg"))
    headline_count = _safe_float(features.get("headline_count"))
    vix_level = _safe_float(features.get("vix_level"))
    spy_change = _safe_float(features.get("spy_change_pct"))
    volume_vs_avg = _safe_float(features.get("volume_vs_avg_10d"), 1.0)
    gap_pct = _safe_float(features.get("gap_pct"))

    direction = ml_prediction.direction
    conviction = "medium"
    delta = 0.0
    thesis_parts: list[str] = []
    key_signals: list[str] = []

    if persona.agent_id == "sentiment_analyst":
        if sentiment >= 0.05:
            delta += 0.08
            key_signals.append("Positive headline sentiment")
            thesis_parts.append("Recent news flow is supportive and sentiment skews positive.")
        elif sentiment <= -0.05:
            direction = "Down"
            delta -= 0.10
            key_signals.append("Negative headline sentiment")
            thesis_parts.append("Headline tone is negative enough to challenge the model.")
        if headline_count >= 5:
            key_signals.append("Dense news tape")
    elif persona.agent_id == "macro_economist":
        if spy_change > 0 and vix_level < 22:
            delta += 0.05
            key_signals.append("Constructive macro tape")
            thesis_parts.append("Broader index action and volatility regime are supportive.")
        elif spy_change < 0 or vix_level >= 26:
            direction = "Down"
            delta -= 0.08
            key_signals.append("Risk-off macro backdrop")
            thesis_parts.append("The broader tape looks fragile and argues for caution.")
    elif persona.agent_id == "risk_manager":
        if vix_level >= 25 or volume_vs_avg >= 1.4:
            direction = "Down"
            conviction = "high"
            delta -= 0.12
            key_signals.append("Elevated volatility/risk")
            thesis_parts.append("Risk conditions are unstable and downside tails look larger.")
        else:
            delta -= 0.02
            key_signals.append("Normal but non-trivial risk")
    else:
        if ml_prediction.direction == "Up":
            direction = "Down"
            delta -= 0.07
        else:
            direction = "Up"
            delta += 0.07
        key_signals.append("Stress-testing consensus")
        thesis_parts.append("The consensus case may be overconfident, so the opposite scenario deserves weight.")
        if abs(gap_pct) > 0.02:
            conviction = "high"
            key_signals.append("Large opening gap")

    if not thesis_parts:
        thesis_parts.append("The live feature snapshot is mostly aligned with the model baseline.")
    if news_items and persona.agent_id == "sentiment_analyst":
        thesis_parts.append(f"Recent headline count is {int(headline_count)} and supports a narrative-led read.")

    agrees_with_model = direction == ml_prediction.direction
    if abs(delta) >= 0.1 and conviction == "medium":
        conviction = "high"
    elif abs(delta) <= 0.03:
        conviction = "low"

    return {
        "direction": direction,
        "confidence_delta": round(delta, 4),
        "conviction": conviction,
        "thesis": " ".join(thesis_parts)[:320],
        "key_signals": key_signals[:3],
        "agrees_with_model": agrees_with_model,
    }


def _heuristic_round2(
    *,
    persona: AgentPersona,
    round1: AgentVerdict,
    peers: list[AgentVerdict],
    ml_prediction: MlPrediction,
) -> dict[str, Any]:
    disagreeing_peers = [peer for peer in peers if peer.direction != round1.direction]
    revised = False
    direction = round1.direction
    delta = round1.confidence_delta
    conviction = round1.conviction
    thesis = round1.thesis

    if persona.agent_id != "contrarian" and len(disagreeing_peers) >= 2 and round1.conviction == "low":
        revised = True
        direction = ml_prediction.direction
        delta = round(delta / 2, 4)
        conviction = "medium"
        thesis = (
            "After weighing the other analysts, the opposing case looks less decisive than"
            " the first pass suggested, so I move closer to the base model."
        )
    elif persona.agent_id == "contrarian" and len(disagreeing_peers) == 0:
        revised = True
        direction = ml_prediction.direction
        delta = 0.02 if ml_prediction.direction == "Up" else -0.02
        conviction = "low"
        thesis = "Consensus held up under challenge, so the stress-test no longer justifies a directional break."

    rebuttals = {
        peer.display_name: f"I acknowledge {peer.display_name.lower()}'s point, but my weighting still favors {direction.lower()}."
        for peer in peers[:2]
    }
    return {
        "revised": revised,
        "direction": direction,
        "confidence_delta": round(delta, 4),
        "conviction": conviction,
        "thesis": thesis[:320],
        "rebuttals": rebuttals,
        "agrees_with_model": direction == ml_prediction.direction,
    }


class DeliberationOrchestrator:
    def __init__(self, repository: SwarmRepository) -> None:
        self.repository = repository
        self.gemini = GeminiSwarmClient()

    def _round1_prompt(
        self,
        *,
        persona: AgentPersona,
        request: SwarmAnalyzeRequest,
    ) -> str:
        subset = _subset_for_agent(persona, request.feature_snapshot)
        headlines = [
            item.get("headline", "")
            for item in sorted(
                request.news_items,
                key=lambda item: item.get("published_at") or "",
                reverse=True,
            )[:5]
        ]
        return (
            f"{persona.system_prompt}\n\n"
            f"Ticker: {request.ticker}\n"
            f"Model prediction: {request.ml_prediction.model_dump()}\n"
            f"Feature subset: {subset}\n"
            f"Recent headlines: {headlines}\n"
            "Return structured JSON only."
        )

    def _round2_prompt(
        self,
        *,
        persona: AgentPersona,
        request: SwarmAnalyzeRequest,
        own_verdict: AgentVerdict,
        peer_verdicts: list[AgentVerdict],
    ) -> str:
        peer_summary = [
            {
                "agent": verdict.display_name,
                "direction": verdict.direction,
                "conviction": verdict.conviction,
                "thesis": verdict.thesis,
            }
            for verdict in peer_verdicts
        ]
        return (
            f"{persona.system_prompt}\n\n"
            f"Ticker: {request.ticker}\n"
            f"Your round 1 verdict: {own_verdict.model_dump()}\n"
            f"Other agents: {peer_summary}\n"
            "Return structured JSON only."
        )

    def _run_round1_for_persona(
        self,
        *,
        deliberation_id: str,
        persona: AgentPersona,
        request: SwarmAnalyzeRequest,
    ) -> AgentVerdict:
        payload = self.gemini.generate_json(
            prompt=self._round1_prompt(persona=persona, request=request),
            response_schema=ROUND1_SCHEMA,
        ) or _heuristic_round1(
            persona=persona,
            ml_prediction=request.ml_prediction,
            feature_snapshot=request.feature_snapshot,
            news_items=request.news_items,
        )
        return AgentVerdict(
            deliberation_id=deliberation_id,
            request_id=request.request_id,
            ticker=request.ticker,
            agent_id=persona.agent_id,
            display_name=persona.display_name,
            round=1,
            ml_direction=request.ml_prediction.direction,
            ml_confidence=request.ml_prediction.confidence,
            revised=False,
            **payload,
        )

    def _run_round2_for_persona(
        self,
        *,
        deliberation_id: str,
        persona: AgentPersona,
        request: SwarmAnalyzeRequest,
        round1_verdicts: list[AgentVerdict],
    ) -> AgentVerdict:
        own_verdict = next(verdict for verdict in round1_verdicts if verdict.agent_id == persona.agent_id)
        peer_verdicts = [verdict for verdict in round1_verdicts if verdict.agent_id != persona.agent_id]
        payload = self.gemini.generate_json(
            prompt=self._round2_prompt(
                persona=persona,
                request=request,
                own_verdict=own_verdict,
                peer_verdicts=peer_verdicts,
            ),
            response_schema=ROUND2_SCHEMA,
        ) or _heuristic_round2(
            persona=persona,
            round1=own_verdict,
            peers=peer_verdicts,
            ml_prediction=request.ml_prediction,
        )
        return AgentVerdict(
            deliberation_id=deliberation_id,
            request_id=request.request_id,
            ticker=request.ticker,
            agent_id=persona.agent_id,
            display_name=persona.display_name,
            round=2,
            ml_direction=request.ml_prediction.direction,
            ml_confidence=request.ml_prediction.confidence,
            key_signals=own_verdict.key_signals,
            **payload,
        )

    def run_round1(
        self,
        *,
        deliberation_id: str,
        request: SwarmAnalyzeRequest,
        personas: list[AgentPersona],
    ) -> list[AgentVerdict]:
        ordered = sorted(personas, key=lambda persona: AGENT_ORDER.index(persona.agent_id))
        with ThreadPoolExecutor(max_workers=4) as executor:
            verdicts = list(
                executor.map(
                    lambda persona: self._run_round1_for_persona(
                        deliberation_id=deliberation_id,
                        persona=persona,
                        request=request,
                    ),
                    ordered,
                )
            )
        self.repository.save_rounds(verdicts)
        return verdicts

    def run_round2(
        self,
        *,
        deliberation_id: str,
        request: SwarmAnalyzeRequest,
        personas: list[AgentPersona],
        round1_verdicts: list[AgentVerdict],
    ) -> list[AgentVerdict]:
        ordered = sorted(personas, key=lambda persona: AGENT_ORDER.index(persona.agent_id))
        with ThreadPoolExecutor(max_workers=4) as executor:
            verdicts = list(
                executor.map(
                    lambda persona: self._run_round2_for_persona(
                        deliberation_id=deliberation_id,
                        persona=persona,
                        request=request,
                        round1_verdicts=round1_verdicts,
                    ),
                    ordered,
                )
            )
        self.repository.save_rounds(verdicts)
        return verdicts
