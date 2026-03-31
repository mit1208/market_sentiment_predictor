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


def _grouped_feature_context(feature_snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for group in feature_snapshot.get("groups") or []:
        label = str(group.get("label") or group.get("key") or "unknown")
        grouped[label] = {}
        for feature in group.get("features", []):
            name = feature.get("name")
            if name:
                grouped[label][str(name)] = feature.get("value")
    return grouped


def _top_feature_context(feature_snapshot: dict[str, Any], limit: int = 20) -> dict[str, Any]:
    features = _feature_map(feature_snapshot)
    preferred = [
        "current_price",
        "day_change_pct",
        "gap_pct",
        "intraday_range_pct",
        "day_range_position",
        "volume_vs_avg_10d",
        "price_vs_50d_avg",
        "return_5d",
        "return_20d",
        "price_acceleration",
        "relative_strength_20d",
        "gemini_sentiment_avg",
        "gemini_sentiment_max",
        "gemini_sentiment_min",
        "gemini_magnitude_avg",
        "headline_count",
        "sentiment_dispersion",
        "yield_curve_spread",
        "credit_spread",
        "spy_change_pct",
        "vix_level",
        "vix_change_pct",
        "dxy_change_5d",
        "fed_funds_rate",
        "fed_funds_change_3m",
        "universe_breadth_50d",
    ]
    selected = {name: features.get(name) for name in preferred if name in features}
    if len(selected) < limit:
        for name, value in features.items():
            if name in selected:
                continue
            selected[name] = value
            if len(selected) >= limit:
                break
    return selected


def _news_context(news_items: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    ordered = sorted(
        news_items,
        key=lambda item: item.get("published_at") or "",
        reverse=True,
    )
    return [
        {
            "headline": item.get("headline"),
            "published_at": item.get("published_at"),
            "sentiment_score": item.get("sentiment_score"),
            "sentiment_label": item.get("sentiment_label"),
            "magnitude": item.get("magnitude"),
            "category": item.get("category"),
        }
        for item in ordered[:limit]
    ]


def _persona_runtime_notes(persona: AgentPersona) -> str:
    stats = persona.calibration_stats or {}
    if not stats:
        return "No recent calibration stats are available yet."
    return (
        "Recent calibration context: "
        f"30d verdicts={stats.get('verdict_count_30d')}, "
        f"30d accuracy={stats.get('accuracy_30d')}, "
        f"30d agree_rate={stats.get('agree_rate_30d')}, "
        f"30d avg_confidence_delta={stats.get('avg_confidence_delta_30d')}."
        " Use this only as self-correction context, not as evidence about the ticker."
    )


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
        all_features = _top_feature_context(request.feature_snapshot, limit=28)
        grouped_features = _grouped_feature_context(request.feature_snapshot)
        headlines = _news_context(request.news_items)
        provider_summary = request.feature_snapshot.get("provider_summary") or {}
        cache_status = request.feature_snapshot.get("cache_status") or {}
        return (
            f"{persona.system_prompt}\n\n"
            "You are participating in a four-agent equity deliberation. Use the full live feature"
            " snapshot, recent sentiment-scored headlines, and historical-style signals already"
            " embedded in the snapshot. Do not ignore past-data proxies such as moving averages,"
            " momentum, breadth, spreads, and volatility regime.\n\n"
            f"{_persona_runtime_notes(persona)}\n"
            f"Ticker: {request.ticker}\n"
            f"Model prediction: {request.ml_prediction.model_dump()}\n"
            f"Your focus features: {subset}\n"
            f"Full key feature frame: {all_features}\n"
            f"Grouped feature frame: {grouped_features}\n"
            f"Recent sentiment-scored headlines: {headlines}\n"
            f"Provider summary: {provider_summary}\n"
            f"Cache status: {cache_status}\n"
            "When reasoning, explicitly weigh trend/momentum, volatility/risk, macro regime,"
            " and the tone/dispersion of news. Confidence_delta should adjust the base model"
            " confidence, not replace it.\n"
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
                "key_signals": verdict.key_signals,
            }
            for verdict in peer_verdicts
        ]
        return (
            f"{persona.system_prompt}\n\n"
            "Re-evaluate your position after reading the other analysts. Preserve your independence,"
            " but update if their use of the same full feature frame is materially stronger.\n\n"
            f"{_persona_runtime_notes(persona)}\n"
            f"Ticker: {request.ticker}\n"
            f"Your round 1 verdict: {own_verdict.model_dump()}\n"
            f"Other agents: {peer_summary}\n"
            f"Full key feature frame: {_top_feature_context(request.feature_snapshot, limit=28)}\n"
            f"Recent sentiment-scored headlines: {_news_context(request.news_items)}\n"
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
