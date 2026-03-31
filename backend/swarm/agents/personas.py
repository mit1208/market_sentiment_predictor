"""Persona loading with BigQuery-backed cache and local defaults."""

from __future__ import annotations

from backend.swarm.config import AGENT_ORDER, DEFAULT_AGENT_PERSONAS
from backend.swarm.models.schemas import AgentPersona


def build_default_personas() -> list[AgentPersona]:
    personas: list[AgentPersona] = []
    for agent_id in AGENT_ORDER:
        definition = DEFAULT_AGENT_PERSONAS[agent_id]
        personas.append(
            AgentPersona(
                agent_id=agent_id,
                display_name=definition["display_name"],
                system_prompt=definition["system_prompt"],
                personality_traits=definition["personality_traits"],
                calibration_stats={},
                feature_subset=list(definition["feature_subset"]),
            )
        )
    return personas
