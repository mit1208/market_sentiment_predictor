"""BigQuery-backed swarm repository with in-memory fallback."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from backend.config import get_env, qualify_table_name
from backend.swarm.agents.personas import build_default_personas
from backend.swarm.models.schemas import AgentPersona, AgentVerdict, ModeratorVerdict

log = logging.getLogger(__name__)

PROJECT_ID = get_env("GCP_PROJECT_ID")
DATASET_ID = get_env("BQ_DATASET", "BQ_DATASET_ID")


class SwarmRepository:
    def __init__(self) -> None:
        self._bq = None
        self._rounds_memory: dict[str, list[AgentVerdict]] = defaultdict(list)
        self._summary_memory: dict[str, ModeratorVerdict] = {}

    @property
    def bq(self):
        if self._bq is None:
            try:
                from google.cloud import bigquery

                self._bq = bigquery.Client(project=PROJECT_ID or None)
            except Exception as exc:
                log.warning("Swarm BigQuery client unavailable: %s", exc)
                self._bq = False
        return self._bq if self._bq is not False else None

    def _table(self, name: str) -> str:
        return qualify_table_name(name, project_id=PROJECT_ID or None, dataset_id=DATASET_ID or None)

    def load_personas(self) -> list[AgentPersona]:
        calibration_stats = self._load_calibration_stats()
        client = self.bq
        if client is None:
            personas = build_default_personas()
            return [self._with_calibration(persona, calibration_stats) for persona in personas]
        try:
            rows = list(
                client.query(
                    f"""
                    SELECT agent_id, display_name, system_prompt, personality_traits,
                           calibration_stats, feature_subset, created_at, updated_at
                    FROM `{self._table("swarm_agent_personas")}`
                    """
                ).result()
            )
        except Exception as exc:
            log.warning("Falling back to default swarm personas: %s", exc)
            personas = build_default_personas()
            return [self._with_calibration(persona, calibration_stats) for persona in personas]
        if not rows:
            personas = build_default_personas()
            return [self._with_calibration(persona, calibration_stats) for persona in personas]
        return [
            self._with_calibration(
                AgentPersona(
                agent_id=row.agent_id,
                display_name=row.display_name,
                system_prompt=row.system_prompt,
                personality_traits=dict(row.personality_traits or {}),
                calibration_stats=dict(row.calibration_stats or {}),
                feature_subset=list(row.feature_subset or []),
                created_at=row.created_at,
                updated_at=row.updated_at,
                ),
                calibration_stats,
            )
            for row in rows
        ]

    def save_rounds(self, verdicts: list[AgentVerdict]) -> None:
        if not verdicts:
            return
        deliberation_id = verdicts[0].deliberation_id
        self._rounds_memory[deliberation_id].extend(verdicts)
        client = self.bq
        if client is None:
            return
        try:
            errors = client.insert_rows_json(
                self._table("swarm_deliberation_rounds"),
                [self._round_row(verdict) for verdict in verdicts],
            )
            if errors:
                log.warning("Swarm rounds insert errors: %s", errors[:2])
        except Exception as exc:
            log.warning("Failed to persist swarm rounds: %s", exc)

    def save_summary(self, summary: ModeratorVerdict) -> None:
        self._summary_memory[summary.deliberation_id] = summary
        client = self.bq
        if client is None:
            return
        try:
            errors = client.insert_rows_json(
                self._table("swarm_deliberation_summary"),
                [summary.model_dump(mode="json")],
            )
            if errors:
                log.warning("Swarm summary insert errors: %s", errors[:2])
        except Exception as exc:
            log.warning("Failed to persist swarm summary: %s", exc)

    def get_deliberation(self, deliberation_id: str) -> dict[str, Any]:
        summary = self._summary_memory.get(deliberation_id)
        rounds = self._rounds_memory.get(deliberation_id, [])
        client = self.bq

        if client is not None and (summary is None or not rounds):
            if summary is None:
                try:
                    summary_rows = list(
                        client.query(
                            f"""
                            SELECT deliberation_id, request_id, ticker, final_direction, final_confidence,
                                   direction_overridden, consensus, dissent, actual_outcome, created_at
                            FROM `{self._table("swarm_deliberation_summary")}`
                            WHERE deliberation_id = @deliberation_id
                            LIMIT 1
                            """,
                            job_config=self._query_config(deliberation_id),
                        ).result()
                    )
                    if summary_rows:
                        row = summary_rows[0]
                        summary = ModeratorVerdict(
                            deliberation_id=row.deliberation_id,
                            request_id=row.request_id,
                            ticker=row.ticker,
                            final_direction=row.final_direction,
                            final_confidence=row.final_confidence,
                            direction_overridden=row.direction_overridden,
                            consensus=row.consensus,
                            dissent=row.dissent,
                            actual_outcome=row.actual_outcome,
                            created_at=row.created_at,
                        )
                except Exception as exc:
                    log.warning("Failed to load swarm summary for %s: %s", deliberation_id, exc)

            if not rounds:
                try:
                    round_rows = list(
                        client.query(
                            f"""
                            SELECT deliberation_id, request_id, ticker, agent_id, round, direction,
                                   confidence_delta, conviction, thesis, key_signals, rebuttals,
                                   agrees_with_model, revised, ml_direction, ml_confidence,
                                   actual_outcome, agent_was_correct, created_at
                            FROM `{self._table("swarm_deliberation_rounds")}`
                            WHERE deliberation_id = @deliberation_id
                            ORDER BY round, created_at
                            """,
                            job_config=self._query_config(deliberation_id),
                        ).result()
                    )
                    rounds = [
                        AgentVerdict(
                            deliberation_id=row.deliberation_id,
                            request_id=row.request_id,
                            ticker=row.ticker,
                            agent_id=row.agent_id,
                            display_name=self._display_name_for_agent(row.agent_id),
                            round=row.round,
                            direction=row.direction,
                            confidence_delta=row.confidence_delta,
                            conviction=row.conviction,
                            thesis=row.thesis,
                            key_signals=self._json_list(row.key_signals),
                            rebuttals=self._json_object(row.rebuttals),
                            agrees_with_model=row.agrees_with_model,
                            revised=row.revised,
                            ml_direction=row.ml_direction,
                            ml_confidence=row.ml_confidence,
                            actual_outcome=row.actual_outcome,
                            agent_was_correct=row.agent_was_correct,
                            created_at=row.created_at,
                        )
                        for row in round_rows
                    ]
                except Exception as exc:
                    log.warning("Failed to load swarm rounds for %s: %s", deliberation_id, exc)

        return {
            "deliberation_id": deliberation_id,
            "summary": summary.model_dump(mode="json") if summary else None,
            "rounds": [verdict.model_dump(mode="json") for verdict in rounds],
        }

    def _query_config(self, deliberation_id: str):
        from google.cloud import bigquery

        return bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("deliberation_id", "STRING", deliberation_id)
            ]
        )

    def _display_name_for_agent(self, agent_id: str) -> str:
        for persona in build_default_personas():
            if persona.agent_id == agent_id:
                return persona.display_name
        return agent_id

    def _round_row(self, verdict: AgentVerdict) -> dict[str, Any]:
        payload = verdict.model_dump(mode="json")
        payload.pop("display_name", None)
        payload.pop("public_notes", None)
        payload["key_signals"] = json.dumps(payload.get("key_signals", []))
        payload["rebuttals"] = json.dumps(payload.get("rebuttals", {}))
        return payload

    def _json_list(self, value: Any) -> list[str]:
        parsed = self._json_value(value, default=[])
        return [str(item) for item in parsed] if isinstance(parsed, list) else []

    def _json_object(self, value: Any) -> dict[str, str]:
        parsed = self._json_value(value, default={})
        if not isinstance(parsed, dict):
            return {}
        return {str(key): str(val) for key, val in parsed.items()}

    def _json_value(self, value: Any, default: Any) -> Any:
        if value in (None, ""):
            return default
        if isinstance(value, (list, dict)):
            return value
        try:
            return json.loads(value)
        except Exception:
            return default

    def _with_calibration(
        self,
        persona: AgentPersona,
        calibration_stats: dict[str, dict[str, Any]],
    ) -> AgentPersona:
        stats = calibration_stats.get(persona.agent_id, {})
        if not stats:
            return persona
        return AgentPersona(
            **{
                **persona.model_dump(),
                "calibration_stats": {
                    **persona.calibration_stats,
                    **stats,
                },
            }
        )

    def _load_calibration_stats(self) -> dict[str, dict[str, Any]]:
        client = self.bq
        if client is None:
            return {}
        try:
            rows = list(
                client.query(
                    f"""
                    SELECT
                      agent_id,
                      COUNT(*) AS verdict_count,
                      AVG(IF(agent_was_correct IS NULL, NULL, IF(agent_was_correct, 1.0, 0.0))) AS accuracy,
                      AVG(confidence_delta) AS avg_confidence_delta,
                      AVG(CASE WHEN direction = ml_direction THEN 1.0 ELSE 0.0 END) AS agree_rate
                    FROM `{self._table("swarm_deliberation_rounds")}`
                    WHERE round = 2
                      AND created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
                    GROUP BY agent_id
                    """
                ).result()
            )
        except Exception as exc:
            log.warning("Failed to load swarm calibration stats: %s", exc)
            return {}
        return {
            row.agent_id: {
                "verdict_count_30d": int(row.verdict_count or 0),
                "accuracy_30d": None if row.accuracy is None else round(float(row.accuracy), 4),
                "avg_confidence_delta_30d": None if row.avg_confidence_delta is None else round(float(row.avg_confidence_delta), 4),
                "agree_rate_30d": None if row.agree_rate is None else round(float(row.agree_rate), 4),
            }
            for row in rows
        }
