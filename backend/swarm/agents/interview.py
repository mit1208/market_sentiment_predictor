"""Ask-the-analysts helper."""

from __future__ import annotations

from backend.swarm.config import MAX_INTERVIEW_TURNS
from backend.swarm.gemini.client import GeminiSwarmClient
from backend.swarm.models.schemas import SwarmInterviewRequest, SwarmInterviewResponse
from backend.swarm.storage.bigquery import SwarmRepository

INTERVIEW_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answer": {"type": "STRING"},
    },
    "required": ["answer"],
}


class InterviewService:
    def __init__(self, repository: SwarmRepository) -> None:
        self.repository = repository
        self.gemini = GeminiSwarmClient()

    def answer(self, request: SwarmInterviewRequest) -> SwarmInterviewResponse:
        deliberation = self.repository.get_deliberation(request.deliberation_id)
        rounds = deliberation.get("rounds") or []
        relevant = [
            verdict for verdict in rounds if verdict.get("agent_id") == request.agent
        ]
        history = request.history[-MAX_INTERVIEW_TURNS:]
        answer = None
        if self.gemini.enabled:
            prompt = (
                f"Answer as the {request.agent} agent.\n"
                f"Prior verdicts: {relevant}\n"
                f"Conversation history: {history}\n"
                f"Question: {request.question}\n"
                "Return structured JSON only."
            )
            payload = self.gemini.generate_json(prompt=prompt, response_schema=INTERVIEW_SCHEMA)
            answer = (payload or {}).get("answer")
        if not answer:
            last_thesis = relevant[-1]["thesis"] if relevant else "I do not have a prior verdict stored."
            answer = (
                f"My view is grounded in the recorded deliberation: {last_thesis} "
                f"In direct response to your question, the main issue is whether the current evidence"
                f" materially changes that risk/reward balance."
            )
        return SwarmInterviewResponse(agent=request.agent, answer=answer, turn=len(history) + 1)
