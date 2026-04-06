"""Swarm FastAPI service with SSE streaming endpoints."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from backend.swarm.agents.deliberation import DeliberationOrchestrator, build_deliberation_id
from backend.swarm.config import AGENT_ORDER
from backend.swarm.agents.interview import InterviewService
from backend.swarm.agents.moderator import moderate
from backend.swarm.models.schemas import SwarmAnalyzeRequest, SwarmInterviewRequest
from backend.swarm.storage.bigquery import SwarmRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

repository = SwarmRepository()
orchestrator = DeliberationOrchestrator(repository)
interview_service = InterviewService(repository)


def format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Swarm service starting")
    yield


app = FastAPI(title="Swarm Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/swarm/interview")
def interview(request: SwarmInterviewRequest) -> dict:
    return interview_service.answer(request).model_dump(mode="json")


@app.get("/swarm/deliberation/{deliberation_id}")
def get_deliberation(deliberation_id: str) -> dict:
    return repository.get_deliberation(deliberation_id)


@app.post("/swarm/analyze")
async def analyze(request: SwarmAnalyzeRequest) -> StreamingResponse:
    personas = repository.load_personas()
    deliberation_id = build_deliberation_id(request.ticker)

    async def event_stream() -> AsyncIterator[str]:
        ordered_personas = sorted(personas, key=lambda persona: AGENT_ORDER.index(persona.agent_id))
        round1_verdicts = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            round1_futures = {}
            for persona in ordered_personas:
                yield format_sse(
                    "agent_status",
                    {
                        "deliberation_id": deliberation_id,
                        "agent_id": persona.agent_id,
                        "agent": persona.display_name,
                        "round": 1,
                        "status": "thinking",
                        "message": f"{persona.display_name} is reviewing the live frame.",
                    },
                )
                future = executor.submit(
                    orchestrator._run_round1_for_persona,
                    deliberation_id=deliberation_id,
                    persona=persona,
                    request=request,
                )
                round1_futures[future] = persona

            for future in as_completed(round1_futures):
                verdict = future.result()
                round1_verdicts.append(verdict)
                for index, note in enumerate(verdict.public_notes, start=1):
                    yield format_sse(
                        "agent_note",
                        {
                            "deliberation_id": deliberation_id,
                            "agent_id": verdict.agent_id,
                            "agent": verdict.display_name,
                            "round": 1,
                            "index": index,
                            "total": len(verdict.public_notes),
                            "note": note,
                        },
                    )
                yield format_sse(
                    "agent_status",
                    {
                        "deliberation_id": deliberation_id,
                        "agent_id": verdict.agent_id,
                        "agent": verdict.display_name,
                        "round": 1,
                        "status": "posted",
                        "message": f"{verdict.display_name} posted an opening take.",
                    },
                )
        round1_verdicts = sorted(round1_verdicts, key=lambda verdict: AGENT_ORDER.index(verdict.agent_id))
        repository.save_rounds(round1_verdicts)
        for verdict in round1_verdicts:
            yield format_sse(
                "round1_verdict",
                {
                    "deliberation_id": deliberation_id,
                    "agent_id": verdict.agent_id,
                    "agent": verdict.display_name,
                    "direction": verdict.direction,
                    "conviction": verdict.conviction,
                    "thesis": verdict.thesis,
                    "public_notes": verdict.public_notes,
                    "key_signals": verdict.key_signals,
                    "agrees_with_model": verdict.agrees_with_model,
                },
            )

        round2_verdicts = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            round2_futures = {}
            for persona in ordered_personas:
                yield format_sse(
                    "agent_status",
                    {
                        "deliberation_id": deliberation_id,
                        "agent_id": persona.agent_id,
                        "agent": persona.display_name,
                        "round": 2,
                        "status": "responding",
                        "message": f"{persona.display_name} is responding to the rest of the room.",
                    },
                )
                future = executor.submit(
                    orchestrator._run_round2_for_persona,
                    deliberation_id=deliberation_id,
                    persona=persona,
                    request=request,
                    round1_verdicts=round1_verdicts,
                )
                round2_futures[future] = persona

            for future in as_completed(round2_futures):
                verdict = future.result()
                round2_verdicts.append(verdict)
                for index, note in enumerate(verdict.public_notes, start=1):
                    yield format_sse(
                        "agent_note",
                        {
                            "deliberation_id": deliberation_id,
                            "agent_id": verdict.agent_id,
                            "agent": verdict.display_name,
                            "round": 2,
                            "index": index,
                            "total": len(verdict.public_notes),
                            "note": note,
                        },
                    )
                yield format_sse(
                    "agent_status",
                    {
                        "deliberation_id": deliberation_id,
                        "agent_id": verdict.agent_id,
                        "agent": verdict.display_name,
                        "round": 2,
                        "status": "posted",
                        "message": f"{verdict.display_name} posted a rebuttal-round update.",
                    },
                )
        round2_verdicts = sorted(round2_verdicts, key=lambda verdict: AGENT_ORDER.index(verdict.agent_id))
        repository.save_rounds(round2_verdicts)
        for verdict in round2_verdicts:
            yield format_sse(
                "round2_verdict",
                {
                    "deliberation_id": deliberation_id,
                    "agent_id": verdict.agent_id,
                    "agent": verdict.display_name,
                    "revised": verdict.revised,
                    "direction": verdict.direction,
                    "conviction": verdict.conviction,
                    "thesis": verdict.thesis,
                    "public_notes": verdict.public_notes,
                    "rebuttals": verdict.rebuttals,
                    "agrees_with_model": verdict.agrees_with_model,
                },
            )

        summary = moderate(
            deliberation_id=deliberation_id,
            request_id=request.request_id,
            ticker=request.ticker,
            ml_prediction=request.ml_prediction,
            agent_verdicts=round2_verdicts,
        )
        repository.save_summary(summary)
        yield format_sse("moderator_verdict", summary.model_dump(mode="json"))
        yield format_sse("done", {"deliberation_id": deliberation_id})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
