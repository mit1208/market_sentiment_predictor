"""Swarm FastAPI service with SSE streaming endpoints."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from backend.swarm.agents.deliberation import DeliberationOrchestrator, build_deliberation_id
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
        round1_verdicts = orchestrator.run_round1(
            deliberation_id=deliberation_id,
            request=request,
            personas=personas,
        )
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
                    "key_signals": verdict.key_signals,
                    "agrees_with_model": verdict.agrees_with_model,
                },
            )

        round2_verdicts = orchestrator.run_round2(
            deliberation_id=deliberation_id,
            request=request,
            personas=personas,
            round1_verdicts=round1_verdicts,
        )
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
