"""Optional Gemini client with deterministic fallback heuristics."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from backend.swarm.config import GCP_PROJECT_ID, GCP_REGION, GEMINI_API_KEY, GEMINI_MODEL

log = logging.getLogger(__name__)
GEMINI_MAX_RETRIES = 4
GEMINI_BACKOFF_SECONDS = 1.5


class GeminiSwarmClient:
    def __init__(self) -> None:
        self._client = None
        self._vertex_mode = False
        try:
            from google import genai

            if GCP_PROJECT_ID and GCP_REGION:
                self._client = genai.Client(
                    vertexai=True,
                    project=GCP_PROJECT_ID,
                    location=GCP_REGION,
                )
                self._vertex_mode = True
            elif GEMINI_API_KEY:
                self._client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as exc:
            log.warning("Swarm Gemini client unavailable: %s", exc)
            self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def generate_json(
        self,
        *,
        prompt: str,
        response_schema: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        if not self._client:
            return None

        from google.genai import errors as genai_errors
        from google.genai import types

        delay_seconds = GEMINI_BACKOFF_SECONDS
        for attempt in range(1, GEMINI_MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                        response_schema=response_schema,
                    ),
                )
                if not response.text:
                    return None
                return json.loads(response.text)
            except (json.JSONDecodeError, ValueError) as exc:
                if attempt == GEMINI_MAX_RETRIES:
                    raise
                log.warning("Swarm Gemini returned invalid JSON on attempt %s/%s: %s", attempt, GEMINI_MAX_RETRIES, exc)
            except genai_errors.ClientError as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code not in {429, 500, 503} or attempt == GEMINI_MAX_RETRIES:
                    raise
                log.warning("Swarm Gemini throttled in %s mode on attempt %s/%s", "vertex" if self._vertex_mode else "api_key", attempt, GEMINI_MAX_RETRIES)
            time.sleep(delay_seconds)
            delay_seconds *= 2
        return None
