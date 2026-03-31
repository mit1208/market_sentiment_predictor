"""Optional Gemini client with deterministic fallback heuristics."""

from __future__ import annotations

import json
from typing import Any

from backend.swarm.config import GEMINI_API_KEY, GEMINI_MODEL


class GeminiSwarmClient:
    def __init__(self) -> None:
        self._client = None
        if GEMINI_API_KEY:
            try:
                from google import genai

                self._client = genai.Client(api_key=GEMINI_API_KEY)
            except Exception:
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def generate_json(
        self,
        *,
        prompt: str,
        response_schema: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self._client:
            return None

        from google.genai import types

        response = self._client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )
        if not response.text:
            return None
        return json.loads(response.text)
