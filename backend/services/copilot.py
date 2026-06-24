"""
copilot.py — LLM reasoning layer.

Receives conversation turns, returns CopilotResult.
Audio capture and STT are handled upstream (main.py + speech.py).
This service only knows about text — not audio, not HTTP.
"""

import json
import time
from dataclasses import dataclass
from typing import Generator

from services.llm import stream_analysis


# ─── Field extractors ─────────────────────────────────────────────────────────

def _extract_intent(payload: dict) -> str:
    return str(payload.get("intent", "")).strip()

def _extract_summary(payload: dict) -> str:
    return str(payload.get("summary", "")).strip()

def _extract_reply(payload: dict) -> str:
    return str(payload.get("reply", "")).strip()


# ─── Result ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CopilotResult:
    intent:  str
    summary: str   # internal only — never displayed
    reply:   str
    raw:     str
    llm_ms:  int

    def display_payload(self) -> dict:
        """For any external consumer that needs the displayable fields."""
        return {"intent": self.intent, "reply": self.reply}


# ─── Service ──────────────────────────────────────────────────────────────────

class CopilotService:
    """LLM-based conversation analysis. Input: turns list. Output: CopilotResult."""

    def stream_turns(self, turns: list) -> Generator[str, None, None]:
        yield from stream_analysis(turns)

    def analyze_turns(self, turns: list) -> CopilotResult:
        t0 = time.perf_counter()
        full_text = "".join(self.stream_turns(turns))
        llm_ms = round((time.perf_counter() - t0) * 1000)
        print(f"[LAT] LLM: {llm_ms}ms")

        parsed = json.loads(full_text)
        return CopilotResult(
            intent=_extract_intent(parsed),
            summary=_extract_summary(parsed),
            reply=_extract_reply(parsed),
            raw=full_text,
            llm_ms=llm_ms,
        )


# Singleton
copilot_service = CopilotService()
