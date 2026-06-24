"""
copilot.py — LLM reasoning layer.

Receives conversation turns, returns CopilotResult.
Audio capture and STT are handled upstream (main.py + speech.py).
This service only knows about text — not audio, not HTTP.
"""

import json
from dataclasses import dataclass

from services.llm import call_llm


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
    intent:            str
    summary:           str   # internal only — never displayed
    reply:             str
    raw:               str
    llm_ms:            int
    ttft_ms:           int
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int

    def display_payload(self) -> dict:
        return {"intent": self.intent, "reply": self.reply}


# ─── Service ──────────────────────────────────────────────────────────────────

class CopilotService:
    """LLM-based conversation analysis. Input: turns list. Output: CopilotResult."""

    def analyze_turns(self, turns: list) -> CopilotResult:
        llm = call_llm(turns)

        try:
            parsed = json.loads(llm.text)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {llm.text!r}") from e

        return CopilotResult(
            intent            = _extract_intent(parsed),
            summary           = _extract_summary(parsed),
            reply             = _extract_reply(parsed),
            raw               = llm.text,
            llm_ms            = llm.total_ms,
            ttft_ms           = llm.ttft_ms,
            prompt_tokens     = llm.prompt_tokens,
            completion_tokens = llm.completion_tokens,
            total_tokens      = llm.total_tokens,
        )


# Singleton
copilot_service = CopilotService()
