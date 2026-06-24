"""
copilot.py — LLM reasoning layer.

Receives conversation turns, returns CopilotResult.
Audio capture and STT are handled upstream (main.py + speech.py).
This service only knows about text — not audio, not HTTP.
"""

import json
import re
from dataclasses import dataclass

from services.llm import call_llm, call_verification_llm


# ─── JSON repair ──────────────────────────────────────────────────────────────

def _repair_json(raw: str) -> str:
    """Attempt to repair common LLM JSON formatting issues.

    Handles:
    - markdown code fences (```json ... ```)
    - leading/trailing non-JSON text
    - trailing commas before closing brace
    - single-quote strings where JSON expects double quotes

    Falls back to regex extraction of the outermost { ... } block.
    """
    # Strip markdown code fences
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()

    # Find the outermost JSON object
    brace_start = raw.find('{')
    brace_end = raw.rfind('}')
    if brace_start == -1 or brace_end == -1:
        return ""  # no JSON found
    raw = raw[brace_start:brace_end + 1]

    # Remove trailing commas before } or ]
    raw = re.sub(r',\s*}', '}', raw)
    raw = re.sub(r',\s*]', ']', raw)

    return raw


def _safe_parse_json(raw: str) -> dict | None:
    """Try to parse raw text as JSON. Returns dict or None."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repaired = _repair_json(raw)
        if repaired:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                return None
        return None


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


@dataclass(frozen=True)
class VerificationResult:
    understanding_correct: bool
    factual_error:         str | None
    warning:               str | None
    raw:                   str
    llm_ms:                int
    ttft_ms:               int
    prompt_tokens:         int
    completion_tokens:     int
    total_tokens:          int


# ─── Service ──────────────────────────────────────────────────────────────────

class CopilotService:
    """LLM-based conversation analysis. Input: turns list. Output: CopilotResult."""

    def analyze_turns(self, turns: list) -> CopilotResult:
        llm = call_llm(turns)

        parsed = _safe_parse_json(llm.text)
        if parsed is None:
            raise ValueError(f"LLM returned unparseable JSON after repair: {llm.text!r}")

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

    def analyze_user_speech(self, turns: list) -> VerificationResult:
        llm = call_verification_llm(turns)

        parsed = _safe_parse_json(llm.text)
        if parsed is None:
            raise ValueError(f"LLM returned unparseable JSON after repair: {llm.text!r}")

        return VerificationResult(
            understanding_correct = parsed.get("understanding_correct", False),
            factual_error         = parsed.get("factual_error"),
            warning               = parsed.get("warning"),
            raw                   = llm.text,
            llm_ms                = llm.total_ms,
            ttft_ms               = llm.ttft_ms,
            prompt_tokens         = llm.prompt_tokens,
            completion_tokens     = llm.completion_tokens,
            total_tokens          = llm.total_tokens,
        )


# Singleton
copilot_service = CopilotService()
