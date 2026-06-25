"""
copilot.py — LLM reasoning layer.

Receives conversation turns, returns CopilotResult or VerificationResult.
Audio capture and STT are handled upstream (main.py + speech.py).
This service only knows about text — not audio, not HTTP.
"""

import json
import re
from dataclasses import dataclass

from services.analyzer import Analyzer
from services.llm import call_verification_llm
from services.reply_generator import ReplyGenerator
from services.state import conversation_state


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

    # Convert single quotes to double quotes for keys and string values
    raw = re.sub(r"(?<=\s)'", '"', raw)
    raw = re.sub(r"^'", '"', raw)
    raw = re.sub(r"'(?=\s|:|,|}|$)", '"', raw)

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


# ─── Result ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CopilotResult:
    intent:              str
    reply:               str
    raw:                 str
    llm_ms:              int
    ttft_ms:             int
    prompt_tokens:       int
    completion_tokens:   int
    total_tokens:        int
    understanding_check: str | None = None

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
        """Analyze conversation → generate reply. Updates conversation state."""
        analysis = Analyzer().analyze(turns)
        reply    = ReplyGenerator().generate(analysis)

        # Update live conversation state for the next turn
        last_turn = turns[-1] if turns else {}
        conversation_state.update(
            intent=analysis.intent,
            social_signal=analysis.social_signal,
            turn_text=last_turn.get("text", ""),
        )

        return CopilotResult(
            intent            = analysis.intent,
            reply             = reply,
            understanding_check = analysis.understanding_check,
            raw               = analysis.raw,
            llm_ms            = analysis.llm_ms,
            ttft_ms           = analysis.ttft_ms,
            prompt_tokens     = analysis.prompt_tokens,
            completion_tokens = analysis.completion_tokens,
            total_tokens      = analysis.total_tokens,
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
