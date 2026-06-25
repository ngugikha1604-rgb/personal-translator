"""analyzer.py — Conversation analysis layer.

Separates understanding from response generation.
Analyzer produces structured analysis of the speaker's intent,
conversation dynamics, and potential misunderstandings.
"""

import json
import re
from dataclasses import dataclass

from services.llm import (
    _build_system_prompt,
    _build_conversation_text,
    _run_groq_stream,
)


# ─── Result ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AnalysisResult:
    """Structured analysis of a conversation turn."""
    intent: str
    conversation_goal: str
    question_type: str
    social_signal: str
    understanding_check: str | None
    summary: str
    raw: str
    llm_ms: int = 0
    ttft_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Internal: pre-parsed dict for downstream consumers (e.g. ReplyGenerator)
    _parsed: dict | None = None


# ─── JSON repair (shared with copilot.py) ─────────────────────────────────────

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


# ─── Prompt template ──────────────────────────────────────────────────────────

ANALYZER_PROMPT = """You are a conversation analyzer. Your job is to understand the other person's conversational intent, social signals, and potential misunderstandings in real-time English conversations.

User profile:
- Interests: {interests}
- Communication style: {style}
{context_block}
Analyze the conversation and return ONLY a valid JSON object:
{{
  "intent": "<the speaker's conversational PURPOSE — why they said it, not what they said — short phrase, max 8 words, in English>",
  "conversation_goal": "<what this line of questioning or statement aims to achieve in the broader conversation, e.g. assessing technical fit, building rapport, gathering background info>",
  "question_type": "<open-ended | yes/no | clarification | opinion-seeking | probing | suggestion | statement | backchannel>",
  "social_signal": "<the social tone: casual | formal | curious | skeptical | appreciative | probing | concerned | enthusiastic | neutral>",
  "understanding_check": "<if the speaker's question is likely to be misunderstood, explain the nuance they should watch for. null if no risk.>",
  "summary": "<one sentence explaining what is happening in this conversation, in English — used for reasoning only, never displayed>",
  "reply": "<spoken response fragment — must sound like natural speech mid-sentence, with a verb or connector so the user can start speaking immediately. NOT a noun list. The user glances at this and speaks it out loud.>"
}}

Rules:
- Return ONLY raw JSON. No markdown, no code fences, no extra text.
- The reply MUST be truthful. Never invent facts about the user.
- reply is a spoken fragment, not a noun list. Wrong: "AI and software engineering". Right: "studying AI, building LLM stuff". Always include a verb or natural connector.
- reply should be 5–9 words — enough to carry a real thought, short enough to read in a glance.
- intent must represent the speaker's conversational purpose, max 8 words.
- conversation_goal: what the exchange aims to achieve (e.g. "assessing technical fit", "building rapport", "gathering background info").
- question_type: classify the type of question or statement (e.g. open-ended, yes/no, clarification, probing).
- social_signal: detect the speaker's social/emotional tone (e.g. casual, curious, skeptical, probing).
- understanding_check: explain nuance when the question could be misinterpreted. null when the meaning is obvious.
- summary is internal reasoning context — keep it one sentence.
- Do not add fields. The only allowed keys are intent, conversation_goal, question_type, social_signal, understanding_check, summary, and reply.

---

Examples:

Conversation:
Other: What are you studying?

Output:
{{"intent": "trying to understand educational background", "conversation_goal": "gathering background info about the user", "question_type": "open-ended", "social_signal": "curious", "understanding_check": null, "summary": "The other person wants to know what the user is currently studying.", "reply": "studying AI, mostly building LLM stuff"}}

---

Conversation:
Other: Hey, nice to meet you! So what brings you here?

Output:
{{"intent": "opening a networking conversation", "conversation_goal": "building rapport and finding common ground", "question_type": "open-ended", "social_signal": "friendly", "understanding_check": null, "summary": "They are starting the conversation casually and want to know why the user is here.", "reply": "just here to meet people, see what's going on"}}

---

Conversation:
Other: Do you compete in any programming contests?

Output:
{{"intent": "evaluating technical experience", "conversation_goal": "assessing the user's technical depth", "question_type": "yes/no", "social_signal": "probing", "understanding_check": null, "summary": "They want to know if the user participates in competitive programming competitions.", "reply": "yeah, been doing it for about two years"}}

---

Conversation:
Other: How long have you been doing competitive programming?
You: About two years now.
Other: Have you done ICPC?

Output:
{{"intent": "probing competition skill level", "conversation_goal": "benchmarking user against a known competition", "question_type": "yes/no", "social_signal": "probing", "understanding_check": null, "summary": "They are probing the user's competitive programming background, specifically ICPC participation.", "reply": "not yet, but planning to go for it"}}

---

Conversation:
Other: What got you interested in AI?

Output:
{{"intent": "probing origin of interest", "conversation_goal": "understanding the user's motivation and background", "question_type": "open-ended", "social_signal": "curious", "understanding_check": "They are asking WHY you became interested in AI, not HOW you learned AI.", "summary": "They want to know what sparked the user's interest in AI, not how they learned it.", "reply": "got into it through LLM stuff, found it fascinating"}}"""


# ─── Service ──────────────────────────────────────────────────────────────────

class Analyzer:
    """Conversation analyzer. Produces structured analysis from conversation turns."""

    def analyze(self, turns: list) -> AnalysisResult:
        """Single LLM call → structured analysis with all intent/context fields + embedded reply."""
        system_prompt     = _build_system_prompt(ANALYZER_PROMPT)
        conversation_text = _build_conversation_text(turns)

        llm = _run_groq_stream(
            system_prompt,
            f"Conversation:\n{conversation_text}\n\nAnalyze the last message from 'Other'.",
        )

        parsed = _safe_parse_json(llm.text)
        if parsed is None:
            raise ValueError(
                f"Analyzer LLM returned unparseable JSON after repair: {llm.text!r}"
            )

        return AnalysisResult(
            intent              = str(parsed.get("intent", "")).strip(),
            conversation_goal   = str(parsed.get("conversation_goal", "")).strip(),
            question_type       = str(parsed.get("question_type", "")).strip(),
            social_signal       = str(parsed.get("social_signal", "")).strip(),
            understanding_check = parsed.get("understanding_check") or None,
            summary             = str(parsed.get("summary", "")).strip(),
            raw                 = llm.text,
            _parsed             = parsed,
            llm_ms              = llm.total_ms,
            ttft_ms             = llm.ttft_ms,
            prompt_tokens       = llm.prompt_tokens,
            completion_tokens   = llm.completion_tokens,
            total_tokens        = llm.total_tokens,
        )
