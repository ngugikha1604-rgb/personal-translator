"""
llm.py — LLM provider abstraction.

Calls configured LLM (currently Groq) via shared client.
Prompt templates live in prompts.py. Profile reads cached in groq_client.py.
Single streaming implementation — no copy-paste between copilot and verification.
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional

from config import LLM_MODEL
from services.groq_client import get_client, get_user_profile
from services.context import context_manager
from services.prompts import COPILOT_SYSTEM_PROMPT, VERIFICATION_SYSTEM_PROMPT


# ─── Result ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LLMResult:
    text:               str
    ttft_ms:            int   # time to first token
    total_ms:           int   # total LLM call time
    prompt_tokens:      int
    completion_tokens:  int
    total_tokens:       int


# ─── Prompt assembly ──────────────────────────────────────────────────────────

def _build_system_prompt(template: str) -> str:
    profile = get_user_profile()
    interests = ", ".join(profile.get("interests", []))
    style = ", ".join(profile.get("communication_style", []))
    context_block = context_manager.get_prompt_block()
    context_section = f"\n{context_block}\n" if context_block else "\n"
    return template.format(
        interests=interests,
        style=style,
        context_block=context_section,
    )


def _build_conversation_text(turns: list) -> str:
    lines = []
    for turn in turns:
        speaker = "Other" if turn["speaker"] == "other" else "You"
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


# ─── Streaming call ───────────────────────────────────────────────────────────

def _run_stream(system_prompt: str, user_content: str, on_token: Callable[[str], None] = None) -> LLMResult:
    """
    Call Groq streaming API, accumulate text + metrics.
    """
    client = get_client()
    t_start = time.perf_counter()
    t_first_token: Optional[float] = None
    chunks: list[str] = []
    # ponytail: usage tracking depends on stream_options support in Groq SDK
    usage = None

    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            stream=True,
            max_tokens=300,
            temperature=0.3,
            stream_options={"include_usage": True},
        )
    except TypeError:
        # Groq SDK version does not support stream_options — retry without it
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            stream=True,
            max_tokens=300,
            temperature=0.3,
        )

    for chunk in stream:
        # Final usage chunk when stream_options is supported (choices is empty)
        if hasattr(chunk, 'usage') and chunk.usage is not None and not chunk.choices:
            usage = chunk.usage
            continue

        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content
        if content:
            if t_first_token is None:
                t_first_token = time.perf_counter()
            chunks.append(content)
            if on_token:
                on_token(content)

    t_end = time.perf_counter()

    return LLMResult(
        text              = "".join(chunks),
        ttft_ms           = round((t_first_token - t_start) * 1000) if t_first_token else 0,
        total_ms          = round((t_end - t_start) * 1000),
        prompt_tokens     = usage.prompt_tokens     if usage else 0,
        completion_tokens = usage.completion_tokens if usage else 0,
        total_tokens      = usage.total_tokens      if usage else 0,
    )


# ─── Main call ────────────────────────────────────────────────────────────────

def call_llm(turns: list, on_token: Callable[[str], None] = None) -> LLMResult:
    """
    Call LLM for copilot analysis (analyze last "Other" turn).
    Returns LLMResult with full text + latency + token metrics.
    Optional on_token callback receives each content chunk as it arrives.
    """
    system_prompt     = _build_system_prompt(COPILOT_SYSTEM_PROMPT)
    conversation_text = _build_conversation_text(turns)

    return _run_stream(system_prompt, conversation_text, on_token)


def call_verification_llm(turns: list, on_token: Callable[[str], None] = None) -> LLMResult:
    """
    Call LLM for user speech verification.
    Returns LLMResult with verification JSON in text field.
    Optional on_token callback receives each content chunk as it arrives.
    """
    system_prompt     = _build_system_prompt(VERIFICATION_SYSTEM_PROMPT)
    conversation_text = _build_conversation_text(turns)

    return _run_stream(
        system_prompt,
        f"Conversation so far:\n{conversation_text}\n\nVerify the last message from 'You'.",
        on_token,
    )


