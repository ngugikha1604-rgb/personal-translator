"""
llm.py — LLM provider abstraction.

Single Groq streaming call for entire copilot analysis.
Returns LLMResult with combined metrics.
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional

from config import GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE
from services.groq_client import get_client as get_groq_client, get_user_profile
from services.context import context_manager


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


# ─── Shared streaming call (Groq only) ───────────────────────────────────────

def _run_groq_stream(
    system_prompt: str,
    user_content: str,
    on_token: Callable[[str], None] = None,
    model: str = None,
) -> LLMResult:
    """
    Single Groq streaming call. Used for both copilot and verification.
    Returns text + timing + token metrics.

    Args:
        model: override model name. Defaults to config.LLM_MODEL.
    """
    client = get_groq_client()
    effective_model = model or LLM_MODEL
    t_start = time.perf_counter()
    t_first_token: Optional[float] = None
    chunks: list[str] = []
    usage = None

    try:
        stream = client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            stream=True,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            stream_options={"include_usage": True},
        )
    except TypeError:
        stream = client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            stream=True,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )

    for chunk in stream:
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


# ─── Main calls ───────────────────────────────────────────────────────────────

def call_verification_llm(turns: list, on_token: Callable[[str], None] = None) -> LLMResult:
    """
    Call Groq LLM for user speech verification.
    Returns LLMResult with verification JSON in text field.
    """
    system_prompt     = _build_system_prompt(VERIFICATION_PROMPT)
    conversation_text = _build_conversation_text(turns)

    return _run_groq_stream(
        system_prompt,
        f"Conversation so far:\n{conversation_text}\n\nVerify the last message from 'You'.",
        on_token,
    )


# ─── Prompt templates ─────────────────────────────────────────────────────────

VERIFICATION_PROMPT = """You are a conversation alignment checker. Your job is to verify whether the user's spoken response actually addresses the speaker's intent and question — not just whether it's factually accurate.

Primary check: Does the user answer WHAT was asked?
Secondary check: Is the user factually consistent with the profile?

User profile:
- Interests: {interests}
- Communication style: {style}
{context_block}

Check the user's last message ("You" speaker) against the conversation history and the user profile. Return ONLY a valid JSON object:
{{
  "understanding_correct": <true if the user's response addresses the speaker's intent AND is factually consistent, false otherwise>,
  "factual_error": "<if understanding_correct is false, describe what went wrong — either intent mismatch or factual inconsistency. null if everything is correct>",
  "warning": "<brief user-facing warning if there's an issue (5-10 words). Describes WHY the answer missed the mark. null if everything is correct>"
}}

Rules:
- Return ONLY raw JSON. No markdown, no code fences, no extra text.
- understanding_correct must be boolean.
- PRIMARY: Check if the user answered the right question. If the speaker asks WHY and the user answers WHAT, that's understanding_correct = false.
- SECONDARY: Check factual consistency against the profile and earlier conversation.
- factual_error: null if correct, or 1 sentence explaining what was wrong (intent mismatch or factual issue).
- warning: null if correct, or a short, natural phrase (5-10 words, e.g. "They asked WHY, you answered WHAT you study").
- Do not add fields. The only allowed keys are understanding_correct, factual_error, and warning.
- False positives are better than false negatives — when in doubt, warn the user.

---

Examples:

Conversation:
Other: Why are you interested in AI?
You: I study software engineering.

Output:
{{"understanding_correct": false, "factual_error": "Speaker asked WHY the user is interested in AI, but user answered WHAT they study — does not address the question.", "warning": "They asked WHY, you answered WHAT you study"}}

---

Conversation:
Other: Why are you interested in AI?
You: I find LLMs fascinating — they changed how I think about building software.

Output:
{{"understanding_correct": true, "factual_error": null, "warning": null}}

---

Conversation:
Other: How long have you been coding?
You: I know Python and JavaScript.

Output:
{{"understanding_correct": false, "factual_error": "Speaker asked HOW LONG the user has been coding, but user answered WHAT languages they know.", "warning": "They asked HOW LONG, you listed languages"}}

---

Conversation:
Other: How long have you been coding?
You: About 5 years now.

Output:
{{"understanding_correct": true, "factual_error": null, "warning": null}}

---

Conversation:
Other: Where did you study computer science?
You: I studied AI and machine learning.

Output:
{{"understanding_correct": false, "factual_error": "Speaker asked WHERE the user studied, but user answered WHAT they studied.", "warning": "They asked WHERE, you said WHAT field"}}

---

Conversation:
Other: Where did you study computer science?
You: At Stanford University.

Output:
{{"understanding_correct": true, "factual_error": null, "warning": null}}

---

Conversation:
Other: What do you think about the new privacy policy?
You: The policy was announced last week.

Output:
{{"understanding_correct": false, "factual_error": "Speaker asked the user's OPINION about the policy, but user answered WHEN it was announced.", "warning": "They asked your OPINION, you said WHEN"}}

---

Conversation:
Other: What do you think about the new privacy policy?
You: I think it's a step forward, but it still needs work on data sharing.

Output:
{{"understanding_correct": true, "factual_error": null, "warning": null}}

---

Conversation:
Other: Do you have experience with React?
You: I've worked with Vue and Angular mostly.

Output:
{{"understanding_correct": false, "factual_error": "Speaker asked about React experience, but user answered with experience in different frameworks.", "warning": "They asked about React, you said Vue"}}

---

Conversation:
Other: Do you have experience with React?
You: Yes, I've been using it for about 2 years.

Output:
{{"understanding_correct": true, "factual_error": null, "warning": null}}

---

Conversation:
Other: What are your career goals?
You: I work at a tech startup.

Output:
{{"understanding_correct": false, "factual_error": "Speaker asked about the user's FUTURE goals, but user answered about their CURRENT job.", "warning": "They asked your GOALS, you said WHERE you work"}}

---

Conversation:
Other: What are your career goals?
You: I want to lead an AI research team in the next 3 years.

Output:
{{"understanding_correct": true, "factual_error": null, "warning": null}}

---

Conversation:
Other: How does this sorting algorithm work?
You: It's very fast compared to others.

Output:
{{"understanding_correct": false, "factual_error": "Speaker asked HOW the algorithm works (its mechanism), but user evaluated its performance instead.", "warning": "They asked HOW it works, you said it's fast"}}

---

Conversation:
Other: Are you free tomorrow?
You: I have meetings in the morning but free after 2 PM.

Output:
{{"understanding_correct": true, "factual_error": null, "warning": null}}

---

Conversation:
Other: So what do you work on?
You: I build LLM applications mostly.
Profile: interests = ["AI", "LLM", "systems programming"]

Output:
{{"understanding_correct": true, "factual_error": null, "warning": null}}

---

Conversation:
Other: Are you from Vietnam?
You: Yeah, I'm from Vietnam, been there my whole life.
Profile: home_country = "USA", years_in_vietnam = 2

Output:
{{"understanding_correct": false, "factual_error": "User said they're from Vietnam and been there whole life, but profile shows they're from USA, only 2 years in Vietnam.", "warning": "Wait — you said Vietnam your whole life, but you're from USA"}}"""
