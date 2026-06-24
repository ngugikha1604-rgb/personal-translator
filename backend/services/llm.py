import json
import time
from dataclasses import dataclass

from groq import Groq

from config import GROQ_API_KEY, LLM_MODEL, USER_PROFILE_PATH
from services.context import context_manager

client = Groq(api_key=GROQ_API_KEY)


def _load_profile() -> dict:
    with open(USER_PROFILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


PROMPT_TEMPLATE = """You are a conversation copilot. Your job is to help the user respond faster in real-time English conversations.

User profile:
- Interests: {interests}
- Communication style: {style}
{context_block}
Analyze the last message from "Other" and return ONLY a valid JSON object:
{{
  "intent": "<what the speaker wants - short phrase, max 6 words, in English>",
  "summary": "<one sentence explaining what is happening in this conversation, in English - used for reasoning only, never displayed>",
  "reply": "<spoken response fragment — must sound like natural speech mid-sentence, with a verb or connector so the user can start speaking immediately. NOT a noun list. The user glances at this and speaks it out loud.>"
}}

Rules:
- Return ONLY raw JSON. No markdown, no code fences, no extra text.
- The reply MUST be truthful. Never invent facts about the user.
- reply is a spoken fragment, not a noun list. Wrong: "AI and software engineering". Right: "studying AI, building LLM stuff". Always include a verb or natural connector.
- reply should be 5–9 words — enough to carry a real thought, short enough to read in a glance.
- intent must be max 6 words.
- summary is internal reasoning context - keep it one sentence.
- Do not add fields. The only allowed keys are intent, summary, and reply.

---

Examples:

Conversation:
Other: What are you studying?

Output:
{{"intent": "asking about field of study", "summary": "The other person wants to know what the user is currently studying.", "reply": "studying AI, mostly building LLM stuff"}}

---

Conversation:
Other: Hey, nice to meet you! So what brings you here?

Output:
{{"intent": "opening small talk", "summary": "They are starting the conversation casually and want to know why the user is here.", "reply": "just here to meet people, see what's going on"}}

---

Conversation:
Other: Do you compete in any programming contests?

Output:
{{"intent": "asking about competitive programming", "summary": "They want to know if the user participates in competitive programming competitions.", "reply": "yeah, been doing it for about two years"}}

---

Conversation:
Other: How long have you been doing competitive programming?
You: About two years now.
Other: Have you done ICPC?

Output:
{{"intent": "asking about ICPC experience", "summary": "They are probing the user's competitive programming background, specifically ICPC participation.", "reply": "not yet, but planning to go for it"}}

---

Conversation:
Other: What do you think about large language models? Are they actually useful?

Output:
{{"intent": "asking opinion on LLMs", "summary": "The other person wants the user's personal take on whether LLMs have real practical value.", "reply": "yeah, really useful especially for coding and reasoning"}}

---

Conversation:
Other: Nice to meet you. So what do you do?
You: I'm into AI and software.
Other: Oh interesting, what kind of AI projects?

Output:
{{"intent": "asking for specific AI work", "summary": "They followed up on the user's AI interest and want concrete examples of projects.", "reply": "been building LLM apps, some systems stuff too"}}

---

Conversation:
Other: So tell me about yourself. What's your background?
You: I studied computer science and work on AI stuff.
Other: That's cool. We're looking for an ML engineer. Have you worked with transformers?
You: Yeah I've built a few LLM applications.
Other: What about production deployment? Ever put models into production?

Output:
{{"intent": "probing production ML experience", "summary": "They seem to be evaluating the user for an ML engineer role, specifically production experience.", "reply": "yeah, deployed a few with Docker and monitoring"}}
"""


def _build_system_prompt() -> str:
    profile = _load_profile()
    interests = ", ".join(profile.get("interests", []))
    style = ", ".join(profile.get("communication_style", []))
    context_block = context_manager.get_prompt_block()
    context_section = f"\n{context_block}\n" if context_block else "\n"
    return PROMPT_TEMPLATE.format(
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


# ─── Result ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LLMResult:
    text:               str
    ttft_ms:            int   # time to first token
    total_ms:           int   # total LLM call time
    prompt_tokens:      int
    completion_tokens:  int
    total_tokens:       int


# ─── Main call ────────────────────────────────────────────────────────────────

def call_llm(turns: list) -> LLMResult:
    """
    Call the LLM and return full text + latency + token metrics.
    Uses streaming internally to capture TTFT accurately.
    """
    system_prompt     = _build_system_prompt()
    conversation_text = _build_conversation_text(turns)

    t_start       = time.perf_counter()
    t_first_token = None
    chunks        = []
    usage         = None

    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Conversation so far:\n{conversation_text}\n\n"
                        "Analyze the last message from 'Other'."
                    ),
                },
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
                {
                    "role": "user",
                    "content": (
                        f"Conversation so far:\n{conversation_text}\n\n"
                        "Analyze the last message from 'Other'."
                    ),
                },
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

    t_end = time.perf_counter()

    return LLMResult(
        text              = "".join(chunks),
        ttft_ms           = round((t_first_token - t_start) * 1000) if t_first_token else 0,
        total_ms          = round((t_end - t_start) * 1000),
        prompt_tokens     = usage.prompt_tokens     if usage else 0,
        completion_tokens = usage.completion_tokens if usage else 0,
        total_tokens      = usage.total_tokens      if usage else 0,
    )
