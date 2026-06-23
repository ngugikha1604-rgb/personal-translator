from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, USER_PROFILE

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a conversation copilot. Your job is to help the user respond faster in real-time English conversations.

User profile:
- Interests: {interests}
- Communication style: {style}

Analyze the last message from "Other" and return ONLY a valid JSON object:
{{
  "intent": "<what the speaker wants — short phrase, max 6 words, in English>",
  "summary": "<one sentence explaining what is happening in this conversation, in English — used for reasoning only, never displayed>",
  "reply": "<short phrase with the key point(s) for the user to say — NOT a full sentence, the user will speak in their own words>"
}}

Rules:
- Return ONLY raw JSON. No markdown, no code fences, no extra text.
- The reply MUST be truthful. Never invent facts about the user.
- reply must be a short phrase, not a full sentence. The user adapts it when speaking.
- intent must be max 6 words.
- summary is internal reasoning context — keep it one sentence.
""".format(
    interests=", ".join(USER_PROFILE["interests"]),
    style=", ".join(USER_PROFILE["communication_style"])
)


def _build_conversation_text(turns: list) -> str:
    lines = []
    for turn in turns:
        if turn["speaker"] == "other":
            speaker = "Other"
        else:
            speaker = "You"
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


def stream_analysis(turns: list):
    """Yields text tokens from LLM streaming response."""
    conversation_text = _build_conversation_text(turns)

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Conversation so far:\n{conversation_text}\n\nAnalyze the last message from 'Other'."}
        ],
        stream=True,
        max_tokens=300,
        temperature=0.3
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token
