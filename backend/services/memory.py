"""
memory.py — Automatic memory learning for user profile.

After each session, extracts interests, communication style, and
personal facts from the conversation, then updates user_profile.json.

This is the LEARNING PATH — runs only after session ends, never on real-time path.
"""

import json
import os
import re
from typing import Optional

from config import USER_PROFILE_PATH
from services.groq_client import get_client as get_groq_client


def extract_profile_from_conversation(turns: list, current_profile: dict) -> dict:
    """
    Use a small LLM call to extract new interests and style cues from the
    conversation that aren't already in the profile.

    Returns a dict with keys to merge into profile:
        new_interests: list[str] — interests observed in this session
        new_style_cues: list[str] — communication style observations
        new_facts: dict[str, str] — key-value personal facts (e.g. "home_country": "USA")
    """
    if not turns:
        return {"new_interests": [], "new_style_cues": [], "new_facts": {}}

    # Build a compact representation of the user's turns
    user_turns = [t for t in turns if t["speaker"] == "user"]
    if not user_turns:
        return {"new_interests": [], "new_style_cues": [], "new_facts": {}}

    existing_interests = ", ".join(current_profile.get("interests", []))
    existing_style = ", ".join(current_profile.get("communication_style", []))
    existing_facts = ", ".join(f"{k}={v}" for k, v in current_profile.get("facts", {}).items())

    user_speech = "\n".join(f"You: {t['text']}" for t in user_turns)

    system_prompt = """You are a user profiling assistant. Extract personal information, interests, and communication traits from the user's own speech in a conversation.

Current profile (already known):
- Interests: {interests}
- Style: {style}
- Facts: {facts}

Return ONLY a valid JSON object with these fields (all optional — omit if nothing new found):
{{
  "new_interests": ["list", "of", "new", "interests", "not", "in", "existing", "profile"],
  "new_style_cues": ["traits", "like", "direct", "humble", "detailed", "humorous"],
  "new_facts": {{"fact_key": "fact_value"}}
}}

Rules:
- Only extract things the USER said about themselves, not things the OTHER person said.
- "new_interests" should be specific (e.g. "Python" not "technology").
- "new_style_cues" describes HOW the user talks (e.g. "short answers", "self-deprecating", "enthusiastic about AI").
- "new_facts" are concrete personal facts (e.g. "years_coding": "5", "home_country": "Vietnam").
- If nothing new is found, return empty arrays and empty object.
- Return ONLY raw JSON. No markdown, no extra text.""".format(
        interests=existing_interests or "(none)",
        style=existing_style or "(none)",
        facts=existing_facts or "(none)",
    )

    user_content = f"Extract profile info from this user speech:\n\n{user_speech}"

    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=200,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        # Extract JSON from possible markdown wrapping
        brace_start = raw.find('{')
        brace_end = raw.rfind('}')
        if brace_start != -1 and brace_end != -1:
            raw = raw[brace_start:brace_end + 1]
        parsed = json.loads(raw)
        return {
            "new_interests": parsed.get("new_interests", []),
            "new_style_cues": parsed.get("new_style_cues", []),
            "new_facts": parsed.get("new_facts", {}),
        }
    except Exception:
        # Silent failure — memory learning must never crash the session
        return {"new_interests": [], "new_style_cues": [], "new_facts": {}}


def merge_into_profile(new_data: dict) -> None:
    """
    Merge extracted data into user_profile.json.
    Deduplicates interests and style cues.
    Adds new facts (non-destructive — won't overwrite existing keys).
    """
    try:
        with open(USER_PROFILE_PATH, "r", encoding="utf-8") as f:
            profile = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        profile = {"interests": [], "communication_style": []}

    changed = False

    # Merge interests (deduplicate)
    existing_interests = set(i.lower() for i in profile.get("interests", []))
    for interest in new_data.get("new_interests", []):
        if interest.lower() not in existing_interests:
            profile.setdefault("interests", []).append(interest)
            existing_interests.add(interest.lower())
            changed = True

    # Merge style cues (deduplicate)
    existing_style = set(s.lower() for s in profile.get("communication_style", []))
    for cue in new_data.get("new_style_cues", []):
        if cue.lower() not in existing_style:
            profile.setdefault("communication_style", []).append(cue)
            existing_style.add(cue.lower())
            changed = True

    # Merge facts (non-destructive — new values only, never overwrite)
    facts = profile.setdefault("facts", {})
    for key, value in new_data.get("new_facts", {}).items():
        if key not in facts:
            facts[key] = value
            changed = True

    if changed:
        with open(USER_PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        # Clear profile cache so next LLM call picks up changes
        from services.groq_client import clear_profile_cache
        clear_profile_cache()


def learn_from_session(turns: list) -> None:
    """
    Top-level function: extract profile data from a session's conversation
    and merge into user_profile.json.

    Call this after session ends (Ctrl+C / Q).
    Non-blocking: runs in a daemon thread.
    """
    if not turns:
        return

    try:
        current_profile = _load_current_profile()
        new_data = extract_profile_from_conversation(turns, current_profile)
        merge_into_profile(new_data)
    except Exception:
        pass  # Silent failure — never crash on memory


def _load_current_profile() -> dict:
    try:
        with open(USER_PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"interests": [], "communication_style": []}
