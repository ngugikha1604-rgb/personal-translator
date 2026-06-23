def extract_suggested_reply(payload: dict) -> str:
    """Return the short truthful reply phrase shown to the user."""
    return str(payload.get("reply", "")).strip()
