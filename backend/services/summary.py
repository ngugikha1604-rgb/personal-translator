def extract_summary(payload: dict) -> str:
    """Return the internal one-sentence conversation summary."""
    return str(payload.get("summary", "")).strip()
