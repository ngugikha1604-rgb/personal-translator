def extract_intent(payload: dict) -> str:
    """Return the compact intent phrase from a parsed copilot payload."""
    return str(payload.get("intent", "")).strip()
