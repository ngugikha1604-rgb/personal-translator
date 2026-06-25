class ContextManager:
    """Holds session-level context. Reset on new session. """

    VALID_MEETING_TYPES = {"casual", "networking", "interview", "academic"}
    VALID_LANGUAGE_LEVELS = {"native", "fluent", "intermediate"}

    def __init__(self):
        self._static: dict = {}
        self._detected: dict = {}

    def set_static(
        self,
        meeting_type: str = None,
        other_name: str = None,
        other_role: str = None,
        user_goal: str = None,
        language_level: str = None,
    ):
        if meeting_type and meeting_type in self.VALID_MEETING_TYPES:
            self._static["meeting_type"] = meeting_type
        if other_name:
            self._static["other_name"] = other_name.strip()
        if other_role:
            self._static["other_role"] = other_role.strip()
        if user_goal:
            self._static["user_goal"] = user_goal.strip()
        if language_level and language_level in self.VALID_LANGUAGE_LEVELS:
            self._static["language_level"] = language_level

    def update_detected(self, context: str = None, confidence: str = None):
        """Store auto-detected context. Only persists if confidence is high."""
        if not context or not confidence:
            return
        if confidence == "high":
            self._detected = {"context": context, "confidence": confidence}
        elif confidence == "medium" and "context" not in self._detected:
            self._detected = {"context": context, "confidence": confidence}
        # Low confidence is ignored entirely

    def get_prompt_block(self) -> str:
        """Returns context block for injection into system prompt.
        Prioritises manual context; falls back to auto-detected."""
        if not self._static and not self._detected:
            return ""

        if self._static:
            lines = ["CONVERSATION CONTEXT:"]
            if "meeting_type" in self._static:
                lines.append(f"Type: {self._static['meeting_type']}")
            other_parts = []
            if "other_name" in self._static:
                other_parts.append(self._static["other_name"])
            if "other_role" in self._static:
                other_parts.append(self._static["other_role"])
            if other_parts:
                lines.append(f"Other person: {', '.join(other_parts)}")
            if "user_goal" in self._static:
                lines.append(f"Your goal: {self._static['user_goal']}")
            if "language_level" in self._static:
                lines.append(f"Their English level: {self._static['language_level']}")
            return "\n".join(lines)

        # Fallback to auto-detected
        lines = ["DETECTED CONTEXT (auto-inferred):"]
        lines.append(f"Type: {self._detected['context']}")
        lines.append(f"Confidence: {self._detected['confidence']}")
        return "\n".join(lines)

    def get_static(self) -> dict:
        return dict(self._static)

    def get_detected(self) -> dict:
        return dict(self._detected)

    def reset(self):
        self._static = {}
        self._detected = {}


# Singleton shared across routes
context_manager = ContextManager()