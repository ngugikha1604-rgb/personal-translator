"""state.py — Lightweight in-memory conversation state tracker.

Tracks active conversational entities during a live session.
No database, no embeddings, no RAG — pure in-memory state.

Updates after each Analyzer call. Injected into the next LLM prompt
to provide continuous context awareness across turns.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConversationState:
    """Active conversational entities for the current session.

    Updated after each turn. Reset on new session.
    Optimized for real-time conversation — no blocking operations.
    """

    current_topic: Optional[str] = None        # What they are discussing right now
    current_question: Optional[str] = None     # The live question being addressed
    active_projects: list[str] = field(default_factory=list)   # Projects/tech mentioned
    last_intent: Optional[str] = None          # Most recent detected intent
    social_context: Optional[str] = None       # Social tone (casual, probing, etc.)
    turn_count: int = 0                        # Total turns analyzed

    def update(self, intent: str, social_signal: str, turn_text: str) -> None:
        """Update state from one analyzed turn.

        Called after each Analyzer.analyze().
        Lightweight field assignment only — no LLM calls, no DB.
        """
        self.turn_count += 1

        if intent:
            self.last_intent = intent

        if social_signal:
            self.social_context = social_signal

        # Topic from intent
        if intent:
            self.current_topic = intent

        # Track the most recent question from the turn text
        if turn_text and turn_text.rstrip().endswith("?"):
            self.current_question = turn_text

        # Lightweight entity extraction from turn text (rule-based, <0.1ms)
        if turn_text:
            self._extract_projects(turn_text)

    def _extract_projects(self, text: str) -> None:
        """Simple heuristic: anything after 'your' in common patterns."""
        import re
        # Match patterns like "your project on X" or "your work on X"
        m = re.search(r'(?:your|the|this)\s+(project|work|app|system|tool|platform)\s+(?:called|named|on|about)\s+(\w+(?:\s+\w+)?)', text, re.IGNORECASE)
        if m:
            project = m.group(2).strip()
            if project and project not in self.active_projects:
                self.active_projects.append(project)

    def get_prompt_block(self) -> str:
        """Returns formatted state block for LLM prompt injection.

        Only includes fields that have meaningful values.
        Designed for glanceable reading by LLM — not the user.
        """
        parts = []
        if self.current_topic:
            parts.append(f"Active topic: {self.current_topic}")
        if self.current_question:
            parts.append(f"Live question: {self.current_question}")
        if self.active_projects:
            # Keep only last 3 to avoid prompt bloat
            for p in self.active_projects[-3:]:
                parts.append(f"Project mentioned: {p}")
        if self.last_intent:
            parts.append(f"Last intent: {self.last_intent}")
        if self.social_context:
            parts.append(f"Social tone: {self.social_context}")
        if not parts:
            return ""
        parts.append(f"Total turns: {self.turn_count}")
        return "CONVERSATION STATE:\n" + "\n".join(parts)

    def reset(self) -> None:
        """Clear all state. Call on new session."""
        self.current_topic = None
        self.current_question = None
        self.active_projects.clear()
        self.last_intent = None
        self.social_context = None
        self.turn_count = 0


# Singleton shared across the session
conversation_state = ConversationState()
