"""
display.py — Output layer.

Current:  terminal output with ANSI colors
Target:   optical waveguide overlay on glasses

Information hierarchy (same regardless of display medium):
    Primary   → reply  (large, immediately actionable)
    Secondary → intent (small, context)
    Hidden    → summary (LLM reasoning only, never shown)
"""

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[90m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
LINE   = "─" * 52


class Display:
    def header(self) -> None:
        print(f"\n{BOLD}Conversation Copilot{RESET}")
        print(f"{DIM}Hold SPACE to mute while you speak  •  Q to quit{RESET}")
        self._divider()

    def result(self, intent: str, reply: str) -> None:
        """Main output. Mirrors glasses overlay layout: reply primary, intent secondary."""
        self._divider()
        print(f"  {BOLD}{GREEN}{reply}{RESET}")
        print(f"  {DIM}{intent}{RESET}")
        self._divider()

    def verification(self, correct: bool, warning: str | None, llm_ms: int) -> None:
        """
        Show verification result for user's own speech.
        Green check if okay, yellow warning if issue found.
        """
        if correct:
            print(f"  {DIM}{GREEN}✓ understood  ({llm_ms}ms){RESET}")
        else:
            msg = warning or "inconsistent — double-check!"
            print(f"  {YELLOW}⚠ {msg}  ({llm_ms}ms){RESET}")

    def status(self, message: str) -> None:
        print(f"  {DIM}{message}{RESET}")

    def error(self, message: str) -> None:
        print(f"  {RED}{message}{RESET}")

    def _divider(self) -> None:
        print(f"\n{DIM}{LINE}{RESET}\n")


# Singleton — import and use directly: from services.display import display
display = Display()
