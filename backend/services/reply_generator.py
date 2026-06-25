"""reply_generator.py — Generates user-facing reply from analyzer output.

Phase 1: extracts reply from the analyzer's single LLM output.
Phase 2 (future): can make a separate tiny local model call
                  using analysis fields for context.
"""

from services.analyzer import AnalysisResult


class ReplyGenerator:
    """Produces the user-facing reply string from analysis result."""

    def generate(self, analysis: AnalysisResult) -> str:
        """Extract reply from analyzer output.

        Phase 1: reply is already embedded in the analyzer's raw JSON output.
        Future phase: can make a separate cheap LLM call using
                      analysis fields (intent, goal, signal) as context.
        """
        if analysis._parsed and "reply" in analysis._parsed:
            return str(analysis._parsed["reply"]).strip()
        return ""
