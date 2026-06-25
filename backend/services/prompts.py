"""
prompts.py — DEPRECATED.

All prompt templates have been moved into services/llm.py
as COPILOT_PROMPT and VERIFICATION_PROMPT constants.

This file kept only to prevent import errors from old references.
Will be removed after full transition.

To update prompts, edit llm.py directly.
"""

# Import from new location for backward compatibility
from services.llm import COPILOT_PROMPT, VERIFICATION_PROMPT

# Old names — kept for any stray imports
COPILOT_SYSTEM_PROMPT = COPILOT_PROMPT
VERIFICATION_SYSTEM_PROMPT = VERIFICATION_PROMPT
