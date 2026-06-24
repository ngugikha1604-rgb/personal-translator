"""
groq_client.py — Single shared Groq client instance.

Created once, reused everywhere. Includes timeout configuration.
Profile cached in memory — no disk reads on every turn.
"""

import json
import os
import threading
from typing import Optional

import httpx
from groq import Groq

from config import GROQ_API_KEY, USER_PROFILE_PATH

_client: Optional[Groq] = None
_lock = threading.Lock()
_profile_cache: Optional[dict] = None


def get_client() -> Groq:
    """Return shared Groq client. Creates on first call (lazy init)."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:  # double-checked locking
                _client = Groq(
                    api_key=GROQ_API_KEY,
                    http_client=httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0)),
                )
    return _client


def get_user_profile() -> dict:
    """Return cached user profile. Loads from disk once."""
    global _profile_cache
    if _profile_cache is None:
        try:
            with open(USER_PROFILE_PATH, "r", encoding="utf-8") as f:
                _profile_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _profile_cache = {"interests": [], "communication_style": []}
    return _profile_cache


def clear_profile_cache() -> None:
    """Force reload on next access. Call when profile file is updated."""
    global _profile_cache
    _profile_cache = None