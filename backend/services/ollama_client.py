"""
ollama_client.py — Single shared Ollama client instance.

Created once, reused everywhere.
"""

import os
import threading
from typing import Optional

import httpx
from ollama import Client

_client: Optional[Client] = None
_lock = threading.Lock()


def get_client() -> Client:
    """Return shared Ollama client. Creates on first call (lazy init)."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:  # double-checked locking
                _client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    return _client