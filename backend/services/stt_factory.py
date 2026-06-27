from __future__ import annotations

from config import STT_PROVIDER
from services.stt_provider import STTProvider

_provider: STTProvider | None = None


def get_stt_provider() -> STTProvider:
    global _provider
    if _provider is not None:
        return _provider
    if STT_PROVIDER == "faster_whisper":
        from services.stt_faster_whisper import FasterWhisperProvider
        _provider = FasterWhisperProvider()
        return _provider
    if STT_PROVIDER == "groq":
        from services.stt_groq import GroqSTTProvider
        _provider = GroqSTTProvider()
        return _provider
    raise ValueError(f"Unknown STT_PROVIDER: {STT_PROVIDER!r}")
