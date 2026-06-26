from config import STT_PROVIDER
from services.stt_provider import STTProvider


def get_stt_provider() -> STTProvider:
    if STT_PROVIDER == "faster_whisper":
        from services.stt_faster_whisper import FasterWhisperProvider

        return FasterWhisperProvider()
    if STT_PROVIDER == "groq":
        from services.stt_groq import GroqSTTProvider

        return GroqSTTProvider()
    raise ValueError(f"Unknown STT_PROVIDER: {STT_PROVIDER!r}")
