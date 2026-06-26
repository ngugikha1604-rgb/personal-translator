"""
DEPRECATED — superseded by stt_groq.GroqSTTProvider via stt_factory.

Kept temporarily for backward compatibility. Will be removed in a follow-up PR.
"""
from config import WHISPER_MODEL
from services.groq_client import get_client


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    transcription = get_client().audio.transcriptions.create(
        file=(filename, audio_bytes),
        model=WHISPER_MODEL,
        language="en"
    )
    return transcription.text.strip()
