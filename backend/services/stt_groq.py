from config import WHISPER_MODEL
from services.groq_client import get_client
from services.stt_provider import STTProvider


class GroqSTTProvider(STTProvider):
    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        transcription = get_client().audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=WHISPER_MODEL,
            language="en",
        )
        return transcription.text.strip()
