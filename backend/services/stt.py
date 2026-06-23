from groq import Groq
from config import GROQ_API_KEY, WHISPER_MODEL

client = Groq(api_key=GROQ_API_KEY)


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    transcription = client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model=WHISPER_MODEL,
        language="en"
    )
    return transcription.text.strip()
