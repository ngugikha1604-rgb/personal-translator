"""
stt_provider.py — Abstract base class for all STT backends.

To add a new provider:
    1. Create services/stt_<name>.py
    2. Subclass STTProvider and implement transcribe()
    3. Register the name in stt_factory.py
    4. Set STT_PROVIDER=<name> in .env
"""
from abc import ABC, abstractmethod


class STTProvider(ABC):
    """Common interface for all speech-to-text backends."""

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """Convert WAV audio bytes to text.

        Args:
            audio_bytes: WAV-formatted audio bytes. Must be valid WAV with a
                         proper header (produced by audio.py's pcm_to_wav()).
            filename:    Filename hint. Cloud APIs (e.g. Groq) use the file
                         extension for format detection. Local providers
                         (e.g. Faster-Whisper) ignore it.

        Returns:
            Transcribed text stripped of leading/trailing whitespace.
            Returns "" if audio contains no speech.

        Raises:
            Exception: on transcription failure. Caller (SpeechService) handles
                       all exceptions and converts them to SpeechStatus.ERROR.
        """
        ...
