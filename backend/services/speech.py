import time
from dataclasses import dataclass
from enum import Enum

from services.stt import transcribe_audio
from services.vad import has_speech


class SpeechStatus(Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"   # VAD filtered — no speech detected
    ERROR   = "error"     # STT failed


@dataclass(frozen=True)
class SpeechResult:
    transcript: str
    stt_ms:     int
    status:     SpeechStatus = SpeechStatus.SUCCESS


class SpeechService:
    """Audio-source independent speech processing. Returns SpeechResult, no printing."""

    def transcribe_other_audio(
        self,
        audio_bytes: bytes,
        filename:    str  = "audio.webm",
        use_vad:     bool = True,
    ) -> SpeechResult:
        if use_vad and not has_speech(audio_bytes):
            return SpeechResult(transcript="", stt_ms=0, status=SpeechStatus.SKIPPED)

        t0 = time.perf_counter()
        try:
            transcript = transcribe_audio(audio_bytes, filename)
        except Exception:
            return SpeechResult(transcript="", stt_ms=round((time.perf_counter() - t0) * 1000), status=SpeechStatus.ERROR)
        stt_ms = round((time.perf_counter() - t0) * 1000)

        return SpeechResult(transcript=transcript, stt_ms=stt_ms)

    def transcribe_user_audio(
        self,
        audio_bytes: bytes,
        filename:    str = "audio_user.wav",
    ) -> SpeechResult:
        """
        Transcribe accumulated user speech after SPACE release.
        No VAD because user deliberately held SPACE.
        """
        t0 = time.perf_counter()
        try:
            transcript = transcribe_audio(audio_bytes, filename)
        except Exception:
            return SpeechResult(transcript="", stt_ms=round((time.perf_counter() - t0) * 1000), status=SpeechStatus.ERROR)
        stt_ms = round((time.perf_counter() - t0) * 1000)

        return SpeechResult(transcript=transcript, stt_ms=stt_ms)


speech_service = SpeechService()