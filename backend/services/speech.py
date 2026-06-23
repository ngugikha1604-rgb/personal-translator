import time
from dataclasses import dataclass

from services.stt import transcribe_audio
from services.vad import has_speech


@dataclass(frozen=True)
class SpeechResult:
    transcript: str
    stt_ms: int
    skipped: bool = False


class SpeechService:
    """Audio-source independent speech processing service."""

    def transcribe_other_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.webm",
        use_vad: bool = True,
    ) -> SpeechResult:
        if use_vad and not has_speech(audio_bytes):
            print("[VAD] Skipped - no speech detected")
            return SpeechResult(transcript="", stt_ms=0, skipped=True)

        t0 = time.perf_counter()
        transcript = transcribe_audio(audio_bytes, filename)
        stt_ms = round((time.perf_counter() - t0) * 1000)
        print(f"[LAT] STT: {stt_ms}ms | transcript: \"{transcript}\"")
        return SpeechResult(transcript=transcript, stt_ms=stt_ms)


speech_service = SpeechService()
