import io
import wave

import numpy as np
from faster_whisper import WhisperModel

from config import (
    FASTER_WHISPER_COMPUTE_TYPE,
    FASTER_WHISPER_CPU_THREADS,
    FASTER_WHISPER_DEVICE,
    FASTER_WHISPER_MODEL,
)
from services.stt_provider import STTProvider


def _wav_bytes_to_float32(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    return audio / 32768.0


class FasterWhisperProvider(STTProvider):
    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        self.model_size = model_size or FASTER_WHISPER_MODEL
        self.device = device or FASTER_WHISPER_DEVICE
        self.compute_type = compute_type or FASTER_WHISPER_COMPUTE_TYPE
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=FASTER_WHISPER_CPU_THREADS,
        )

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        audio = _wav_bytes_to_float32(audio_bytes)
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
