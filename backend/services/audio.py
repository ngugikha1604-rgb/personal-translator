"""
audio.py — Microphone capture service.

Current:  sounddevice (cross-platform desktop mic)
Target:   hardware mic array with beamforming (Mic 1, other person)

Interface contract:
    record_chunk(duration) → WAV bytes
Callers do not change when hardware implementation is swapped.
"""

import wave
from io import BytesIO

import numpy as np
import sounddevice as sd

SAMPLE_RATE   = 16_000   # Hz — Whisper optimal
CHANNELS      = 1        # Mono
DTYPE         = "int16"  # 16-bit PCM
CHUNK_SECONDS = 2.0      # Duration per chunk — tune empirically after latency testing


def record_chunk(duration: float = CHUNK_SECONDS) -> bytes:
    """Record `duration` seconds from default mic. Returns WAV bytes."""
    frames = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocking=True,
    )
    return _to_wav(frames)


def _to_wav(audio: np.ndarray) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)           # int16 = 2 bytes per sample
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    buf.seek(0)
    return buf.read()
