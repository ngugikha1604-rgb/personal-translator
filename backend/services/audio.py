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
CHUNK_SECONDS   = 5.0    # Duration per chunk — enough context for complete utterances
WAV_HEADER_SIZE = 44     # Standard PCM WAV header size (bytes)


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


def extract_pcm(wav_bytes: bytes) -> bytes:
    """Strip WAV header, return raw PCM bytes only.

    WAV files cannot be naively concatenated — each has its own 44-byte header.
    Use this when accumulating chunks to join later.
    """
    return wav_bytes[WAV_HEADER_SIZE:]


def pcm_to_wav(pcm_bytes: bytes) -> bytes:
    """Wrap concatenated raw PCM bytes in a single valid WAV header.

    Pair with extract_pcm():
        wav = pcm_to_wav(b"".join(extract_pcm(c) for c in chunks))
    """
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    return _to_wav(audio.reshape(-1, CHANNELS))
