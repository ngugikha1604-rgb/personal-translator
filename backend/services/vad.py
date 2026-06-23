"""
VAD (Voice Activity Detection) service.

Current implementation: lightweight energy-based heuristic.
Planned upgrade: Silero-VAD (roadmap item #6).

Interface is intentionally stable — callers don't change when implementation is swapped.
"""


def has_speech(audio_bytes: bytes, min_size_bytes: int = 8000) -> bool:
    """Return True if the audio chunk likely contains speech.

    Heuristic: samples raw byte variance as a proxy for audio energy.
    Works as a basic pre-Whisper gate; not codec-aware.
    Replace body with Silero-VAD when upgrading (roadmap #6).
    """
    if len(audio_bytes) < min_size_bytes:
        return False

    # Sample every 4th byte to approximate waveform energy cheaply
    sample = audio_bytes[::4]
    if not sample:
        return False

    mean = sum(sample) / len(sample)
    variance = sum((b - mean) ** 2 for b in sample) / len(sample)

    # Silence / codec headers typically variance < 100
    return variance > 100
