"""stt_latency.py — Measure Groq Whisper transcription latency independently.

Measures STT-only latency by recording audio, then transcribing multiple times.

Usage:
    cd backend
    python benchmark/pipeline/stt_latency.py --record          # capture + transcribe 20x
    python benchmark/pipeline/stt_latency.py --wav path.wav   # transcribe existing file 20x
    python benchmark/pipeline/stt_latency.py --runs 50         # 50 transcriptions (default 20)

Output:
    benchmark_results/stt_latency.jsonl
"""

import json
import os
import sys
import time
from statistics import mean, median

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from services.stt_factory import get_stt_provider
from services.vad import has_speech

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmark_results"
)
SLEEP_BETWEEN = 0.5  # avoid rate limits

# ── Helpers ────────────────────────────────────────────────────

def load_audio(path: str) -> bytes:
    """Read WAV file."""
    with open(path, "rb") as f:
        return f.read()


def record_audio(duration: float = 5.0) -> bytes:
    """Record from microphone and return WAV bytes."""
    from services.audio import record_chunk
    print(f"  Recording {duration}s of audio from mic...")
    audio = record_chunk(duration)
    print(f"  Captured {len(audio)} bytes")
    return audio


# ── Benchmark ─────────────────────────────────────────────────

def transcribe_once(stt, audio_bytes: bytes, filename: str) -> dict:
    """Single transcription with timing."""
    t0 = time.perf_counter()
    try:
        transcript = stt.transcribe(audio_bytes, filename)
    except Exception as exc:
        t1 = time.perf_counter()
        return {
            "stt_ms": round((t1 - t0) * 1000),
            "transcript": "",
            "transcript_length_chars": 0,
            "transcript_length_words": 0,
            "error": str(exc)[:200],
        }
    t1 = time.perf_counter()

    return {
        "stt_ms": round((t1 - t0) * 1000),
        "transcript": transcript,
        "transcript_length_chars": len(transcript),
        "transcript_length_words": len(transcript.split()),
    }


def run_benchmark(audio_bytes: bytes, filename: str, runs: int):
    """Transcribe the same audio N times, save results."""
    stt = get_stt_provider()

    print(f"\n  STT provider: {type(stt).__name__}")
    print(f"  Audio bytes: {len(audio_bytes)}")
    print(f"  Runs: {runs}")

    # Check VAD
    speech_detected = has_speech(audio_bytes)
    print(f"  VAD detected speech: {speech_detected}")
    if not speech_detected:
        print("  [WARN] VAD says no speech — results will show silence transcription")

    all_results = []
    for i in range(runs):
        print(f"    [{i+1}/{runs}] transcribing...", end="")
        result = transcribe_once(stt, audio_bytes, filename)
        print(f" stt_ms={result['stt_ms']}")
        all_results.append(result)
        time.sleep(SLEEP_BETWEEN)

    # Evaluate
    stt_times = sorted([r["stt_ms"] for r in all_results])
    n = len(stt_times)

    print(f"\n  {'=' * 50}")
    print(f"  STT latency — {n} samples")
    print(f"  {'=' * 50}")
    print(f"    count:  {n}")
    print(f"    mean:   {mean(stt_times):.0f}ms")
    print(f"    median: {median(stt_times):.0f}ms")
    print(f"    p95:    {stt_times[int(n * 0.95)]:.0f}ms")
    print(f"    min:    {min(stt_times):.0f}ms")
    print(f"    max:    {max(stt_times):.0f}ms")

    # Save JSONL
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "stt_latency.jsonl")
    with open(out_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\n  Saved: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="STT latency benchmark")
    parser.add_argument("--wav", type=str, default=None, help="WAV file to transcribe")
    parser.add_argument("--record", action="store_true", help="Record from mic")
    parser.add_argument("--runs", type=int, default=20, help="Number of transcriptions")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration (s)")
    args = parser.parse_args()

    if args.wav:
        audio = load_audio(args.wav)
        filename = os.path.basename(args.wav)
    elif args.record:
        audio = record_audio(args.duration)
        filename = "recorded_chunk.wav"
    else:
        print("Need --wav path.wav or --record to capture audio from mic")
        sys.exit(1)

    run_benchmark(audio, filename, args.runs)


if __name__ == "__main__":
    main()
