"""stt_inference_probe.py — Benchmark FasterWhisper sub-stage timing.

Measures the three internal stages of transcription separately:
  A. wav_to_float_ms      WAV → float32 conversion
  B. transcribe_call_ms   model.transcribe() call (generator creation)
  C. segment_iteration_ms generator consumption (actual inference)

Does NOT modify production code. Accesses the provider's internal
model object for fine-grained timing.

Usage:
    cd backend
    python benchmark/pipeline/stt_inference_probe.py --record
    python benchmark/pipeline/stt_inference_probe.py --wav path.wav
    python benchmark/pipeline/stt_inference_probe.py --runs 10

Output:
    benchmark_results/stt_inference_probe.jsonl
"""

import json
import os
import sys
import time
from statistics import mean, median
from io import BytesIO
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from services.stt_factory import get_stt_provider
from services.stt_faster_whisper import _wav_bytes_to_float32
from config import FASTER_WHISPER_MODEL, FASTER_WHISPER_DEVICE, FASTER_WHISPER_COMPUTE_TYPE

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmark_results"
)
SLEEP_BETWEEN = 0.5

import multiprocessing


def get_audio_duration_ms(audio_bytes: bytes) -> int:
    """Estimate audio duration from WAV header."""
    try:
        with wave.open(BytesIO(audio_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return int(frames / rate * 1000) if rate else 0
    except Exception:
        return 0


def probe_once(provider, audio_bytes: bytes, filename: str) -> dict:
    """Measure A/B/C stages of one transcription."""
    # A. wav_to_float_ms
    t0 = time.perf_counter()
    audio = _wav_bytes_to_float32(audio_bytes)
    t1 = time.perf_counter()
    wav_to_float_ms = (t1 - t0) * 1000

    # B. transcribe_call_ms — generator creation only
    t2 = time.perf_counter()
    segments, _info = provider._model.transcribe(
        audio,
        language="en",
        beam_size=1,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    t3 = time.perf_counter()
    transcribe_call_ms = (t3 - t2) * 1000

    # C. segment_iteration_ms — actual inference (generator consumption)
    t4 = time.perf_counter()
    texts = [segment.text for segment in segments]
    t5 = time.perf_counter()
    segment_iteration_ms = (t5 - t4) * 1000

    transcript = " ".join(texts).strip()
    total_ms = (t5 - t0) * 1000

    return {
        "wav_to_float_ms": round(wav_to_float_ms, 2),
        "transcribe_call_ms": round(transcribe_call_ms, 2),
        "segment_iteration_ms": round(segment_iteration_ms, 2),
        "total_ms": round(total_ms, 2),
        "segments_found": len(texts),
        "transcript_length_words": len(transcript.split()),
        "transcript": transcript[:80],
    }


def run_benchmark(audio_bytes: bytes, filename: str, runs: int):
    """Run probe N times on the same audio."""
    provider = get_stt_provider()

    # Verify we're using FasterWhisper
    provider_type = type(provider).__name__
    if "FasterWhisper" not in provider_type:
        print(f"  [WARN] Provider is {provider_type}, not FasterWhisperProvider.")
        print(f"  [WARN] Set STT_PROVIDER=faster_whisper in .env")
        print(f"  [WARN] This benchmark measures FasterWhisper only.")
        print()

    audio_duration_ms = get_audio_duration_ms(audio_bytes)
    cpu_count = multiprocessing.cpu_count()

    print(f"\n  Provider: {provider_type}")
    print(f"  Model:    {FASTER_WHISPER_MODEL}")
    print(f"  Device:   {FASTER_WHISPER_DEVICE}")
    print(f"  Compute:  {FASTER_WHISPER_COMPUTE_TYPE}")
    print(f"  CPU:      {cpu_count} cores")
    print(f"  Audio:    {len(audio_bytes)} bytes, ~{audio_duration_ms}ms duration")
    print(f"  Runs:     {runs}")
    print()

    all_rows = []
    for i in range(runs):
        print(f"    [{i+1}/{runs}] probing...", end="", flush=True)
        row = probe_once(provider, audio_bytes, filename)
        all_rows.append(row)
        # Show dominant stage
        stages = ["wav_to_float_ms", "transcribe_call_ms", "segment_iteration_ms"]
        dominant = max(stages, key=lambda s: row[s])
        dom_pct = row[dominant] / row["total_ms"] * 100 if row["total_ms"] > 0 else 0
        print(f" total={row['total_ms']:.0f}ms  {dominant}={row[dominant]:.0f}ms ({dom_pct:.1f}%)")
        time.sleep(SLEEP_BETWEEN)

    # ── Summary ──
    stages = ["wav_to_float_ms", "transcribe_call_ms", "segment_iteration_ms", "total_ms"]

    print(f"\n  {'=' * 65}")
    print(f"  Faster Whisper Inference Probe")
    print(f"  {'=' * 65}")
    print(f"  {'Stage':25s} {'mean':>9s} {'median':>9s} {'p95':>9s} {'min':>9s} {'max':>9s}")
    print(f"  {'-'*65}")

    for stage in stages:
        vals = sorted([float(r[stage]) for r in all_rows])
        n = len(vals)
        mn = mean(vals)
        md = median(vals)
        p95 = vals[int(n * 0.95)]
        lo = min(vals)
        hi = max(vals)
        print(f"  {stage:25s} {mn:>9.2f} {md:>9.2f} {p95:>9.2f} {lo:>9.2f} {hi:>9.2f}")

    # Dominant stage identification
    print(f"\n  {'=' * 65}")
    print(f"  Dominant stage analysis")
    print(f"  {'=' * 65}")
    dominant_counts = {}
    for stage in ["wav_to_float_ms", "transcribe_call_ms", "segment_iteration_ms"]:
        dominant_counts[stage] = 0
    for r in all_rows:
        dominant = max(["wav_to_float_ms", "transcribe_call_ms", "segment_iteration_ms"],
                       key=lambda s: r[s])
        dominant_counts[dominant] += 1
    for stage, count in sorted(dominant_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_rows) * 100
        bar = "█" * max(1, int(pct / 10))
        print(f"  {bar} {stage:25s}: {count}/{len(all_rows)} runs ({pct:.0f}%)")
    last_dominant = max(dominant_counts.items(), key=lambda x: x[1])[0]
    print(f"\n  → Primary bottleneck: {last_dominant}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "stt_inference_probe.jsonl")
    with open(out_path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n  Saved: {out_path}")


def load_audio(path: str) -> bytes:
    if not os.path.isfile(path):
        abs_path = os.path.abspath(path)
        raise FileNotFoundError(f"WAV file not found: {abs_path}")
    with open(path, "rb") as f:
        return f.read()


def record_audio(duration: float = 5.0) -> bytes:
    from services.audio import record_chunk
    print(f"  Recording {duration}s from mic...")
    return record_chunk(duration)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FasterWhisper inference probe")
    parser.add_argument("--wav", type=str, default=None, help="WAV file")
    parser.add_argument("--record", action="store_true", help="Record from mic")
    parser.add_argument("--runs", type=int, default=10, help="Iterations")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration (s)")
    args = parser.parse_args()

    # Ensure we're using faster_whisper
    os.environ.setdefault("STT_PROVIDER", "faster_whisper")

    if args.wav:
        audio = load_audio(args.wav)
        filename = os.path.basename(args.wav)
    elif args.record:
        audio = record_audio(args.duration)
        filename = "recorded_chunk.wav"
    else:
        print("Need --wav path.wav or --record")
        sys.exit(1)

    run_benchmark(audio, filename, args.runs)


if __name__ == "__main__":
    main()
