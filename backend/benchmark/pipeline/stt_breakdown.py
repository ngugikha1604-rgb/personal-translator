"""stt_breakdown.py — Measure STT latency in detail.

Timing breakdown:
  audio_load_ms      — reading WAV file from disk (0 if using --record)
  audio_prepare_ms   — any conversion before API call (currently negligible for Groq)
  api_request_ms     — time inside Groq transcription API call (upload + inference + download)
  response_parse_ms  — extracting transcript text from response
  total_ms           — sum of all stages

Usage:
    cd backend
    python benchmark/pipeline/stt_breakdown.py --wav path.wav          # single file 20x
    python benchmark/pipeline/stt_breakdown.py --record                # record then transcribe 20x
    python benchmark/pipeline/stt_breakdown.py --runs 50 --wav path    # 50 iterations

Output:
    benchmark_results/stt_breakdown.jsonl
    (also prints summary to stdout)
"""

import json
import os
import sys
import time
from statistics import mean, median

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from services.stt_factory import get_stt_provider
from services.stt_provider import STTProvider
from services.groq_client import get_client

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmark_results"
)
SLEEP_BETWEEN = 0.5


def benchmark_provider(
    stt: STTProvider,
    audio_bytes: bytes,
    filename: str,
    runs: int,
) -> list:
    """Run transcription N times with detailed timing."""
    all_rows = []

    for i in range(runs):
        t_load_start = time.perf_counter()

        # Already loaded — audio_load_ms = 0 for pre-loaded bytes
        t_load_done = time.perf_counter()

        # audio_prepare_ms: any conversion before API (today: none)
        t_prep_start = time.perf_counter()
        t_prep_done = time.perf_counter()

        # api_request_ms: the Groq API call
        t_api_start = time.perf_counter()
        try:
            # We call the same method as GroqSTTProvider.transcribe()
            transcription = stt.transcribe(audio_bytes, filename)
        except Exception as exc:
            t_api_end = time.perf_counter()
            row = {
                "audio_load_ms": round((t_load_done - t_load_start) * 1000, 2),
                "audio_prepare_ms": round((t_prep_done - t_prep_start) * 1000, 2),
                "api_request_ms": round((t_api_end - t_api_start) * 1000, 2),
                "response_parse_ms": 0,
                "total_ms": round((t_api_end - t_load_start) * 1000, 2),
                "transcript_length_words": 0,
                "error": str(exc)[:200],
            }
            all_rows.append(row)
            time.sleep(SLEEP_BETWEEN)
            continue

        t_api_end = time.perf_counter()

        # response_parse_ms: extract text from response
        t_parse_start = time.perf_counter()
        transcript = transcription.strip()
        t_parse_end = time.perf_counter()

        row = {
            "audio_load_ms": round((t_load_done - t_load_start) * 1000, 2),
            "audio_prepare_ms": round((t_prep_done - t_prep_start) * 1000, 2),
            "api_request_ms": round((t_api_end - t_api_start) * 1000, 2),
            "response_parse_ms": round((t_parse_end - t_parse_start) * 1000, 2),
            "total_ms": round((t_parse_end - t_load_start) * 1000, 2),
            "transcript_length_words": len(transcript.split()),
            "transcript": transcript[:100],
        }

        all_rows.append(row)
        time.sleep(SLEEP_BETWEEN)

    return all_rows


def load_audio(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def record_audio(duration: float = 5.0) -> bytes:
    from services.audio import record_chunk
    print(f"  Recording {duration}s...")
    return record_chunk(duration)


def print_summary(all_rows: list, label: str):
    """Print timing breakdown summary."""
    stages = ["audio_load_ms", "audio_prepare_ms", "api_request_ms",
              "response_parse_ms", "total_ms"]
    ok_rows = [r for r in all_rows if "error" not in r]

    if not ok_rows:
        print(f"\n  No successful runs to summarize.")
        return

    print(f"\n  {'=' * 60}")
    print(f"  STT Breakdown — {label} ({len(ok_rows)} runs)")
    print(f"  {'=' * 60}")
    print(f"  {'Stage':25s} {'mean':>8s} {'median':>8s} {'p95':>8s} {'min':>8s} {'max':>8s}  {'% of total':>10s}")

    for stage in stages:
        vals = sorted([float(r[stage]) for r in ok_rows])
        n = len(vals)
        mn = mean(vals)
        md = median(vals)
        p95 = vals[int(n * 0.95)]
        lo = min(vals)
        hi = max(vals)

        # Percentage of total (using this row's own total)
        if stage == "total_ms":
            pct = "100%"
        else:
            # Average ratio of this stage to total
            ratios = [float(r[stage]) / float(r["total_ms"]) * 100
                      for r in ok_rows if float(r["total_ms"]) > 0]
            pct = f"{mean(ratios):5.1f}%" if ratios else "  N/A"

        print(f"  {stage:25s} {mn:>8.1f} {md:>8.1f} {p95:>8.1f} {lo:>8.1f} {hi:>8.1f}  {pct:>10s}")

    # Word count info
    wc = [r["transcript_length_words"] for r in ok_rows]
    print(f"\n  Transcript words: mean={mean(wc):.0f}  min={min(wc)}  max={max(wc)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="STT latency breakdown")
    parser.add_argument("--wav", type=str, default=None, help="WAV file")
    parser.add_argument("--record", action="store_true", help="Record from mic")
    parser.add_argument("--runs", type=int, default=20, help="Iterations")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration (s)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.wav:
        print(f"  Loading WAV: {args.wav}")
        audio = load_audio(args.wav)
        filename = os.path.basename(args.wav)
        label = os.path.basename(args.wav)
    elif args.record:
        audio = record_audio(args.duration)
        filename = "recorded_chunk.wav"
        label = f"recorded_{args.duration}s"
    else:
        print("Need --wav path.wav or --record")
        sys.exit(1)

    stt = get_stt_provider()
    print(f"  Provider: {type(stt).__name__}")
    print(f"  Audio: {len(audio)} bytes, {len(audio) / 32000:.1f}s estimated")
    print(f"  Runs: {args.runs}")
    print()

    rows = benchmark_provider(stt, audio, filename, args.runs)

    # Save
    out_path = os.path.join(OUTPUT_DIR, "stt_breakdown.jsonl")
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved: {out_path} ({len(rows)} rows)")

    # Summary
    print_summary(rows, label)


if __name__ == "__main__":
    main()
