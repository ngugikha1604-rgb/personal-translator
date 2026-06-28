"""stt_thread_scaling.py — Measure FasterWhisper CPU thread scaling.

Creates a fresh WhisperModel for each thread count (1, 2, 4, 8, 16).
Runs 5 transcriptions per configuration.
Uses tiny.en for fast iterations.

Do NOT modify production code.

Usage:
    cd backend
    python benchmark/pipeline/stt_thread_scaling.py --record
    python benchmark/pipeline/stt_thread_scaling.py --wav path.wav

Output:
    benchmark_results/stt_thread_scaling.jsonl
    benchmark_results/stt_thread_scaling_report.json
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

from services.stt_faster_whisper import _wav_bytes_to_float32

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmark_results"
)
SLEEP_AFTER_MODEL = 1.0  # cool-down between model configs

THREAD_COUNTS = [1, 2, 4, 8, 16]
MODEL_NAME = "tiny.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
RUNS_PER_CONFIG = 5
from faster_whisper import WhisperModel


def get_audio_duration_ms(audio_bytes: bytes) -> int:
    try:
        with wave.open(BytesIO(audio_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return int(frames / rate * 1000) if rate else 0
    except Exception:
        return 0


def transcribe_with_threads(audio_float32: np.ndarray, cpu_threads: int) -> list:
    """Run RUNS_PER_CONFIG transcriptions with a specific thread count.

    Creates a fresh model for each thread count.
    Returns list of per-run timing dicts.
    """
    import gc

    gc.collect()  # free previous model

    t_load_start = time.perf_counter()
    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=cpu_threads,
    )
    t_load_ready = time.perf_counter()
    load_ms = (t_load_ready - t_load_start) * 1000

    rows = []
    for i in range(RUNS_PER_CONFIG):
        t0 = time.perf_counter()
        segments, _info = model.transcribe(
            audio_float32,
            language="en",
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        t1 = time.perf_counter()
        texts = [seg.text for seg in segments]
        t2 = time.perf_counter()

        transcript = " ".join(texts).strip()
        row = {
            "cpu_threads": cpu_threads,
            "run": i + 1,
            "gen_call_ms": round((t1 - t0) * 1000, 2),
            "gen_iter_ms": round((t2 - t1) * 1000, 2),
            "total_ms": round((t2 - t0) * 1000, 2),
            "transcript": transcript[:60],
            "segments": len(texts),
            "words": len(transcript.split()),
        }
        rows.append(row)

    # Free model
    del model
    gc.collect()

    return rows, load_ms


def run_benchmark(audio_bytes: bytes):
    """Run thread scaling benchmark."""
    duration_ms = get_audio_duration_ms(audio_bytes)
    audio_float32 = _wav_bytes_to_float32(audio_bytes)

    audio_seconds = duration_ms / 1000.0

    print(f"\n  Model:     {MODEL_NAME}")
    print(f"  Device:    {DEVICE}")
    print(f"  Compute:   {COMPUTE_TYPE}")
    print(f"  Audio:     ~{duration_ms}ms ({audio_seconds:.1f}s)")
    print(f"  Runs/conf: {RUNS_PER_CONFIG}")
    print()

    all_rows = []
    config_summaries = []

    for n in THREAD_COUNTS:
        print(f"  ── Threads={n} ──")
        rows, load_ms = transcribe_with_threads(audio_float32, n)
        all_rows.extend(rows)

        totals = sorted([r["total_ms"] for r in rows])
        calls = sorted([r["gen_call_ms"] for r in rows])
        iters = sorted([r["gen_iter_ms"] for r in rows])

        mn_t = mean(totals)
        md_t = median(totals)
        p95_t = totals[int(len(totals) * 0.95)]
        lo_t = min(totals)
        hi_t = max(totals)

        # Realtime factor (RTF = processing_time / audio_duration)
        rtf = mn_t / (duration_ms if duration_ms > 0 else 1)

        config_summaries.append({
            "cpu_threads": n,
            "load_model_ms": round(load_ms, 1),
            "total_ms": {
                "mean": round(mn_t, 1),
                "median": round(md_t, 1),
                "p95": round(p95_t, 1),
                "min": round(lo_t, 1),
                "max": round(hi_t, 1),
            },
            "gen_call_ms": {
                "mean": round(mean(calls), 1),
                "median": round(median(calls), 1),
                "p95": round(calls[int(len(calls)*0.95)], 1),
            },
            "gen_iter_ms": {
                "mean": round(mean(iters), 1),
                "median": round(median(iters), 1),
                "p95": round(iters[int(len(iters)*0.95)], 1),
            },
            "realtime_factor": round(rtf, 3),
            "notes": (
                "real-time capable" if rtf < 1.0
                else "near real-time" if rtf < 2.0
                else f"slow ({rtf:.1f}x audio duration)"
            ),
        })

        rtf_str = f"RTF={rtf:.3f}"
        print(f"    load={load_ms:.0f}ms  total={mn_t:.0f}±{(hi_t-lo_t)/2:.0f}ms  {rtf_str}")
        for r in rows:
            print(f"      run {r['run']}: call={r['gen_call_ms']:.1f}ms  iter={r['gen_iter_ms']:.1f}ms  "
                  f"total={r['total_ms']:.0f}ms  words={r['words']}")

        time.sleep(SLEEP_AFTER_MODEL)

    # ── Save results ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    jsonl_path = os.path.join(OUTPUT_DIR, "stt_thread_scaling.jsonl")
    with open(jsonl_path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n  Saved: {jsonl_path}")

    # ── Find optimal thread count ──
    best = min(config_summaries, key=lambda c: c["total_ms"]["mean"])
    best_threads = best["cpu_threads"]

    # Diminishing returns: find where adding threads gives <10% improvement
    dim_return = best_threads
    prev_time = None
    for cs in sorted(config_summaries, key=lambda c: c["cpu_threads"]):
        cur = cs["total_ms"]["mean"]
        if prev_time is not None:
            improvement = (prev_time - cur) / prev_time * 100
            if improvement < 10 and cs["cpu_threads"] > 1:
                dim_return = cs["cpu_threads"]
                break
        prev_time = cur

    # ── Report ──
    report = {
        "audio_duration_ms": duration_ms,
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "runs_per_config": RUNS_PER_CONFIG,
        "configurations": config_summaries,
        "recommendation": {
            "fastest_thread_count": best_threads,
            "fastest_mean_ms": best["total_ms"]["mean"],
            "fastest_rtf": best["realtime_factor"],
            "diminishing_returns_at": dim_return,
            "recommended_threads": (
                best_threads
                if best_threads <= dim_return
                else dim_return
            ),
            "rationale": (
                f"cpu_threads={best_threads} gives lowest mean latency ({best['total_ms']['mean']:.0f}ms). "
                f"Beyond {dim_return} threads, additional cores provide <10% improvement. "
                f"Recommended: {best_threads if best_threads <= dim_return else dim_return} threads."
            ),
        },
        "@recommend_note": (
            "Set CPU_THREADS=<N> in .env or pass cpu_threads=N to faster_whisper initialization. "
            "This is NOT the same as CPU count — it controls internal thread pool for the model."
        ),
    }

    report_path = os.path.join(OUTPUT_DIR, "stt_thread_scaling_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")

    # ── Print summary ──
    print(f"\n  {'=' * 65}")
    print(f"  Thread Scaling Summary")
    print(f"  {'=' * 65}")
    print(f"  {'Threads':>8s} {'Mean':>8s} {'Median':>8s} {'P95':>8s} {'Min':>8s} {'Max':>8s} {'RTF':>8s}")
    print(f"  {'-'*56}")
    for cs in sorted(config_summaries, key=lambda c: c["cpu_threads"]):
        t = cs["total_ms"]
        print(f"  {cs['cpu_threads']:>8d} {t['mean']:>8.1f} {t['median']:>8.1f} "
              f"{t['p95']:>8.1f} {t['min']:>8.1f} {t['max']:>8.1f} {cs['realtime_factor']:>8.3f}")
    print(f"\n  'real-time' means RTF < 1.0 (processes audio faster than it plays)")
    print()
    print(f"  Fastest: cpu_threads={report['recommendation']['fastest_thread_count']} "
          f"({report['recommendation']['fastest_mean_ms']:.0f}ms)")
    print(f"  Recommendation: {report['recommendation']['rationale']}")
    print(f"\n  {'=' * 65}")


def load_audio(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def record_audio(duration: float = 5.0) -> bytes:
    from services.audio import record_chunk
    print(f"  Recording {duration}s...")
    return record_chunk(duration)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FasterWhisper thread scaling")
    parser.add_argument("--wav", type=str, default=None, help="WAV file")
    parser.add_argument("--record", action="store_true", help="Record from mic")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration (s)")
    args = parser.parse_args()

    if args.wav:
        audio = load_audio(args.wav)
    elif args.record:
        audio = record_audio(args.duration)
    else:
        print("Need --wav path.wav or --record")
        sys.exit(1)

    run_benchmark(audio)


if __name__ == "__main__":
    main()
