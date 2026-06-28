"""stt_duration_scaling.py — Measure FasterWhisper latency vs audio duration.

Purpose:
    Determine whether STT latency scales linearly with audio duration,
    and estimate fixed overhead vs per-second processing cost.

Durations: 0.5, 1, 2, 3, 4, 5, 6, 8, 10 seconds.
Each runs 10 iterations (configurable via --runs).

Supports --record (captures per-duration audio) and --wav path.wav.

Output:
    benchmark_results/stt_duration_scaling.jsonl
    benchmark_results/stt_duration_scaling_report.json

Usage:
    cd backend
    python benchmark/pipeline/stt_duration_scaling.py --record
    python benchmark/pipeline/stt_duration_scaling.py --record --runs 5
    python benchmark/pipeline/stt_duration_scaling.py --wav path.wav
"""

import json
import os
import sys
import time
import math
from statistics import mean, median, stdev
from io import BytesIO
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from services.stt_faster_whisper import _wav_bytes_to_float32
from faster_whisper import WhisperModel

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmark_results"
)
SLEEP_BETWEEN = 0.3

DURATIONS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
MODEL_NAME = "tiny.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"


def record_audio(duration: float) -> bytes:
    """Record `duration` seconds from mic, return WAV bytes."""
    from services.audio import record_chunk
    return record_chunk(duration)


def load_wav(path: str) -> bytes:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"WAV not found: {path}")
    with open(path, "rb") as f:
        return f.read()


def run_duration(
    model: WhisperModel,
    audio_float32: np.ndarray,
    duration_s: float,
    runs: int,
) -> list:
    """Run `runs` transcriptions at a given audio duration.

    Returns list of per-run result dicts.
    """
    rows = []
    for i in range(runs):
        t0 = time.perf_counter()
        segments_gen, _info = model.transcribe(
            audio_float32,
            language="en",
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        t1 = time.perf_counter()
        texts = [s.text for s in segments_gen]
        t2 = time.perf_counter()
        transcript = " ".join(texts).strip()
        rows.append({
            "duration_s": duration_s,
            "wav_to_float_ms": round((t0 - (t0 - 0.001)) * 1000, 2),  # placeholder — actual is outside
            "transcribe_call_ms": round((t1 - t0) * 1000, 2),
            "segment_iteration_ms": round((t2 - t1) * 1000, 2),
            "total_ms": round((t2 - t0) * 1000, 2),
            "segments_found": len(texts),
            "transcript_length_words": len(transcript.split()),
            "transcript": transcript[:80],
        })
        time.sleep(SLEEP_BETWEEN)
    return rows


def linear_fit(x: list, y: list) -> dict:
    """Simple linear regression: y = a + b*x.

    Returns {fixed_overhead_ms, ms_per_second_audio, r_squared}.
    """
    n = len(x)
    if n < 2:
        return {"fixed_overhead_ms": 0, "ms_per_second_audio": 0, "r_squared": 0}
    x_bar = mean(x)
    y_bar = mean(y)
    num = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x, y))
    den = sum((xi - x_bar) ** 2 for xi in x)
    b = num / den if den != 0 else 0
    a = y_bar - b * x_bar
    # R-squared
    ss_res = sum((yi - (a + b * xi)) ** 2 for xi, yi in zip(x, y))
    ss_tot = sum((yi - y_bar) ** 2 for yi in y)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return {
        "fixed_overhead_ms": round(a, 1),
        "ms_per_second_audio": round(b, 1),
        "r_squared": round(r2, 4),
    }


def run_benchmark(audio_by_duration: dict, runs_per: int):
    """Run duration scaling benchmark.

    audio_by_duration: {duration_s: wav_bytes, ...}
    """
    actual_durations = sorted(audio_by_duration.keys())
    print(f"  Model:     {MODEL_NAME}")
    print(f"  Device:    {DEVICE}")
    print(f"  Compute:   {COMPUTE_TYPE}")
    print(f"  Durations: {actual_durations}")
    print(f"  Runs/conf: {runs_per}")
    print()

    # Load model once — same instance for all durations
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

    all_rows = []
    summary = []

    for dur in actual_durations:
        wav_bytes = audio_by_duration[dur]
        # Measure wav_to_float separately (not per-run since it's deterministic)
        tw0 = time.perf_counter()
        audio_f32 = _wav_bytes_to_float32(wav_bytes)
        tw1 = time.perf_counter()
        wav_to_float_ms = (tw1 - tw0) * 1000

        rows = run_duration(model, audio_f32, dur, runs_per)

        # Inject the wav_to_float measurement into each row
        for row in rows:
            row["wav_to_float_ms"] = round(wav_to_float_ms, 2)

        all_rows.extend(rows)

        totals = sorted([r["total_ms"] for r in rows])
        mn = mean(totals)
        md = median(totals)
        p95 = totals[int(len(totals) * 0.95)]
        lo = min(totals)
        hi = max(totals)
        sd = stdev(totals) if len(totals) > 1 else 0
        dur_ms = dur * 1000
        rtf = mn / dur_ms if dur_ms > 0 else 0
        lat_per_s = mn / dur if dur > 0 else 0

        summary.append({
            "duration_s": dur,
            "mean_total_ms": round(mn, 1),
            "median_total_ms": round(md, 1),
            "p95_total_ms": round(p95, 1),
            "min_total_ms": round(lo, 1),
            "max_total_ms": round(hi, 1),
            "std_total_ms": round(sd, 1),
            "rtf": round(rtf, 3),
            "latency_per_second": round(lat_per_s, 1),
        })

        print(f"  {dur:5.1f}s  mean={mn:>7.1f}ms  median={md:>7.1f}ms  "
              f"p95={p95:>7.1f}ms  RTF={rtf:.3f}  {lat_per_s:.0f}ms/s")

    # ── Linear fit ──
    x = [s["duration_s"] for s in summary]
    y = [s["mean_total_ms"] for s in summary]
    fit = linear_fit(x, y)

    # ── Interpretation ──
    if fit["r_squared"] > 0.95:
        linearity = "Latency scales almost linearly with audio duration."
    elif fit["r_squared"] > 0.8:
        linearity = "Latency scales roughly linearly with audio duration."
    else:
        linearity = "Latency does not scale linearly with audio duration."

    fixed_pct = (fit["fixed_overhead_ms"] / (fit["fixed_overhead_ms"] + fit["ms_per_second_audio"] * 5)) * 100 \
        if (fit["fixed_overhead_ms"] + fit["ms_per_second_audio"] * 5) > 0 else 0

    if fit["fixed_overhead_ms"] > fit["ms_per_second_audio"] * 2:
        overhead_note = "Fixed overhead dominates — Dynamic Chunking may offer minimal benefit."
    elif fixed_pct < 20:
        overhead_note = "Processing cost per second dominates — reducing chunk length directly reduces latency."
    else:
        overhead_note = "Mixed behavior — reducing audio duration will proportionally reduce total latency."

    interpretation = (
        f"Estimated fixed overhead: {fit['fixed_overhead_ms']:.0f} ms\n"
        f"Estimated processing cost: {fit['ms_per_second_audio']:.0f} ms per second of audio\n"
        f"R²: {fit['r_squared']:.4f}\n"
        f"{linearity}\n"
        f"{overhead_note}"
    )

    # ── Build report ──
    report = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "runs_per_duration": runs_per,
        "results": summary,
        "linear_fit": fit,
        "interpretation": interpretation,
        "recommendation": {
            "dynamic_chunking_benefit": (
                "High" if fixed_pct < 30
                else "Moderate" if fixed_pct < 60
                else "Low"
            ),
            "fixed_overhead_pct_at_5s": round(fixed_pct, 1),
            "note": (
                "Reducing audio chunk length is likely to reduce latency proportionally "
                "if per-second processing cost dominates."
            ),
        },
    }

    # ── Save ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    jsonl_path = os.path.join(OUTPUT_DIR, "stt_duration_scaling.jsonl")
    with open(jsonl_path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n  Saved: {jsonl_path}")

    report_path = os.path.join(OUTPUT_DIR, "stt_duration_scaling_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")

    # ── Console summary ──
    print(f"\n  {'=' * 75}")
    print(f"  Duration Scaling Summary")
    print(f"  {'=' * 75}")
    h = f"  {'Duration':>8s} {'Mean':>8s} {'Median':>8s} {'P95':>8s} {'RTF':>8s} {'ms/s':>8s}"
    print(h)
    print(f"  {'-' * len(h)}")
    for s in summary:
        print(f"  {s['duration_s']:>7.1f}s {s['mean_total_ms']:>8.1f} {s['median_total_ms']:>8.1f} "
              f"{s['p95_total_ms']:>8.1f} {s['rtf']:>8.3f} {s['latency_per_second']:>8.1f}")
    print(f"\n  {interpretation}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FasterWhisper duration scaling")
    parser.add_argument("--wav", type=str, default=None, help="Single WAV file (uses its duration)")
    parser.add_argument("--record", action="store_true", help="Record per-duration audio")
    parser.add_argument("--runs", type=int, default=10, help="Iterations per duration")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override: benchmark a single duration only (e.g. --duration 3)")
    args = parser.parse_args()

    if args.wav:
        # Single file: determine duration from header and use it
        wav_bytes = load_wav(args.wav)
        with wave.open(BytesIO(wav_bytes), "rb") as wf:
            dur_s = wf.getnframes() / wf.getframerate() if wf.getframerate() > 0 else 5.0
        dur_s = round(dur_s, 1)
        audio_by_duration = {dur_s: wav_bytes}
        print(f"  Single WAV: ~{dur_s}s ({len(wav_bytes)} bytes)")
    elif args.record:
        print(f"  Recording {len(DURATIONS)} audio clips from mic...")
        audio_by_duration = {}
        for dur in DURATIONS:
            if args.duration is not None and abs(dur - args.duration) > 0.01:
                continue
            print(f"    {dur:.1f}s ...", end="", flush=True)
            audio_by_duration[dur] = record_audio(dur)
            print(f" {len(audio_by_duration[dur])} bytes")
    else:
        # Generate synthetic speech-like audio (no mic needed)
        print(f"  Generating synthetic audio for {len(DURATIONS)} durations...")
        audio_by_duration = {}
        sample_rate = 16000
        for dur in DURATIONS:
            if args.duration is not None and abs(dur - args.duration) > 0.01:
                continue
            n_samples = int(dur * sample_rate)
            t = np.linspace(0, dur, n_samples, endpoint=False)
            # Amplitude-modulated noise to simulate speech
            noise = np.random.randn(n_samples) * 0.3
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
            samples = (noise * mod * 32767).astype(np.int16).reshape(-1, 1)
            from services.audio import _to_wav
            audio_by_duration[dur] = _to_wav(samples)
            print(f"    {dur:.1f}s -> {len(audio_by_duration[dur])} bytes")

    run_benchmark(audio_by_duration, args.runs)


if __name__ == "__main__":
    main()
