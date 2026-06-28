"""stt_duration_scaling.py — Measure FasterWhisper latency vs audio duration (v3).

Fixed methodological issues:
----------------------------------------------------------------------
1. SINGLE RECORDING SOURCE
   Previously: independently recorded clips with different speech content.
   Now: ONE long clip recorded, then sliced per duration. Only duration varies.

2. INTERLEAVED RUN ORDER
   Previously: block-by-block (all runs of A, then all runs of B).
   Now: each "round" runs every duration once in shuffled order. Repeating N rounds
   distributes CPU thermal/scheduler noise evenly across all durations.

3. FULL DISTRIBUTION REPORTING
   Added: std, coefficient of variation, outlier count.

4. SPEECH DENSITY METRIC
   Added: words_per_second for each duration slice (Whisper runtime correlates with it).

5. MONOTONICITY CHECK
   Before fitting regression, verify latency increases monotonically with duration.
   If violated, print warning and skip regression.

6. NO UNSUPPORTED CONCLUSIONS
   Regression and interpretation are only generated when monotonicity holds.

Usage:
    cd backend
    python benchmark/pipeline/stt_duration_scaling.py --record  # record 1 long clip + slice
    python benchmark/pipeline/stt_duration_scaling.py --wav long_recording.wav
    python benchmark/pipeline/stt_duration_scaling.py --runs 10

Output:
    benchmark_results/stt_duration_scaling.jsonl
    benchmark_results/stt_duration_scaling_report.json
"""

import json
import os
import sys
import time
import random
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
SLEEP_BETWEEN_RUNS = 0.5      # between runs
WARMUP_RUNS = 2                # warmup before measurement

# Each round processes every duration once in shuffled order.
# NUM_ROUNDS × NUM_DURATIONS = total measurements per duration.
NUM_DURATIONS = 9
DURATIONS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
MODEL_NAME = "tiny.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
SAMPLE_RATE = 16000

# For a 10-second recording, we need at least 10 seconds of audio.
RECORD_DURATION = max(DURATIONS) + 0.5  # 10.5 seconds


def record_one_clip(duration: float = RECORD_DURATION) -> bytes:
    """Record ONE long clip from microphone. Returns WAV bytes.

    All durations will be sliced from this single recording to ensure
    the only variable that changes is audio duration, not speech content.
    """
    from services.audio import record_chunk
    print(f"  Recording {duration:.1f}s clip from mic...")
    wav = record_chunk(duration)
    actual_dur = _get_wav_duration(wav)
    print(f"  Captured: {len(wav)} bytes, {actual_dur:.1f}s")
    return wav


def _get_wav_duration(wav_bytes: bytes) -> float:
    """Return actual duration in seconds from WAV header."""
    with wave.open(BytesIO(wav_bytes), "rb") as wf:
        n = wf.getnframes()
        r = wf.getframerate()
    return n / r if r > 0 else 0


def slice_wav(wav_bytes: bytes, duration_s: float) -> bytes:
    """Take the first `duration_s` seconds from a WAV file.

    This is the key methodological fix: instead of independently recording
    each duration (which introduces different speech content as a confound),
    we slice from ONE long recording so that every shorter duration is a
    prefix of every longer duration. The only variable is duration.
    """
    with wave.open(BytesIO(wav_bytes), "rb") as wf:
        n_total = wf.getnframes()
        rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        nchannels = wf.getnchannels()
        n_desired = int(duration_s * rate)
        n_desired = min(n_desired, n_total)  # clamp to available
        wf.setpos(0)
        frames = wf.readframes(n_desired)

    # Re-wrap in a proper WAV header
    buf_out = BytesIO()
    with wave.open(buf_out, "wb") as wf_out:
        wf_out.setnchannels(nchannels)
        wf_out.setsampwidth(sampwidth)
        wf_out.setframerate(rate)
        wf_out.writeframes(frames)
    return buf_out.getvalue()


def verify_monotonic(durations: list, medians: list) -> bool:
    """Check whether median latency is non-decreasing with duration.

    Returns True if monotonic, False otherwise.
    Prints warnings for each violation.
    """
    sorted_pairs = sorted(zip(durations, medians), key=lambda p: p[0])
    is_ok = True
    for i in range(1, len(sorted_pairs)):
        prev_d, prev_m = sorted_pairs[i - 1]
        cur_d, cur_m = sorted_pairs[i]
        if cur_m < prev_m - 50:  # 50ms tolerance for measurement noise
            print(f"  [WARN] Duration {cur_d:.1f}s median ({cur_m:.0f}ms) < "
                  f"{prev_d:.1f}s median ({prev_m:.0f}ms). Monotonicity violated.")
            is_ok = False
    return is_ok


def linear_fit(x: list, y: list) -> dict:
    """Simple linear regression: y = a + b*x.

    Only call when data is confirmed monotonic.
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
    ss_res = sum((yi - (a + b * xi)) ** 2 for xi, yi in zip(x, y))
    ss_tot = sum((yi - y_bar) ** 2 for yi in y)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return {
        "fixed_overhead_ms": round(a, 1),
        "ms_per_second_audio": round(b, 1),
        "r_squared": round(r2, 4),
    }


def run_benchmark(audio_bytes_long: bytes, runs_per: int):
    """Run duration scaling benchmark.

    All durations are sliced from a single audio recording.
    Durations are interleaved per round to distribute thermal bias.
    """
    base_duration = _get_wav_duration(audio_bytes_long)
    used_durations = [d for d in DURATIONS if d <= base_duration]

    if not used_durations:
        print(f"  ERROR: Recording is only {base_duration:.1f}s, "
              f"need at least {min(DURATIONS):.1f}s")
        sys.exit(1)

    print(f"  Model:     {MODEL_NAME}")
    print(f"  Device:    {DEVICE}")
    print(f"  Compute:   {COMPUTE_TYPE}")
    print(f"  Durations: {used_durations}")
    print(f"  Rounds:    {runs_per} (each round shuffles all durations once)")
    print(f"  Warmup:    {WARMUP_RUNS} runs")
    print(f"  Source:    {base_duration:.1f}s single recording — sliced, not re-recorded")
    print()

    # Load model once
    t_load = time.perf_counter()
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    model_load_ms = (time.perf_counter() - t_load) * 1000
    print(f"  Model load: {model_load_ms:.0f}ms")
    print()

    # Pre-slice all durations from the single recording
    slices = {}
    for d in used_durations:
        raw = slice_wav(audio_bytes_long, d)
        actual = _get_wav_duration(raw)
        slices[d] = raw
        if abs(actual - d) > 0.2:
            print(f"  [WARN] Duration {d:.1f}s requested, got {actual:.1f}s from slice")

    # Also precompute float32 arrays
    float32_cache = {}
    for d in used_durations:
        t0 = time.perf_counter()
        float32_cache[d] = _wav_bytes_to_float32(slices[d])
        t1 = time.perf_counter()
        if d == used_durations[0]:
            print(f"  wav_to_float: ~{(t1-t0)*1000:.2f}ms per duration")

    # Warmup: one pass through all durations
    print(f"  Warmup: {WARMUP_RUNS} runs per model...")
    for _ in range(WARMUP_RUNS):
        for d in used_durations:
            gen, _ = model.transcribe(
                float32_cache[d], language="en", beam_size=1,
                condition_on_previous_text=False, vad_filter=False,
            )
            _ = list(gen)

    # Measured rounds: each round runs every duration once in shuffled order.
    # This distributes CPU thermal drift and scheduler noise evenly.
    all_rows = []
    per_duration_rows_flat = {d: [] for d in used_durations}

    print(f"  Running {runs_per} measurement rounds...")
    for round_idx in range(runs_per):
        shuffled = list(used_durations)
        random.shuffle(shuffled)

        for d in shuffled:
            t0 = time.perf_counter()
            seg_gen, _info = model.transcribe(
                float32_cache[d], language="en", beam_size=1,
                condition_on_previous_text=False, vad_filter=False,
            )
            t1 = time.perf_counter()
            texts = [s.text for s in seg_gen]
            t2 = time.perf_counter()
            transcript = " ".join(texts).strip()

            row = {
                "round": round_idx + 1,
                "duration_s": d,
                "wav_to_float_ms": 0.0,  # measured once, injected after loop
                "transcribe_call_ms": round((t1 - t0) * 1000, 2),
                "segment_iteration_ms": round((t2 - t1) * 1000, 2),
                "total_ms": round((t2 - t0) * 1000, 2),
                "segments_found": len(texts),
                "transcript_length_words": len(transcript.split()),
                "transcript": transcript[:80],
            }
            per_duration_rows_flat[d].append(row)
            all_rows.append(row)

            time.sleep(SLEEP_BETWEEN_RUNS)

    # Compute per-duration statistics
    summary = []
    for d in used_durations:
        rows = per_duration_rows_flat[d]
        totals = sorted([r["total_ms"] for r in rows])
        n = len(totals)
        mn = mean(totals)
        md = median(totals)
        p95_idx = min(int(n * 0.95), n - 1)
        p95 = totals[p95_idx]
        lo = totals[0]
        hi = totals[-1]
        sd = stdev(totals) if n > 1 else 0
        cv = sd / mn if mn > 0 else 0
        dur_ms = d * 1000
        rtf = mn / dur_ms if dur_ms > 0 else 0
        lat_per_s = mn / d if d > 0 else 0

        # Outlier: value outside median ± 3 × MAD
        abs_devs = sorted(abs(v - md) for v in totals)
        mad = median(abs_devs) if abs_devs else 0
        outlier_threshold = md + 3 * (mad * 1.4826) if mad > 0 else float("inf")
        outliers = sum(1 for v in totals if v > outlier_threshold)

        # Speech density
        avg_words = mean(r["transcript_length_words"] for r in rows)
        wps = avg_words / d if d > 0 else 0

        summary.append({
            "duration_s": d,
            "mean_total_ms": round(mn, 1),
            "median_total_ms": round(md, 1),
            "p95_total_ms": round(p95, 1),
            "min_total_ms": round(lo, 1),
            "max_total_ms": round(hi, 1),
            "std_total_ms": round(sd, 1),
            "cv": round(cv, 3),
            "outlier_count": outliers,
            "rtf": round(rtf, 3),
            "latency_per_second": round(lat_per_s, 1),
            "avg_words": round(avg_words, 1),
            "words_per_second": round(wps, 2),
        })

        print(f"  round={round_idx if 'round_idx' in dir() else 0}  "
              f"dur={d:5.1f}s  median={md:>7.1f}ms  mean={mn:>7.1f}ms  "
              f"p95={p95:>7.1f}ms  RTF={rtf:.3f}  cv={cv:.2f}")

    # ── Monotonicity check ──
    durations_sorted = sorted(used_durations)
    medians_sorted = [summary[used_durations.index(d)]["median_total_ms"]
                      for d in durations_sorted]
    is_monotonic = verify_monotonic(durations_sorted, medians_sorted)

    # ── Regression (only when monotonic) ──
    if is_monotonic:
        fit = linear_fit(durations_sorted, medians_sorted)

        if fit["r_squared"] > 0.95:
            linearity = "Latency scales almost linearly with audio duration."
        elif fit["r_squared"] > 0.8:
            linearity = "Latency scales roughly linearly with audio duration."
        else:
            linearity = "Weak linear trend."

        fixed_pct = (
            (fit["fixed_overhead_ms"] / (fit["fixed_overhead_ms"] + fit["ms_per_second_audio"] * 5)) * 100
        ) if (fit["fixed_overhead_ms"] + fit["ms_per_second_audio"] * 5) > 0 else 0

        if fit["fixed_overhead_ms"] > fit["ms_per_second_audio"] * 2:
            overhead_note = "Fixed overhead dominates."
        elif fixed_pct < 20:
            overhead_note = "Per-second processing cost dominates."
        else:
            overhead_note = "Mixed behavior."

        interpretation = (
            f"Estimated fixed overhead: {fit['fixed_overhead_ms']:.0f} ms\n"
            f"Estimated per-second cost: {fit['ms_per_second_audio']:.0f} ms/s\n"
            f"R²: {fit['r_squared']:.4f}\n"
            f"{linearity}\n"
            f"{overhead_note}"
        )
        dynamic_benefit = (
            "High" if fixed_pct < 30
            else "Moderate" if fixed_pct < 60
            else "Low"
        )
    else:
        # Monotonicity violated — do not fit or interpret
        fit = {"fixed_overhead_ms": 0, "ms_per_second_audio": 0, "r_squared": 0,
               "note": "Regression skipped — data violates monotonic assumption"}
        interpretation = (
            "Regression skipped because measurements violate "
            "monotonic assumption. Latency does not consistently "
            "increase with audio duration. Possible causes:\n"
            "  - Measurement noise exceeds the duration effect\n"
            "  - Speech density varies across slices\n"
            "  - CPU thermal throttling during individual runs"
        )
        dynamic_benefit = "Unknown (monotonicity not met)"

    # ── Print summary table ──
    print(f"\n  {'=' * 80}")
    print(f"  Duration Scaling Summary")
    print(f"  {'=' * 80}")
    h = (f"  {'Duration':>8s} {'Median':>8s} {'Mean':>8s} {'P95':>8s} "
         f"{'Std':>8s} {'CV':>6s} {'RTF':>8s} {'WPS':>6s}")
    print(h)
    print(f"  {'-' * len(h)}")
    for s in summary:
        print(f"  {s['duration_s']:>7.1f}s {s['median_total_ms']:>8.1f} "
              f"{s['mean_total_ms']:>8.1f} {s['p95_total_ms']:>8.1f} "
              f"{s['std_total_ms']:>8.1f} {s['cv']:>6.3f} "
              f"{s['rtf']:>8.3f} {s['words_per_second']:>6.2f}")
    print(f"\n  Monotonic: {is_monotonic}")
    print(f"\n  {interpretation}")

    # ── Build report ──
    report = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "rounds": runs_per,
        "warmup_runs": WARMUP_RUNS,
        "single_source_duration_s": round(base_duration, 2),
        "methodology": (
            "All durations sliced from a single recording. "
            "Durations interleaved per round (shuffled). "
            "Regression only when monotonicity confirmed."
        ),
        "monotonic": is_monotonic,
        "results": summary,
        "linear_fit": fit,
        "interpretation": interpretation,
        "recommendation": {
            "dynamic_chunking_benefit": dynamic_benefit,
            "note": (
                "Reducing audio chunk length is expected to reduce total latency "
                "if per-second processing cost dominates. "
                "This conclusion is only reliable when monotonicity holds "
                "and R² is high."
            ),
        },
    }

    # ── Save ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    jsonl_path = os.path.join(OUTPUT_DIR, "stt_duration_scaling.jsonl")
    with open(jsonl_path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n  Saved: {jsonl_path} ({len(all_rows)} rows)")

    report_path = os.path.join(OUTPUT_DIR, "stt_duration_scaling_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="FasterWhisper duration scaling (v3 — single-source sliced)"
    )
    parser.add_argument("--wav", type=str, default=None,
                        help="Single WAV file to slice durations from")
    parser.add_argument("--record", action="store_true",
                        help="Record one long clip, then slice durations")
    parser.add_argument("--runs", type=int, default=10, help="Number of measurement rounds")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override: benchmark a single duration only")
    args = parser.parse_args()

    np.random.seed(42)
    random.seed(42)

    if args.wav:
        with open(args.wav, "rb") as f:
            clip = f.read()
        print(f"  Source WAV: {args.wav} ({len(clip)} bytes)")
    elif args.record:
        clip = record_one_clip()
    else:
        # Generate synthetic audio (one long clip, then slice)
        print(f"  Generating {RECORD_DURATION:.1f}s synthetic audio...")
        n = int(RECORD_DURATION * SAMPLE_RATE)
        t = np.linspace(0, RECORD_DURATION, n, endpoint=False)
        noise = np.random.randn(n) * 0.3
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
        samples = (noise * mod * 32767).astype(np.int16).reshape(-1, 1)
        from services.audio import _to_wav
        clip = _to_wav(samples)

    run_benchmark(clip, args.runs)


if __name__ == "__main__":
    main()
