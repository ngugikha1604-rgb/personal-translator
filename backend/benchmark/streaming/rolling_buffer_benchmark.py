"""rolling_buffer_benchmark.py — Evaluate FasterWhisper with rolling window streaming.

Research question:
  "How does rolling buffer size affect STT accuracy, latency and computational cost?"

Methodology:
  Use ONE complete audio recording. Simulate streaming offline by sliding a rolling
  window at a fixed interval. Only the rolling buffer size changes between configurations.

  Each rolling window is an independent STT request. NO transcript merging, NO
  stabilization, NO hypothesis revision. Every decode is evaluated independently.

  Streaming interval is fixed (default 500ms) for the entire benchmark.
  Only one independent variable: buffer size.

  Ground truth uses word-level timestamps. Words are assigned to the window
  containing their midpoint: midpoint = (start + end) / 2.

  WER/CER are computed per-window, then aggregated as a weighted average
  (weighted by reference word count). Silent windows (zero reference words)
  are excluded from accuracy metrics.

  Synthetic audio mode reports WER/CER as "N/A" — no ground truth available.

Usage:
    cd backend

    # Accuracy + latency (requires ground truth transcript)
    python benchmark/streaming/rolling_buffer_benchmark.py --wav audio.wav --transcript transcript.json

    # Latency only (synthetic audio)
    python benchmark/streaming/rolling_buffer_benchmark.py --runs 3

    # Latency only (recorded audio without transcript)
    python benchmark/streaming/rolling_buffer_benchmark.py --record --runs 3

Output:
    benchmark_results/rolling_buffer_results.jsonl
    benchmark_results/rolling_buffer_report.json
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
SLEEP_BETWEEN_WINDOWS = 0.05
SLEEP_BETWEEN_CONFIGS = 1.0
WARMUP_RUNS = 2

# Default streaming interval — must remain fixed throughout the benchmark
STREAMING_INTERVAL_MS = 500
STREAMING_INTERVAL_S = STREAMING_INTERVAL_MS / 1000.0

# Only buffer size varies between configurations
BUFFER_SIZES_MS = [500, 1000, 1500, 2000, 3000, 4000]
MODEL_NAME = "tiny.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
SAMPLE_RATE = 16000


# ── Levenshtein-based WER/CER ─────────────────────────────

def _levenshtein(ref: list, hyp: list) -> int:
    """Compute Levenshtein distance between two sequences."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def compute_wer(reference: str, hypothesis: str) -> tuple:
    """Compute (WER, CER) given reference and hypothesis strings.

    WER = word-level Levenshtein / reference_word_count
    CER = character-level Levenshtein / reference_char_count

    Returns (wer, cer) as floats in [0, 1].
    If reference is empty, returns (None, None).
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if not ref_words:
        return (None, None)

    ref_chars = list(" ".join(ref_words))
    hyp_chars = list(" ".join(hyp_words))

    word_dist = _levenshtein(ref_words, hyp_words)
    char_dist = _levenshtein(ref_chars, hyp_chars)

    return (word_dist / len(ref_words), char_dist / len(ref_chars))


# ── Ground truth handling ─────────────────────────────────

def load_transcript(path: str) -> list:
    """Load word-level transcript JSON.

    Expected format:
      [{"word": "hello", "start": 0.21, "end": 0.54}, ...]
    """
    with open(path, "r") as f:
        data = json.load(f)
    # Normalise: ensure every entry has word, start, end
    validated = []
    for entry in data:
        if not all(k in entry for k in ("word", "start", "end")):
            print(f"  [WARN] Skipping invalid transcript entry: {entry}")
            continue
        validated.append(entry)
    return validated


def words_in_window(
    transcript: list, window_start: float, window_end: float
) -> list:
    """Extract ground truth words whose MIDPOINT falls in [window_start, window_end).

    Assignment rule: midpoint = (start + end) / 2.
    This is documented in the report methodology.
    """
    result = []
    for entry in transcript:
        midpoint = (entry["start"] + entry["end"]) / 2.0
        if window_start <= midpoint < window_end:
            result.append(entry["word"])
    return result


# ── Streaming simulation ──────────────────────────────────

def simulate_streaming_windows(
    audio_duration_s: float, buffer_s: float, interval_s: float
) -> list:
    """Compute rolling window [start, end) timestamps.

    Starts at t=0. Each step advances by interval_s.
    Window = [current_time, current_time + buffer_s).
    Stops when window_start >= audio_duration_s.
    Returns list of (window_start, window_end).
    """
    windows = []
    t = 0.0
    while t < audio_duration_s:
        end = t + buffer_s
        # Clamp end to audio duration (don't pad with silence)
        end = min(end, audio_duration_s)
        windows.append((t, end))
        t += interval_s
        # If buffer is smaller than interval, we can still move forward
        # but we ensure we don't duplicate windows
    return windows


# ── WAV slicing ──────────────────────────────────────────

def slice_wav_interval(wav_bytes: bytes, start_s: float, end_s: float) -> bytes:
    """Extract a WAV segment from start_s to end_s.

    Returns bytes with a valid WAV header.
    """
    with wave.open(BytesIO(wav_bytes), "rb") as wf:
        rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        nchannels = wf.getnchannels()
        n_total = wf.getnframes()

    start_frame = int(start_s * rate)
    end_frame = min(int(end_s * rate), n_total)
    n_frames = max(0, end_frame - start_frame)

    if n_frames <= 0:
        return b""

    with wave.open(BytesIO(wav_bytes), "rb") as wf:
        wf.setpos(start_frame)
        frames = wf.readframes(n_frames)

    buf = BytesIO()
    with wave.open(buf, "wb") as wf_out:
        wf_out.setnchannels(nchannels)
        wf_out.setsampwidth(sampwidth)
        wf_out.setframerate(rate)
        wf_out.writeframes(frames)
    return buf.getvalue()


def _get_wav_duration(wav_bytes: bytes) -> float:
    if not wav_bytes:
        return 0.0
    with wave.open(BytesIO(wav_bytes), "rb") as wf:
        n = wf.getnframes()
        r = wf.getframerate()
    return n / r if r > 0 else 0


# ── Helpers ──────────────────────────────────────────────

def try_psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        return None


def get_cpu_rss(psutil_mod) -> tuple:
    if psutil_mod is None:
        return (0.0, 0.0)
    proc = psutil_mod.Process()
    cpu = proc.cpu_percent(interval=0)
    rss = proc.memory_info().rss / (1024 * 1024)
    return (cpu, rss)


def record_audio(duration: float = 10.0) -> bytes:
    from services.audio import record_chunk
    return record_chunk(duration)


def generate_synthetic_audio(duration_s: float) -> bytes:
    """Generate synthetic noise audio (no speech — WER=N/A)."""
    n = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n, endpoint=False)
    noise = np.random.randn(n) * 0.3
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    samples = (noise * mod * 32767).astype(np.int16)
    from services.audio import _to_wav
    return _to_wav(samples.reshape(-1, 1))


# ── Run one configuration ───────────────────────────────

def run_configuration(
    model: WhisperModel,
    wav_bytes: bytes,
    buffer_ms: int,
    interval_s: float,
    transcript: list,
    psutil_mod,
) -> dict:
    """Run streaming simulation for one buffer size.

    Returns aggregated results + per-window rows.
    """
    audio_float32 = _wav_bytes_to_float32(wav_bytes)
    duration_s = _get_wav_duration(wav_bytes)
    buffer_s = buffer_ms / 1000.0

    windows = simulate_streaming_windows(duration_s, buffer_s, interval_s)

    per_window_rows = []
    all_wer_vals = []
    all_cer_vals = []
    all_ref_wc = []
    all_latencies = []

    for win_start, win_end in windows:
        # Slice audio for this window
        window_wav = slice_wav_interval(wav_bytes, win_start, win_end)
        if not window_wav:
            continue
        window_f32 = _wav_bytes_to_float32(window_wav)

        # CPU/RSS before
        cpu_before, rss_before = get_cpu_rss(psutil_mod)

        # Transcribe
        t0 = time.perf_counter()
        gen, _ = model.transcribe(
            window_f32, language="en", temperature=0.0, beam_size=1,
            condition_on_previous_text=False, vad_filter=False,
        )
        texts = list(gen)
        t1 = time.perf_counter()

        cpu_after, rss_after = get_cpu_rss(psutil_mod)

        decode_latency_ms = (t1 - t0) * 1000
        window_duration_ms = (win_end - win_start) * 1000
        rtf = decode_latency_ms / window_duration_ms if window_duration_ms > 0 else 0

        transcript_text = " ".join(seg.text for seg in texts).strip()

        # Ground truth for this window
        ref_words = words_in_window(transcript, win_start, win_end)
        ref_text = " ".join(ref_words)

        wer_val, cer_val = compute_wer(ref_text, transcript_text)

        cpu_avg = (cpu_before + cpu_after) / 2
        rss_max = max(rss_before, rss_after)

        row = {
            "buffer_ms": buffer_ms,
            "window_start": round(win_start, 3),
            "window_end": round(win_end, 3),
            "window_duration_ms": round(window_duration_ms, 1),
            "decode_latency_ms": round(decode_latency_ms, 2),
            "rtf": round(rtf, 4),
            "cpu_percent": round(cpu_avg, 1),
            "rss_mb": round(rss_max, 1),
            "hypothesis": transcript_text[:100],
            "reference": ref_text[:100],
        }

        # Only add WER/CER if ground truth was available
        if wer_val is not None and len(ref_words) > 0:
            row["wer"] = round(wer_val, 4)
            row["cer"] = round(cer_val, 4)
            all_wer_vals.append(wer_val)
            all_cer_vals.append(cer_val)
            all_ref_wc.append(len(ref_words))
        else:
            row["wer"] = None
            row["cer"] = None

        all_latencies.append(decode_latency_ms)
        per_window_rows.append(row)

        time.sleep(SLEEP_BETWEEN_WINDOWS)

    # ── Aggregate ──
    lats = sorted(all_latencies)
    n = len(lats)
    mn_lat = mean(lats)
    md_lat = median(lats)
    p95_lat = lats[min(int(n * 0.95), n - 1)]
    sd_lat = stdev(lats) if n > 1 else 0
    cv_lat = sd_lat / mn_lat if mn_lat > 0 else 0

    # RTF (mean)
    rtfs = [r["rtf"] for r in per_window_rows]
    mn_rtf = mean(rtfs) if rtfs else 0

    # Backlog
    backlog_ratio_p95 = p95_lat / (interval_s * 1000)
    miss_count = sum(1 for la in all_latencies if la > interval_s * 1000)
    backlog_miss_rate = miss_count / n * 100 if n > 0 else 0

    # Accuracy (weighted average of per-window WER/CER)
    if all_ref_wc:
        total_wc = sum(all_ref_wc)
        avg_wer = sum(w * c for w, c in zip(all_wer_vals, all_ref_wc)) / total_wc
        avg_cer = sum(w * c for w, c in zip(all_cer_vals, all_ref_wc)) / total_wc
    else:
        avg_wer = None
        avg_cer = None

    # CPU/RSS
    cpus = [r["cpu_percent"] for r in per_window_rows]
    rsses = [r["rss_mb"] for r in per_window_rows]

    # Redundancy (derived)
    redundancy_ratio = buffer_s / interval_s

    config_result = {
        "buffer_ms": buffer_ms,
        "total_windows": n,
        "redundancy_ratio": round(redundancy_ratio, 2),
        "decode_latency_ms": {
            "mean": round(mn_lat, 1),
            "median": round(md_lat, 1),
            "p95": round(p95_lat, 1),
            "min": round(min(lats), 1) if lats else 0,
            "max": round(max(lats), 1) if lats else 0,
            "std": round(sd_lat, 1),
            "cv": round(cv_lat, 3),
        },
        "rtf": {"mean": round(mn_rtf, 3)},
        "backlog_ratio_p95": round(backlog_ratio_p95, 2),
        "backlog_miss_rate_pct": round(backlog_miss_rate, 1),
        "cpu_percent": {
            "mean": round(mean(cpus), 1) if cpus else 0,
            "observed_max": round(max(cpus), 1) if cpus else 0,
        },
        "rss_mb": {
            "mean": round(mean(rsses), 1) if rsses else 0,
            "observed_max": round(max(rsses), 1) if rsses else 0,
        },
    }

    if avg_wer is not None:
        config_result["wer"] = round(avg_wer, 4)
        config_result["cer"] = round(avg_cer, 4)
    else:
        config_result["wer"] = None
        config_result["cer"] = None

    return config_result, per_window_rows


# ── Run benchmark ────────────────────────────────────────

def run_benchmark(
    wav_bytes: bytes,
    transcript: list,
    buffer_sizes: list,
    interval_ms: int,
):
    """Run the rolling buffer benchmark for all buffer sizes."""
    psutil_mod = try_psutil()
    has_psutil = psutil_mod is not None
    interval_s = interval_ms / 1000.0

    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

    duration_s = _get_wav_duration(wav_bytes)
    has_transcript = transcript is not None and len(transcript) > 0

    print(f"  Model:           {MODEL_NAME}")
    print(f"  Device:          {DEVICE}")
    print(f"  Compute:         {COMPUTE_TYPE}")
    print(f"  Audio:           {duration_s:.1f}s")
    print(f"  Streaming interval: {interval_ms}ms (fixed)")
    print(f"  Buffer sizes:    {buffer_sizes} ms")
    print(f"  Ground truth:    {'YES' if has_transcript else 'NO — WER/CER=N/A'}")
    print(f"  psutil:          {'available' if has_psutil else 'NOT available'}")
    print()

    all_rows = []
    config_results = []

    for bs in buffer_sizes:
        print(f"  ── Buffer={bs}ms ──")
        result, rows = run_configuration(
            model, wav_bytes, bs, interval_s, transcript, psutil_mod
        )
        config_results.append(result)
        all_rows.extend(rows)

        wer_str = "N/A" if result.get("wer") is None else f"{result['wer']:.3f}"
        print(f"  latency={result['decode_latency_ms']['mean']:>6.0f}ms  "
              f"p95={result['decode_latency_ms']['p95']:>6.0f}ms  "
              f"RTF={result['rtf']['mean']:.3f}  "
              f"backlog_p95={result['backlog_ratio_p95']:.2f}  "
              f"miss={result['backlog_miss_rate_pct']:.0f}%  "
              f"wer={wer_str}  "
              f"decodes={result['total_windows']}")
        time.sleep(SLEEP_BETWEEN_CONFIGS)

    # ── Tradeoff summary (NO winner) ──
    # Present per-metric extremes. No single "optimal" buffer.
    tradeoff_lines = [
        "No single 'optimal' rolling buffer. Tradeoffs by metric:",
    ]
    if has_transcript:
        best_wer = min(config_results, key=lambda c: c["wer"] if c["wer"] is not None else float("inf"))
        tradeoff_lines.append(
            f"  Best accuracy (WER):  {best_wer['buffer_ms']}ms "
            f"(WER={best_wer['wer']:.3f})"
        )
    best_rtf = min(config_results, key=lambda c: c["rtf"]["mean"])
    tradeoff_lines.append(
        f"  Best RTF (efficiency):      {best_rtf['buffer_ms']}ms "
        f"(RTF={best_rtf['rtf']['mean']:.3f})"
    )
    lowest_delay = min(config_results, key=lambda c: c["decode_latency_ms"]["mean"])
    tradeoff_lines.append(
        f"  Lowest mean latency:         {lowest_delay['buffer_ms']}ms "
        f"({lowest_delay['decode_latency_ms']['mean']:.0f}ms)"
    )
    best_backlog = min(config_results, key=lambda c: c["backlog_ratio_p95"])
    tradeoff_lines.append(
        f"  Best backlog ratio (P95):    {best_backlog['buffer_ms']}ms "
        f"({best_backlog['backlog_ratio_p95']:.2f})"
    )
    lowest_redundancy = min(config_results, key=lambda c: c["redundancy_ratio"])
    tradeoff_lines.append(
        f"  Lowest redundancy:           {lowest_redundancy['buffer_ms']}ms "
        f"({lowest_redundancy['redundancy_ratio']:.1f}x)"
    )
    tradeoff_lines.append(
        "The appropriate rolling buffer depends on application requirements "
        "(accuracy vs latency vs compute cost)."
    )
    tradeoff_block = "\n".join(tradeoff_lines)

    # ── Console table ──
    # Build header based on whether WER is available
    has_wer_col = has_transcript
    col_width = 105 if has_wer_col else 95
    print(f"\n  {'=' * col_width}")
    print(f"  Rolling Buffer Benchmark Summary")
    print(f"  {'=' * col_width}")
    header = (f"  {'Buffer':>7s} {'Latency':>8s} {'P95':>8s} "
              f"{'RTF':>8s} {'Backlog':>8s} {'Miss':>7s} {'Redund':>7s} "
              f"{'CPU%':>7s} {'RSS':>7s}")
    if has_wer_col:
        header += f" {'WER':>7s}"
    header += f" {'Decodes':>8s}"
    print(header)
    print(f"  {'-' * len(header)}")
    for cr in config_results:
        l = cr["decode_latency_ms"]
        line = (f"  {cr['buffer_ms']:>5d}ms {l['mean']:>8.1f} {l['p95']:>8.1f} "
                f"{cr['rtf']['mean']:>8.3f} {cr['backlog_ratio_p95']:>8.2f} "
                f"{cr['backlog_miss_rate_pct']:>6.1f}% {cr['redundancy_ratio']:>6.1f}x "
                f"{cr['cpu_percent']['mean']:>6.1f}% {cr['rss_mb']['mean']:>6.1f}")
        if has_wer_col:
            wer_cell = "N/A" if cr.get("wer") is None else f"{cr['wer']:.3f}"
            line += f" {wer_cell:>7s}"
        line += f" {cr['total_windows']:>8d}"
        print(line)
    print()

    # ── Build report ──
    report = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "configuration": {
            "temperature": 0.0,
            "beam_size": 1,
            "condition_on_previous_text": False,
            "vad_filter": False,
            "language": "en",
            "streaming_interval_ms": interval_ms,
            "buffer_sizes_ms": buffer_sizes,
            "warmup_runs": WARMUP_RUNS,
        },
        "has_ground_truth": has_transcript,
        "results": config_results,
        "tradeoff": tradeoff_block,
        "methodology_notes": (
            "This benchmark evaluates rolling window STT inference. "
            "Each rolling window is an independent STT request. NO transcript merging, "
            "NO stabilization, NO hypothesis revision. "
            "Words are assigned to windows by midpoint: midpoint = (start + end) / 2. "
            "WER/CER are weighted averages of per-window values (weighted by "
            "reference word count). Silent windows (zero reference words) are excluded.\n\n"
            "Backlog ratio P95 = P95 decode latency / streaming interval. "
            "Backlog miss rate = % of decodes exceeding the streaming interval. "
            "If backlog_ratio_p95 > 1.0, the system cannot sustain realtime streaming "
            "for 5% of decodes. "
            "Redundancy ratio = buffer_size / interval (derived, not measured)."
        ),
        "limitations": [
            "NO transcript merging — windows are evaluated independently.",
            "NO transcript stabilization — no multi-hypothesis refinement.",
            "NO incremental decoding — no partial hypothesis updates.",
            "NO LLM integration — measures STT only.",
            "NO VAD simulation — audio is pre-trimmed; real streaming has VAD delay.",
            "Offline simulation only — does not measure network or UI latency.",
            "Single streaming interval tested; changing the interval may shift tradeoffs.",
            "CPU utilization is sampled (psutil before/after inference) and is approximate.",
            "RSS is observed resident memory during sampling, not peak memory.",
        ],
    }

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    jsonl_path = os.path.join(OUTPUT_DIR, "rolling_buffer_results.jsonl")
    with open(jsonl_path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved: {jsonl_path} ({len(all_rows)} rows)")

    report_path = os.path.join(OUTPUT_DIR, "rolling_buffer_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {report_path}")


# ── Main ─────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="FasterWhisper rolling buffer benchmark"
    )
    parser.add_argument("--wav", type=str, default=None,
                        help="WAV file for accuracy+latency benchmark")
    parser.add_argument("--transcript", type=str, default=None,
                        help="Word-level transcript JSON (required with --wav)")
    parser.add_argument("--record", action="store_true",
                        help="Record audio (latency-only, no WER)")
    parser.add_argument("--interval", type=int, default=STREAMING_INTERVAL_MS,
                        help=f"Fixed streaming interval in ms (default: {STREAMING_INTERVAL_MS})")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of measurement rounds (default: 1)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Recording duration in seconds")
    args = parser.parse_args()

    np.random.seed(42)

    transcript = None
    wav_bytes = None

    if args.wav:
        with open(args.wav, "rb") as f:
            wav_bytes = f.read()
        dur = _get_wav_duration(wav_bytes)
        print(f"  WAV: {args.wav} ({len(wav_bytes)} bytes, {dur:.1f}s)")

        if args.transcript:
            transcript = load_transcript(args.transcript)
            print(f"  Transcript: {args.transcript} ({len(transcript)} words)")
        else:
            print(f"  No transcript provided — WER/CER will be N/A")
    elif args.record:
        print(f"  Recording {args.duration:.1f}s clip from mic...")
        wav_bytes = record_audio(args.duration)
        dur = _get_wav_duration(wav_bytes)
        print(f"  Captured: {len(wav_bytes)} bytes, {dur:.1f}s")
        print(f"  No transcript — WER/CER will be N/A")
    else:
        print(f"  Generating {args.duration:.1f}s synthetic audio...")
        wav_bytes = generate_synthetic_audio(args.duration)
        dur = _get_wav_duration(wav_bytes)
        print(f"  Generated: {len(wav_bytes)} bytes, {dur:.1f}s")
        print(f"  Synthetic audio — WER/CER will be N/A")

    run_benchmark(wav_bytes, transcript or [], BUFFER_SIZES_MS, args.interval)


if __name__ == "__main__":
    main()
