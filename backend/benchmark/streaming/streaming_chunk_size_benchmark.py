"""streaming_chunk_size_benchmark.py — Measure STT latency vs streaming chunk size.

Research question: "What streaming chunk size gives the best latency/performance tradeoff?"

Methodology:
  Given one complete audio clip, simulate streaming by splitting into fixed-duration
  chunks. Each chunk is transcribed independently with NO rolling buffer, NO accumulated
  context, and NO transcript merging. This measures ONLY the raw STT cost per chunk.

  This is NOT a full streaming benchmark. It does NOT measure:
    - rolling buffer overhead
    - transcript stability
    - LLM integration
    - suggestion latency
  Those are separate benchmarks.

Chunk sizes tested:
    250, 500, 750, 1000, 1500, 2000, 3000, 4000 milliseconds
  (configurable via CHUNK_SIZES constant)

Warmup runs: 2 before the first measurement round.

Usage:
    cd backend
    python benchmark/streaming/streaming_chunk_size_benchmark.py --record
    python benchmark/streaming/streaming_chunk_size_benchmark.py --wav path.wav
    python benchmark/streaming/streaming_chunk_size_benchmark.py --runs 3

Output:
    benchmark_results/streaming_chunk_size_results.jsonl
    benchmark_results/streaming_chunk_size_report.json
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
SLEEP_BETWEEN_CHUNKS = 0.1    # between chunks within same configuration
SLEEP_BETWEEN_CONFIGS = 1.0   # between different chunk sizes (CPU cooldown)
WARMUP_RUNS = 2                # full warmup rounds before measurement

CHUNK_SIZES_MS = [250, 500, 750, 1000, 1500, 2000, 3000, 4000]
MODEL_NAME = "tiny.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
SAMPLE_RATE = 16000


# ── Helpers ───────────────────────────────────────────────────

def _get_wav_duration(wav_bytes: bytes) -> float:
    with wave.open(BytesIO(wav_bytes), "rb") as wf:
        n = wf.getnframes()
        r = wf.getframerate()
    return n / r if r > 0 else 0


def slice_wav_chunks(wav_bytes: bytes, chunk_ms: int) -> list:
    """Split WAV into independent chunks of exactly chunk_ms duration.

    Each chunk is a proper WAV with a valid header — exactly what
    the STT provider receives in production. No overlap, no gaps.
    Returns list of (chunk_index, wav_bytes) tuples.
    """
    with wave.open(BytesIO(wav_bytes), "rb") as wf:
        n_total = wf.getnframes()
        rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        nchannels = wf.getnchannels()

    chunk_frames = int(chunk_ms / 1000.0 * rate)
    n_chunks = math.ceil(n_total / chunk_frames)

    chunks = []
    for i in range(n_chunks):
        start = i * chunk_frames
        end = min(start + chunk_frames, n_total)
        n_frames = end - start
        if n_frames <= 0:
            break

        with wave.open(BytesIO(wav_bytes), "rb") as wf:
            wf.setpos(start)
            frames = wf.readframes(n_frames)

        buf = BytesIO()
        with wave.open(buf, "wb") as wf_out:
            wf_out.setnchannels(nchannels)
            wf_out.setsampwidth(sampwidth)
            wf_out.setframerate(rate)
            wf_out.writeframes(frames)
        chunks.append((i, buf.getvalue()))

    return chunks


def try_psutil():
    """Attempt to import psutil. Return None if unavailable."""
    try:
        import psutil
        return psutil
    except ImportError:
        return None


def get_cpu_rss(psutil_mod) -> tuple:
    """Return (cpu_percent, rss_mb) for current process."""
    if psutil_mod is None:
        return (0.0, 0.0)
    proc = psutil_mod.Process()
    cpu = proc.cpu_percent(interval=0)
    rss = proc.memory_info().rss / (1024 * 1024)
    return (cpu, rss)


# ── Benchmark logic ───────────────────────────────────────────

def benchmark_chunk_size(
    model: WhisperModel,
    audio_float32: np.ndarray,
    chunk_ms: int,
    runs: int,
    psutil_mod,
) -> list:
    """Simulate streaming by transcribing independent chunks.

    Each run transcribes every chunk of the audio once.
    Returns list of per-chunk result dicts.
    """
    # Pre-compute chunks from raw audio
    wav_bytes = _audio_float32_to_wav(audio_float32)
    chunks = slice_wav_chunks(wav_bytes, chunk_ms)
    chunk_duration_s = chunk_ms / 1000.0

    all_results = []

    # Warmup: 2 full passes
    for _ in range(WARMUP_RUNS):
        for chunk_idx, chunk_wav in chunks:
            chunk_f32 = _wav_bytes_to_float32(chunk_wav)
            gen, _ = model.transcribe(
                chunk_f32, language="en", temperature=0.0, beam_size=1,
                condition_on_previous_text=False, vad_filter=False,
            )
            _ = list(gen)

    # Measured runs
    for run_idx in range(runs):
        for chunk_idx, chunk_wav in chunks:
            chunk_f32 = _wav_bytes_to_float32(chunk_wav)
            cpu_before, rss_before = get_cpu_rss(psutil_mod)

            t0 = time.perf_counter()
            gen, _ = model.transcribe(
                chunk_f32, language="en", temperature=0.0, beam_size=1,
                condition_on_previous_text=False, vad_filter=False,
            )
            _ = list(gen)
            t1 = time.perf_counter()

            cpu_after, rss_after = get_cpu_rss(psutil_mod)

            latency_ms = (t1 - t0) * 1000
            chunk_duration_ms = chunk_duration_s * 1000
            first_result_delay_ms = chunk_duration_ms + latency_ms
            rtf = latency_ms / chunk_duration_ms if chunk_duration_ms > 0 else 0
            cpu_avg = (cpu_before + cpu_after) / 2
            rss_observed = max(rss_before, rss_after)

            all_results.append({
                "chunk_ms": chunk_ms,
                "chunk_index": chunk_idx,
                "chunk_duration": chunk_duration_s,
                "run": run_idx,
                "latency_ms": round(latency_ms, 2),
                "first_result_delay_ms": round(first_result_delay_ms, 2),
                "rtf": round(rtf, 4),
                "cpu_percent": round(cpu_avg, 1),
                "rss_observed_mb": round(rss_observed, 1),
            })

            time.sleep(SLEEP_BETWEEN_CHUNKS)

    return all_results


def _audio_float32_to_wav(audio_f32: np.ndarray) -> bytes:
    """Convert float32 audio to WAV bytes (mono, 16-bit, 16kHz)."""
    int16 = (audio_f32 * 32767).clip(-32768, 32767).astype(np.int16)
    from services.audio import _to_wav
    return _to_wav(int16.reshape(-1, 1))


# ── Run ────────────────────────────────────────────────────────

def run_benchmark(audio_float32: np.ndarray, runs: int, chunk_sizes: list):
    """Run the streaming chunk size benchmark."""
    psutil_mod = try_psutil()
    has_psutil = psutil_mod is not None

    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

    duration_s = len(audio_float32) / SAMPLE_RATE

    print(f"  Model:       {MODEL_NAME}")
    print(f"  Device:      {DEVICE}")
    print(f"  Compute:     {COMPUTE_TYPE}")
    print(f"  Audio:       {len(audio_float32)} samples, {duration_s:.1f}s")
    print(f"  Chunk sizes: {chunk_sizes} ms")
    print(f"  Runs/config: {runs} (+{WARMUP_RUNS} warmup)")
    print(f"  psutil:      {'available' if has_psutil else 'NOT available (CPU/RSS=0)'}")
    print()

    all_rows = []
    config_summaries = []

    for cs in chunk_sizes:
        print(f"  ── Chunk={cs}ms ──")
        rows = benchmark_chunk_size(model, audio_float32, cs, runs, psutil_mod)
        all_rows.extend(rows)

        lats = sorted([r["latency_ms"] for r in rows])
        n = len(lats)
        mn = mean(lats)
        md = median(lats)
        p95 = lats[min(int(n * 0.95), n - 1)]
        lo = lats[0]
        hi = lats[-1]
        sd = stdev(lats) if n > 1 else 0
        cv = sd / mn if mn > 0 else 0

        delays = sorted([r["first_result_delay_ms"] for r in rows])
        d_mn = mean(delays)
        d_md = median(delays)
        d_p95 = delays[min(int(len(delays) * 0.95), len(delays) - 1)]

        rtfs = [r["rtf"] for r in rows]
        mn_rtf = mean(rtfs)

        cpus = [r["cpu_percent"] for r in rows]
        mn_cpu = mean(cpus) if cpus else 0
        pk_cpu = max(cpus) if cpus else 0

        rsses = [r["rss_observed_mb"] for r in rows]
        mn_rss = mean(rsses) if rsses else 0
        mx_rss = max(rsses) if rsses else 0

        cs_sec = cs / 1000.0
        total_chunks = len([r for r in rows if r["run"] == 0])  # chunks per run
        decodes_per_min = 60.0 / cs_sec if cs_sec > 0 else 0

        config_summaries.append({
            "chunk_ms": cs,
            "total_chunks_per_run": total_chunks,
            "total_decodes_per_run": total_chunks,
            "decodes_per_minute": round(decodes_per_min, 1),
            "latency_ms": {
                "mean": round(mn, 1),
                "median": round(md, 1),
                "p95": round(p95, 1),
                "min": round(lo, 1),
                "max": round(hi, 1),
                "std": round(sd, 1),
                "cv": round(cv, 3),
            },
            "first_result_delay_ms": {
                "mean": round(d_mn, 1),
                "median": round(d_md, 1),
                "p95": round(d_p95, 1),
            },
            "rtf": {"mean": round(mn_rtf, 3)},
            "cpu": {"mean": round(mn_cpu, 1), "observed_max": round(pk_cpu, 1)},
            "rss_observed_mb": {"mean": round(mn_rss, 1), "observed_max": round(mx_rss, 1)},
        })

        print(f"    n={total_chunks}  delay={d_mn:.0f}ms  mean={mn:.0f}ms  "
              f"p95={p95:.0f}ms  RTF={mn_rtf:.3f}  "
              f"approx_cpu={mn_cpu:.0f}%")
        time.sleep(SLEEP_BETWEEN_CONFIGS)

    # ── Tradeoff summary (NO winner) ──
    # The benchmark provides data; the application selects the tradeoff.
    # Smaller chunks → lower first_result_delay but more decodes and worse RTF.
    # Larger chunks → better efficiency but higher delay.
    tradeoff_block = (
        "No single 'optimal' chunk size. Tradeoffs by metric:\n"
        + "\n".join([
            f"  Lowest first_result_delay:  {min(config_summaries, key=lambda c: c['first_result_delay_ms']['mean'])['chunk_ms']}ms",
            f"  Lowest decode latency:      {min(config_summaries, key=lambda c: c['latency_ms']['mean'])['chunk_ms']}ms",
            f"  Best RTF (efficiency):      {min(config_summaries, key=lambda c: c['rtf']['mean'])['chunk_ms']}ms",
            f"  Fewest decodes/min:         {max(config_summaries, key=lambda c: c['chunk_ms'])['chunk_ms']}ms",
        ]) + "\n" + (
            "Decision depends on application requirements (latency vs throughput)."
        ))

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
            "chunk_sizes_ms": chunk_sizes,
            "runs_per_config": runs,
            "warmup_runs": WARMUP_RUNS,
        },
        "chunk_results": config_summaries,
        "tradeoff": tradeoff_block,
        "methodology_notes": (
            "This benchmark measures only independent chunk inference cost. "
            "It does NOT simulate rolling buffers, incremental decoding, "
            "context accumulation, partial hypothesis revision, or "
            "transcript stabilization. Those are evaluated in separate benchmarks.\n\n"
            "RTF (Real-Time Factor) measures computational efficiency. "
            "During fixed-interval streaming, RTF can also be interpreted as "
            "the fraction of each streaming interval consumed by inference. "
            "Therefore a separate Occupancy metric is unnecessary."
        ),
        "limitations": [
            "NO rolling buffer — chunks are independent, not accumulated.",
            "NO transcript stabilization — raw per-chunk outputs, no merging logic.",
            "NO incremental decoding — each chunk is a fresh STT request.",
            "NO partial hypothesis revision — no mid-utterance updates.",
            "NO LLM integration — measures STT only, not full pipeline.",
            "NO VAD simulation — audio is pre-trimmed; real streaming has VAD delay.",
            "CPU utilization is sampled using psutil (before/after inference) "
            "and should be interpreted as approximate, not continuous profiling.",
            "rss_observed_mb is the observed resident memory during sampling, "
            "NOT peak memory usage. True peak would require continuous monitoring.",
            "Single audio clip — results may vary with different speech content.",
        ],
    }

    # ── Save ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    jsonl_path = os.path.join(OUTPUT_DIR, "streaming_chunk_size_results.jsonl")
    with open(jsonl_path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n  Saved: {jsonl_path} ({len(all_rows)} rows)")

    report_path = os.path.join(OUTPUT_DIR, "streaming_chunk_size_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {report_path}")

    # ── Console table ──
    print(f"\n  {'=' * 95}")
    print(f"  Streaming Chunk Size Benchmark Summary")
    print(f"  {'=' * 95}")
    header = (f"  {'Chunk':>7s} {'Mean':>8s} {'Delay':>8s} {'P95':>8s} "
              f"{'RTF':>8s} {'CPU%':>7s} {'RSS':>7s} {'Dec/min':>8s}")
    print(header)
    print(f"  {'-' * len(header)}")
    for cs in config_summaries:
        l = cs["latency_ms"]
        d = cs["first_result_delay_ms"]
        print(f"  {cs['chunk_ms']:>5d}ms {l['mean']:>8.1f} {d['mean']:>8.1f} "
              f"{l['p95']:>8.1f} {cs['rtf']['mean']:>8.3f} "
              f"{cs['cpu']['mean']:>6.1f}% "
              f"{cs['rss_observed_mb']['mean']:>6.1f} "
              f"{cs['decodes_per_minute']:>8.1f}")
    print(f"  Tradeoff summary (no CPU recommendation — CPU values are approximate):")
    print(f"    Lowest delay:       {min(config_summaries, key=lambda c: c['first_result_delay_ms']['mean'])['chunk_ms']}ms")
    print(f"    Best RTF:           {min(config_summaries, key=lambda c: c['rtf']['mean'])['chunk_ms']}ms")
    print(f"    Decodes/min range:  {config_summaries[0]['decodes_per_minute']:.0f} — {config_summaries[-1]['decodes_per_minute']:.0f}")
    print()


# ── Main ──────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="FasterWhisper streaming chunk size benchmark"
    )
    parser.add_argument("--wav", type=str, default=None,
                        help="WAV file to slice into chunks")
    parser.add_argument("--record", action="store_true",
                        help="Record one long clip, then slice into chunks")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of measurement rounds per chunk size")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Recording duration (s) for --record")
    args = parser.parse_args()

    np.random.seed(42)

    if args.wav:
        with open(args.wav, "rb") as f:
            wav_bytes = f.read()
        audio_f32 = _wav_bytes_to_float32(wav_bytes)
        dur = _get_wav_duration(wav_bytes)
        print(f"  Source WAV: {args.wav} ({len(wav_bytes)} bytes, {dur:.1f}s)")
    elif args.record:
        from services.audio import record_chunk
        print(f"  Recording {args.duration:.1f}s clip from mic...")
        wav_bytes = record_chunk(args.duration)
        audio_f32 = _wav_bytes_to_float32(wav_bytes)
        dur = _get_wav_duration(wav_bytes)
        print(f"  Captured: {len(wav_bytes)} bytes, {dur:.1f}s")
    else:
        print(f"  Generating {args.duration:.1f}s synthetic audio...")
        n = int(args.duration * SAMPLE_RATE)
        t = np.linspace(0, args.duration, n, endpoint=False)
        noise = np.random.randn(n) * 0.3
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
        samples = (noise * mod * 32767).astype(np.int16)
        from services.audio import _to_wav
        wav_bytes = _to_wav(samples.reshape(-1, 1))
        audio_f32 = _wav_bytes_to_float32(wav_bytes)

    run_benchmark(audio_f32, args.runs, CHUNK_SIZES_MS)


if __name__ == "__main__":
    main()
