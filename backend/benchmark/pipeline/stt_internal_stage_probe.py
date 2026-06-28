"""stt_internal_stage_probe.py — Profile internal FasterWhisper stages.

Research question: "Which internal stage of FasterWhisper consumes the time?"

Instruments the following stages via monkey-patching on a live model instance:
  1. wav_to_float_ms     — WAV decoding + int16→float32 normalization
  2. log_mel_ms          — log-Mel spectrogram extraction (feature_extractor)
  3. encoder_ms           — self.encode() → self.model.encode(): the Whisper encoder
  4. decoder_ms           — self.model.generate(): decoder forward + beam search (combined by ct2)
  5. segment_build_ms     — self._split_segments_by_timestamps(): raw output → Segment objects
  6. transcript_join_ms   — " ".join(...) transcript assembly

Note: ct2 does not expose encoder vs decoder step as separate public calls.
      The best we can do is time self.encode() (encoder only) and
      self.model.generate() (decoder + beam search combined).
      Beam search itself is inside the ct2 binary and not measurable separately
      without modifying CTranslate2 source.

This is a RESEARCH benchmark. It monkey-patches the model instance.
No production code is modified.

Usage:
    cd backend
    python benchmark/pipeline/stt_internal_stage_probe.py --record
    python benchmark/pipeline/stt_internal_stage_probe.py --wav path.wav
    python benchmark/pipeline/stt_internal_stage_probe.py --runs 10

Output:
    benchmark_results/stt_internal_stage_probe.jsonl
    benchmark_results/stt_internal_stage_probe_report.json
"""

import json
import os
import sys
import time
import types
from statistics import mean, median, stdev
from io import BytesIO
import wave
import collections

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from services.stt_faster_whisper import _wav_bytes_to_float32

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmark_results"
)
SLEEP_BETWEEN = 0.3
WARMUP_RUNS = 2
MEASURED_RUNS = 10

MODEL_NAME = "tiny.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"


# ── Timing collector ──────────────────────────────────────────
# We store per-call timings in a thread-safe list attached to the model.
# Each entry: {"stage": str, "ms": float, "call": int}


class StageTimingCollector:
    """Collects per-stage timing data from monkey-patched methods."""

    def __init__(self):
        self.records = []
        self._call_counter = 0

    def next_call(self) -> int:
        self._call_counter += 1
        return self._call_counter

    def record(self, stage: str, ms: float, call: int):
        self.records.append({"stage": stage, "ms": ms, "call": call})

    def reset(self):
        self.records.clear()
        self._call_counter = 0

    def group_by_stage(self) -> dict:
        """Aggregate per-stage timing.

        FasterWhisper's generate_segments() loops over segment windows.
        encode() is called once per window. generate() is called per window.
        We sum all calls to the same stage to get total time per stage.
        """
        totals = collections.defaultdict(float)
        counts = collections.defaultdict(int)
        for rec in self.records:
            totals[rec["stage"]] += rec["ms"]
            counts[rec["stage"]] += rec["call"]
        return {
            stage: {
                "total_ms": round(ms, 2),
                "calls": counts[stage],
            }
            for stage, ms in totals.items()
        }


def instrument_model(model, collector: StageTimingCollector):
    """Monkey-patch encoder, decoder, and feature extraction on a live model.

    This is the core of the benchmark. We wrap:
      - model.encode()           → encoder stage
      - model.model.generate()   → decoder + beam search stage (ct2 internal)

    The feature extractor is called inside generate_segments() before encode().
    We wrap that too.

    Note: We do NOT patch the entire generate_segments() because that would
    also include the loop overhead and segment filtering logic, which we
    want to exclude from encoder/decoder timing.
    """
    collector.reset()

    # ── 1. Monkey-patch model.encode() ──
    # This is the Whisper encoder: Conv1 + transformer encoder blocks.
    original_encode = model.encode

    def timed_encode(self_obj, features):
        call_n = collector.next_call()
        t0 = time.perf_counter()
        result = original_encode(features)
        t1 = time.perf_counter()
        collector.record("encoder_ms", (t1 - t0) * 1000, call_n)
        return result

    model.encode = types.MethodType(timed_encode, model)

    # ── 2. Monkey-patch model.generate_with_fallback() ──
    # This wraps the entire decoder invocation: self.model.generate() + scoring + result formatting.
    # ct2 does NOT expose decoder steps vs beam search separately — both are inside the
    # self.model.generate() C++ call which cannot be monkey-patched (ct2 extension object).
    # generate_with_fallback() is the closest Python wrapper we can instrument.
    original_gwf = model.generate_with_fallback

    def timed_generate_with_fallback(self_obj, encoder_output, prompt, tokenizer, options):
        call_n = collector.next_call()
        t0 = time.perf_counter()
        result = original_gwf(encoder_output, prompt, tokenizer, options)
        t1 = time.perf_counter()
        collector.record("decoder_ms", (t1 - t0) * 1000, call_n)
        return result

    model.generate_with_fallback = types.MethodType(
        timed_generate_with_fallback, model
    )

    # ── 3. Wrap feature_extractor with a proxy class ──
    # self.feature_extractor is called once per segment window.
    # It converts float32 audio → log-Mel spectrogram.
    #
    # NOTE: Patching __call__ on an instance does NOT work in Python — Python
    # resolves dunder methods on the *type*, not the instance, so instance-level
    # assignment is silently ignored. We replace the attribute with a proxy object
    # whose class defines __call__, which Python's lookup will find correctly.
    if hasattr(model, 'feature_extractor'):
        _orig_fe = model.feature_extractor
        _fe_collector = collector  # close over collector

        class _TimedFeatureExtractor:
            """Proxy for feature_extractor that records timing."""
            def __init__(self, wrapped):
                self._wrapped = wrapped

            def __call__(self, *args, **kwargs):
                call_n = _fe_collector.next_call()
                t0 = time.perf_counter()
                result = self._wrapped(*args, **kwargs)
                t1 = time.perf_counter()
                _fe_collector.record("log_mel_ms", (t1 - t0) * 1000, call_n)
                return result

            def __getattr__(self, name):
                # Forward all other attribute access to the real extractor
                return getattr(self._wrapped, name)

        model.feature_extractor = _TimedFeatureExtractor(_orig_fe)


def run_one(model, collector, audio_float32: np.ndarray) -> dict:
    """Run transcription with instrumentation, return per-call timing + result."""
    collector.reset()

    t_start = time.perf_counter()
    segments_gen, info = model.transcribe(
        audio_float32,
        language="en",
        beam_size=1,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    t_after_call = time.perf_counter()

    # Consume the generator (this is where encoder/decoder work happens for
    # lazy evaluation, plus segment assembly)
    texts = list(segments_gen)  # list of Segment objects, not strings
    t_after_segments = time.perf_counter()

    # Join transcript — extract .text from each Segment
    transcript = " ".join(seg.text for seg in texts).strip()
    t_done = time.perf_counter()

    # Build timing breakdown
    stage_groups = collector.group_by_stage()

    # wav_to_float is NOT part of model.transcribe — we measure it separately
    # gen call time includes encoder + decoder stages (already captured in stage_groups)
    # segment iteration time = model.generate() return → segments consumed
    segment_iter_ms = (t_after_segments - t_after_call) * 1000

    # Total wall time (from outside) — should match sum of sub-stages
    wall_total_ms = (t_done - t_start) * 1000

    result = {
        "encoder_ms": stage_groups.get("encoder_ms", {}).get("total_ms", 0),
        "decoder_ms": stage_groups.get("decoder_ms", {}).get("total_ms", 0),
        "log_mel_ms": stage_groups.get("log_mel_ms", {}).get("total_ms", 0),
        "segment_iter_ms": round(segment_iter_ms, 2),
        "encoder_calls": stage_groups.get("encoder_ms", {}).get("calls", 0),
        "decoder_calls": stage_groups.get("decoder_ms", {}).get("calls", 0),
        "total_ms": round(wall_total_ms, 2),
        "segments_found": len(texts),
        "transcript": transcript[:80],
    }

    return result


def print_stage_breakdown(all_results: list):
    """Print per-stage timing with percentage contribution."""
    stages = ["log_mel_ms", "encoder_ms", "decoder_ms", "segment_iter_ms"]
    stage_labels = {
        "log_mel_ms": "Log-Mel spectrogram",
        "encoder_ms": "Encoder",
        "decoder_ms": "Decoder + beam search",
        "segment_iter_ms": "Segment assembly",
    }

    # Aggregate per-stage across all runs
    print(f"\n  {'=' * 65}")
    print(f"  Stage breakdown by percentage")
    print(f"  {'=' * 65}")
    print(f"  {'Stage':25s} {'Mean':>9s} {'Median':>9s} {'P95':>9s} {'% of total':>10s}")
    print(f"  {'-' * 65}")

    totals_per_run = [r["total_ms"] for r in all_results]
    for stage in stages:
        vals = sorted([r[stage] for r in all_results])
        n = len(vals)
        mn = mean(vals)
        md = median(vals)
        p95 = vals[int(n * 0.95)]
        # Percentage: mean(stage) / mean(total) per run
        pcts = [
            r[stage] / r["total_ms"] * 100
            for r in all_results if r["total_ms"] > 0
        ]
        avg_pct = mean(pcts) if pcts else 0
        label = stage_labels.get(stage, stage)
        print(f"  {label:25s} {mn:>9.2f} {md:>9.2f} {p95:>9.2f} {avg_pct:>9.1f}%")

    # Total at bottom
    mnt = mean(totals_per_run)
    print(f"  {'-' * 65}")
    print(f"  {'Total':25s} {mnt:>9.2f}")
    print()

    # Dominant stage
    stage_means = {stage_labels[s]: mean([r[s] for r in all_results]) for s in stages}
    dominant = max(stage_means, key=stage_means.get)
    dom_val = stage_means[dominant]
    dom_pct = dom_val / mnt * 100 if mnt > 0 else 0
    print(f"  Dominant stage: {dominant} ({dom_val:.0f}ms, {dom_pct:.1f}%)")

    # If encoder + decoder explain >90% of time
    enc_dec = sum(stage_means[s] for s in ["Encoder", "Decoder + beam search"]
                   if s in stage_means) if False else (
        mean([r["encoder_ms"] for r in all_results]) +
        mean([r["decoder_ms"] for r in all_results])
    )
    enc_dec_pct = enc_dec / mnt * 100 if mnt > 0 else 0
    print(f"  Encoder + decoder together: {enc_dec_pct:.1f}% of total")
    print()


def run_benchmark(audio_float32: np.ndarray, runs: int):
    """Run probe N times, save results."""
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

    collector = StageTimingCollector()
    instrument_model(model, collector)

    print(f"  Model:     {MODEL_NAME}")
    print(f"  Device:    {DEVICE}")
    print(f"  Compute:   {COMPUTE_TYPE}")
    print(f"  Measured:  {runs} runs (+ {WARMUP_RUNS} warmup)")
    print(f"  Audio:     {len(audio_float32)} samples, "
          f"~{len(audio_float32) / 16000:.1f}s")
    print()

    # Warmup
    for _ in range(WARMUP_RUNS):
        gen, _ = model.transcribe(
            audio_float32, language="en", beam_size=1,
            condition_on_previous_text=False, vad_filter=False,
        )
        _ = list(gen)

    # Measured runs
    all_results = []
    for i in range(runs):
        # wav_to_float is already done before this function — it's <1ms
        result = run_one(model, collector, audio_float32)
        all_results.append(result)
        print(f"  [{i+1}/{runs}] enc={result['encoder_ms']:.0f}ms  "
              f"dec={result['decoder_ms']:.0f}ms  "
              f"mel={result['log_mel_ms']:.0f}ms  "
              f"seg_iter={result['segment_iter_ms']:.0f}ms  "
              f"total={result['total_ms']:.0f}ms  "
              f"segments={result['segments_found']}")
        time.sleep(SLEEP_BETWEEN)

    # ── Print breakdown ──
    print_stage_breakdown(all_results)

    # ── Build report ──
    stage_fields = ["log_mel_ms", "encoder_ms", "decoder_ms", "segment_iter_ms"]
    per_stage_stats = {}
    for stage in stage_fields:
        vals = sorted([r[stage] for r in all_results])
        n = len(vals)
        mn = mean(vals)
        md = median(vals)
        p95 = vals[int(n * 0.95)]
        lo = vals[0]
        hi = vals[-1]
        sd = stdev(vals) if n > 1 else 0
        cv = sd / mn if mn > 0 else 0
        per_stage_stats[stage] = {
            "mean": round(mn, 2),
            "median": round(md, 2),
            "p95": round(p95, 2),
            "min": round(lo, 2),
            "max": round(hi, 2),
            "std": round(sd, 2),
            "cv": round(cv, 3),
        }

    totals = sorted([r["total_ms"] for r in all_results])
    per_stage_stats["total_ms"] = {
        "mean": round(mean(totals), 2),
        "median": round(median(totals), 2),
        "p95": round(totals[int(len(totals) * 0.95)], 2),
        "min": round(min(totals), 2),
        "max": round(max(totals), 2),
        "std": round(stdev(totals), 2) if len(totals) > 1 else 0,
    }

    # Dominant stage identification
    stage_means = {s: per_stage_stats[s]["mean"] for s in stage_fields}
    dominant_stage = max(stage_means, key=stage_means.get)
    dom_val = stage_means[dominant_stage]
    total_mean = per_stage_stats["total_ms"]["mean"]
    dom_pct = round(dom_val / total_mean * 100, 1) if total_mean > 0 else 0

    report = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "runs": runs,
        "warmup_runs": WARMUP_RUNS,
        "dominant_stage": {
            "stage": dominant_stage,
            "mean_ms": round(dom_val, 1),
            "pct_of_total": dom_pct,
        },
        "per_stage_stats": per_stage_stats,
        "note": (
            "encoder_ms = self.encode() — Whisper encoder (Conv1 + transformer). "
            "decoder_ms = self.model.generate() — CTranslate2 decoder + beam search (combined, "
            "cannot be separated without modifying ct2 source). "
            "log_mel_ms = feature_extractor — float32 → log-Mel spectrogram. "
            "segment_iter_ms = time to consume generator — segment assembly + python loop overhead."
        ),
    }

    # Save JSONL
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    jsonl_path = os.path.join(OUTPUT_DIR, "stt_internal_stage_probe.jsonl")
    with open(jsonl_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\n  Saved: {jsonl_path}")

    report_path = os.path.join(OUTPUT_DIR, "stt_internal_stage_probe_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")


def record_audio(duration: float = 5.0) -> bytes:
    from services.audio import record_chunk
    return record_chunk(duration)


def load_wav(path: str) -> bytes:
    if not os.path.isfile(path):
        abs_path = os.path.abspath(path)
        raise FileNotFoundError(
            f"WAV file not found: {abs_path}\n"
            f"  - Make sure the path is correct (run from backend/ dir)\n"
            f"  - Or use --record / no args (synthetic audio) instead"
        )
    with open(path, "rb") as f:
        return f.read()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="FasterWhisper internal stage probe"
    )
    parser.add_argument("--wav", type=str, default=None, help="WAV file")
    parser.add_argument("--record", action="store_true", help="Record from mic")
    parser.add_argument("--runs", type=int, default=MEASURED_RUNS, help="Measurements")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration")
    args = parser.parse_args()

    np.random.seed(42)

    if args.wav:
        wav_bytes = load_wav(args.wav)
        audio_f32 = _wav_bytes_to_float32(wav_bytes)
        print(f"  WAV: {args.wav} ({len(wav_bytes)} bytes)")
    elif args.record:
        wav_bytes = record_audio(args.duration)
        audio_f32 = _wav_bytes_to_float32(wav_bytes)
        print(f"  Recorded: {len(wav_bytes)} bytes")
    else:
        # Synthetic audio
        print(f"  Generating {args.duration}s synthetic audio...")
        n = int(args.duration * 16000)
        t = np.linspace(0, args.duration, n, endpoint=False)
        noise = np.random.randn(n) * 0.3
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
        samples = (noise * mod * 32767).astype(np.int16)
        from services.audio import _to_wav
        wav_bytes = _to_wav(samples.reshape(-1, 1))
        audio_f32 = _wav_bytes_to_float32(wav_bytes)

    run_benchmark(audio_f32, args.runs)


if __name__ == "__main__":
    main()
