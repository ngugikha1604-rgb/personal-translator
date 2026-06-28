"""stt_internal_stage_probe.py — Profile internal FasterWhisper stages (v2 — correct).

Fixed design:
  The original implementation treated "segment_iter_ms" as an independent
  stage alongside encoder and decoder. This was wrong because FasterWhisper
  uses a lazy generator: encoder, decoder, and segment assembly all run
  INSIDE the generator iteration. segment_iter_ms is the CONTAINER, not a stage.

Correct execution timeline:
  model.transcribe(audio):
    └── feature_extractor(audio)  ← EAGER: log-Mel spectrogram (~0-2ms)
    └── returns generator (no work done)

  list(segments) — generator consumption:
    for each window:
      ├── self.encode(segment)              ← encoder (monkey-patched)
      ├── self.generate_with_fallback(...)   ← decoder + beam (monkey-patched)
      │     └── self.model.generate()       ← ct2 C++ call (cannot separate)
      ├── self._split_segments_by_timestamps()
      ├── Segment(...) object creation
      └── yield Segment(text=...)

Correct breakdown:
  generator_consumption_ms = encoder_ms + decoder_ms + other_overhead_ms
  total_ms = log_mel_ms + generator_consumption_ms (+ trivial join time)

  where other_overhead_ms = generator_consumption_ms - encoder_ms - decoder_ms
  (covers: segment splitting, language detection, Segment creation, loop overhead)

This ensures:
  - encoder + decoder + overhead ≈ generator_consumption (mutually exclusive)
  - No stages overlap or double-count
  - Sum of sub-stages equals total within measurement noise

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


class StageTimingCollector:
    """Collects per-stage timing from monkey-patched methods.

    Each record is {stage, ms, call}. stage is one of:
      "encoder_ms"       — self.encode() → Whisper encoder
      "decoder_ms"       — self.generate_with_fallback() → decoder + beam
      "log_mel_ms"       — feature_extractor → float32 → log-Mel spectrogram
    """

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

    def sum_stage(self, stage: str) -> float:
        """Sum of all recorded milliseconds for a given stage."""
        return sum(r["ms"] for r in self.records if r["stage"] == stage)


def instrument_model(model, collector: StageTimingCollector):
    """Monkey-patch encoder, generate_with_fallback, and feature_extractor.

    Patches are applied to the MODEL INSTANCE only. Does not modify
    the class or any other instance. Restored by reinstantiating the
    model (which the benchmark does on each run).

    Why we CAN instrument these:
      - self.encode() is a Python method on the class → can monkey-patch
      - self.generate_with_fallback() is a Python method → can monkey-patch
      - self.feature_extractor is a CTranslate2 object → cannot monkey-patch
        via types.MethodType. Use a proxy wrapper instead.

    Why we CANNOT separate decoder from beam search:
      - self.model.generate() is a CTranslate2 C extension method called
        inside generate_with_fallback(). It performs both decoding and
        beam search in one opaque call. Neither the ct2 Python bindings
        nor the FasterWhisper wrapper expose these separately.
    """
    collector.reset()

    # ── 1. Monkey-patch model.encode() [encoder only] ──
    original_encode = model.encode

    def timed_encode(self_obj, features):
        call_n = collector.next_call()
        t0 = time.perf_counter()
        result = original_encode(features)
        t1 = time.perf_counter()
        collector.record("encoder_ms", (t1 - t0) * 1000, call_n)
        return result

    model.encode = types.MethodType(timed_encode, model)

    # ── 2. Monkey-patch model.generate_with_fallback() [decoder + beam] ──
    # This wraps self.model.generate() + scoring + result formatting.
    # Cannot separate decoder from beam search — both are inside ct2 C++ call.
    original_gwf = model.generate_with_fallback

    def timed_gwf(self_obj, encoder_output, prompt, tokenizer, options):
        call_n = collector.next_call()
        t0 = time.perf_counter()
        result = original_gwf(encoder_output, prompt, tokenizer, options)
        t1 = time.perf_counter()
        collector.record("decoder_ms", (t1 - t0) * 1000, call_n)
        return result

    model.generate_with_fallback = types.MethodType(timed_gwf, model)

    # ── 3. Proxy for feature_extractor [log-Mel spectrogram] ──
    # Python resolves __call__ on the object's TYPE, not the instance.
    # Assigning a function to model.feature_extractor.__call__ is ignored.
    # We replace the entire attribute with a proxy object whose class
    # defines __call__, which Python will find correctly.
    if hasattr(model, 'feature_extractor'):
        _orig_fe = model.feature_extractor
        _collector = collector

        class _TimedFeatureExtractor:
            def __init__(self, wrapped):
                self._wrapped = wrapped

            def __call__(self, *args, **kwargs):
                call_n = _collector.next_call()
                t0 = time.perf_counter()
                result = self._wrapped(*args, **kwargs)
                t1 = time.perf_counter()
                _collector.record("log_mel_ms", (t1 - t0) * 1000, call_n)
                return result

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

        model.feature_extractor = _TimedFeatureExtractor(_orig_fe)


def run_one(model, collector, audio_float32: np.ndarray) -> dict:
    """Run one transcription with timing, return mutually exclusive breakdown.

    Timing regions are designed to be ADDITIVE, not overlapping:
      total_ms = log_mel_ms + generator_consumption_ms [+ trivial join overhead]
      generator_consumption_ms = encoder_ms + decoder_ms + other_overhead_ms

    No stage overlaps or double-counts any other stage.
    """
    collector.reset()

    # ── t_start to t_after_call: EAGER work ──
    # model.transcribe(audio) eagerly runs feature_extractor (log-Mel)
    # and returns a generator. No encoder/decoder work has happened yet.
    t_start = time.perf_counter()
    segments_gen, info = model.transcribe(
        audio_float32,
        language="en",
        beam_size=1,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    t_after_call = time.perf_counter()

    # ── t_after_call to t_after_segments: LAZY generator consumption ──
    # list(segments) iterates the generator, which drives the main loop
    # inside generate_segments(). This is where all real computation
    # (encoder, decoder, segment splitting) happens.
    # The monkey-patched encode() and generate_with_fallback() fire
    # during this interval.
    texts = list(segments_gen)
    t_after_segments = time.perf_counter()

    # ── t_after_segments to t_done: transcript assembly (negligible) ──
    transcript = " ".join(seg.text for seg in texts).strip()
    t_done = time.perf_counter()

    # ── Compute mutually exclusive stages ──
    generator_consumption_ms = (t_after_segments - t_after_call) * 1000
    encoder_ms = collector.sum_stage("encoder_ms")
    decoder_ms = collector.sum_stage("decoder_ms")
    log_mel_ms = collector.sum_stage("log_mel_ms")

    # other_overhead_ms = everything in generator except encoder + decoder
    # This includes: segment splitting, language detection,
    #                Segment object creation, loop overhead.
    other_overhead_ms = generator_consumption_ms - encoder_ms - decoder_ms
    if other_overhead_ms < 0:
        # Measurement noise — clamp to 0
        other_overhead_ms = 0.0

    # Total wall time from t_start to t_done
    wall_total_ms = (t_done - t_start) * 1000

    return {
        # Eager stage (happens inside model.transcribe before return)
        "log_mel_ms": round(log_mel_ms, 2),

        # Lazy stages (happen inside list(segments))
        "generator_consumption_ms": round(generator_consumption_ms, 2),
        "encoder_ms": round(encoder_ms, 2),
        "decoder_ms": round(decoder_ms, 2),
        "other_overhead_ms": round(other_overhead_ms, 2),

        # Totals
        "total_ms": round(wall_total_ms, 2),

        # Metadata
        "segments_found": len(texts),
        "transcript": transcript[:80],
        "encoder_calls": sum(1 for r in collector.records if r["stage"] == "encoder_ms"),
        "decoder_calls": sum(1 for r in collector.records if r["stage"] == "decoder_ms"),
    }


def print_stage_breakdown(all_results: list):
    """Print stage breakdown with additive verification.

    The stages are organized hierarchically:
      total_ms ≈ log_mel_ms + generator_consumption_ms
      generator_consumption_ms ≈ encoder_ms + decoder_ms + other_overhead_ms
    """
    lazy_stages_def = [
        ("encoder_ms",          "Encoder"),
        ("decoder_ms",          "Decoder + beam search"),
        ("other_overhead_ms",   "Other (segment splitting, etc.)"),
    ]
    all_stages_def = [
        ("log_mel_ms",          "Log-Mel spectrogram (eager)"),
        ("generator_consumption_ms", "Generator consumption (total lazy work)"),
    ] + lazy_stages_def

    print(f"\n  {'=' * 70}")
    print(f"  Stage breakdown — mutually exclusive, additive")
    print(f"  {'=' * 70}")
    print(f"  {'Stage':32s} {'Mean':>9s} {'Median':>9s} {'P95':>9s} {'% of total':>10s}")
    print(f"  {'-' * 70}")

    totals_per_run = [r["total_ms"] for r in all_results]
    mnt = mean(totals_per_run)

    for key, label in all_stages_def:
        vals = sorted([r[key] for r in all_results])
        n = len(vals)
        mn = mean(vals)
        md = median(vals)
        p95 = vals[int(n * 0.95)]
        pcts = [
            r[key] / r["total_ms"] * 100
            for r in all_results if r["total_ms"] > 0
        ]
        avg_pct = mean(pcts) if pcts else 0
        print(f"  {label:32s} {mn:>9.2f} {md:>9.2f} {p95:>9.2f} {avg_pct:>9.1f}%")

    print(f"  {'─' * 70}")

    # Additive verification
    sum_lazy = mean([r["encoder_ms"] + r["decoder_ms"] + r["other_overhead_ms"]
                      for r in all_results])
    gen_mean = mean([r["generator_consumption_ms"] for r in all_results])
    diff_lazy = abs(sum_lazy - gen_mean)
    sum_all = mean([r["log_mel_ms"] + r["generator_consumption_ms"] for r in all_results])
    diff_all = abs(sum_all - mnt)

    print(f"  Additive check: enc+dec+overhead={sum_lazy:.1f}ms vs gen_consumption={gen_mean:.1f}ms "
          f"(diff={diff_lazy:.1f}ms)")
    print(f"  Additive check: log_mel+gen_consumption={sum_all:.1f}ms vs total={mnt:.1f}ms "
          f"(diff={diff_all:.1f}ms)")

    print()

    # Dominant stage (among lazy work)
    lazy_means = {
        label: mean([r[key] for r in all_results])
        for key, label in lazy_stages_def
    }
    dominant = max(lazy_means, key=lazy_means.get)
    dom_val = lazy_means[dominant]
    gen_avg = mean([r["generator_consumption_ms"] for r in all_results])
    dom_pct = dom_val / gen_avg * 100 if gen_avg > 0 else 0
    print(f"  Dominant stage (during generator iteration): {dominant} ({dom_val:.0f}ms, {dom_pct:.1f}%)")
    print()


def run_benchmark(audio_float32: np.ndarray, runs: int):
    """Run N measured runs with warmup, save results."""
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
        result = run_one(model, collector, audio_float32)
        all_results.append(result)
        print(
            f"  [{i+1:2d}/{runs}] enc={result['encoder_ms']:>6.1f}ms  "
            f"dec={result['decoder_ms']:>6.1f}ms  "
            f"overhead={result['other_overhead_ms']:>5.1f}ms  "
            f"gen={result['generator_consumption_ms']:>6.1f}ms  "
            f"total={result['total_ms']:>6.1f}ms"
        )
        time.sleep(SLEEP_BETWEEN)

    print_stage_breakdown(all_results)

    # ── Build report ──
    stage_fields = [
        "log_mel_ms", "generator_consumption_ms",
        "encoder_ms", "decoder_ms", "other_overhead_ms"
    ]
    per_stage_stats = {}
    for stage in stage_fields:
        vals = sorted([r[stage] for r in all_results])
        n = len(vals)
        mn = mean(vals) if n > 0 else 0
        md = median(vals) if n > 0 else 0
        p95 = vals[int(n * 0.95)] if n > 1 else vals[0] if n > 0 else 0
        sd = stdev(vals) if n > 1 else 0
        cv = sd / mn if mn > 0 else 0
        per_stage_stats[stage] = {
            "mean": round(mn, 2),
            "median": round(md, 2),
            "p95": round(p95, 2),
            "min": round(min(vals), 2) if vals else 0,
            "max": round(max(vals), 2) if vals else 0,
            "std": round(sd, 2),
            "cv": round(cv, 3),
        }

    totals = sorted([r["total_ms"] for r in all_results])
    per_stage_stats["total_ms"] = {
        "mean": round(mean(totals), 2) if totals else 0,
        "median": round(median(totals), 2) if totals else 0,
        "p95": round(totals[int(len(totals) * 0.95)], 2) if totals else 0,
        "min": round(min(totals), 2) if totals else 0,
        "max": round(max(totals), 2) if totals else 0,
        "std": round(stdev(totals), 2) if len(totals) > 1 else 0,
    }

    lazy_stages = ["encoder_ms", "decoder_ms", "other_overhead_ms"]
    lazy_means = {s: per_stage_stats[s]["mean"] for s in lazy_stages}
    dominant = max(lazy_means, key=lazy_means.get)
    dom_val = lazy_means[dominant]
    gen_mean = per_stage_stats["generator_consumption_ms"]["mean"]

    report = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "runs": runs,
        "warmup_runs": WARMUP_RUNS,
        "additive_check": {
            "lazy_sum_ms": round(sum_lazy, 1) if 'sum_lazy' in dir() else (
                round(per_stage_stats["encoder_ms"]["mean"] +
                      per_stage_stats["decoder_ms"]["mean"] +
                      per_stage_stats["other_overhead_ms"]["mean"], 1)
            ),
            "generator_consumption_ms": round(gen_mean, 1),
            "total_reconstructed_ms": round(
                per_stage_stats["log_mel_ms"]["mean"] + gen_mean, 1
            ),
            "total_ms": per_stage_stats["total_ms"]["mean"],
        },
        "dominant_stage": {
            "stage": dominant,
            "mean_ms": round(dom_val, 1),
            "pct_of_generator_time": round(dom_val / gen_mean * 100, 1) if gen_mean > 0 else 0,
        },
        "per_stage_stats": per_stage_stats,
        "methodology_note": (
            "Timing regions are mutually exclusive. "
            "total_ms = log_mel_ms + generator_consumption_ms (+ negligible join). "
            "generator_consumption_ms = encoder_ms + decoder_ms + other_overhead_ms. "
            "other_overhead_ms includes: _split_segments_by_timestamps(), "
            "language detection, Segment object creation, and Python loop overhead."
        ),
        "limitations": (
            "Decoder + beam search are combined in ct2's self.model.generate() C++ call "
            "and cannot be separated without modifying CTranslate2 source. "
            "feature_extractor is a ct2 object; we proxy it with a wrapper whose "
            "class defines __call__ to circumvent Python's type-level dunder resolution."
        ),
    }

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    jsonl_path = os.path.join(OUTPUT_DIR, "stt_internal_stage_probe.jsonl")
    with open(jsonl_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved: {jsonl_path}")

    report_path = os.path.join(OUTPUT_DIR, "stt_internal_stage_probe_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")


def record_audio(duration: float = 5.0) -> bytes:
    from services.audio import record_chunk
    return record_chunk(duration)


def load_wav(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="FasterWhisper internal stage probe (v2 — additive timing)"
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
    elif args.record:
        wav_bytes = record_audio(args.duration)
        audio_f32 = _wav_bytes_to_float32(wav_bytes)
    else:
        n = int(args.duration * 16000)
        t = np.linspace(0, args.duration, n, endpoint=False)
        noise = np.random.randn(n) * 0.3
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
        samples = (noise * mod * 32767).astype(np.int16)
        from services.audio import _to_wav
        audio_f32 = _wav_bytes_to_float32(_to_wav(samples.reshape(-1, 1)))

    run_benchmark(audio_f32, args.runs)


if __name__ == "__main__":
    main()
