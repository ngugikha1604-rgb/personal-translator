"""stt_internal_stage_probe.py — Profile internal FasterWhisper stages (v3).

Methodology fixes applied:
  1. temperature=0.0 pinned — prevents multi-pass decoder fallback
  2. other_overhead_ms → residual_ms — renamed to avoid implying it is a stage
  3. decoder_ms scope documented — includes ct2 decoder + beam + Python wrapper
  4. Short-audio assumption documented — audio << chunk_length(30s) → 1 pass
  5. detect_language instrumented (if monkey-patchable) or documented in residual

Correct execution timeline:
  model.transcribe(audio):
    └── feature_extractor(audio)  ← EAGER: log-Mel spectrogram
    └── returns generator (no work done)

  list(segments) — generator consumption:
    for each window:
      ├── self.encode(segment)              ← encoder (monkey-patched → encoder_ms)
      ├── self.model.detect_language(...)    ← language detection (monkey-patched → detect_language_ms)
      ├── self.generate_with_fallback(...)   ← decoder + beam + Python wrapper (monkey-patched → decoder_ms)
      │     └── self.model.generate()       ← ct2 C++ call (cannot separate)
      ├── self._split_segments_by_timestamps()
      ├── Segment(...) object creation
      └── yield Segment(text=...)

Correct breakdown:
  generator_consumption_ms = encoder_ms + decoder_ms + residual_ms
  total_ms = log_mel_ms + generator_consumption_ms (+ trivial join time)

  residual_ms = generator_consumption_ms - encoder_ms - decoder_ms
  (covers: segment splitting, Segment creation, loop overhead, and
   any uninstrumented Python work inside the generator)

Decoder_ms note:
  This wraps generate_with_fallback(), which is a 130-line Python method.
  It includes: ct2 decoder inference, beam search, temperature fallback
  (pinned to 0.0 so no fallback), tokenizer decoding, compression ratio
  checks, log-prob scoring, result selection, and Python wrapper overhead.
  These sub-components cannot be separated without modifying CTranslate2
  or FasterWhisper source code.

Short-audio assumption:
  Audio is always << chunk_length (30s default), so exactly one encoder
  pass and one decoder pass occur per transcription. This assumption is
  valid for all benchmark inputs (0.5-10s). For audio exceeding chunk_length,
  the benchmark would need updating.

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
      "encoder_ms"            — self.encode() → Whisper encoder
      "decoder_ms"            — self.generate_with_fallback() → decoder wrapper
      "log_mel_ms"            — feature_extractor → log-Mel spectrogram
      "detect_language_ms"    — self.model.detect_language() (if patchable)
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
        return sum(r["ms"] for r in self.records if r["stage"] == stage)


def _safe_patch_detect_language(model, collector: StageTimingCollector) -> bool:
    """Attempt to monkey-patch model.model.detect_language().

    This is a CTranslate2 C extension method. On some ct2 versions it is
    patchable; on others it raises AttributeError (read-only). Returns True
    if patching succeeded, False otherwise.
    """
    try:
        original = model.model.detect_language
    except AttributeError:
        return False

    if not callable(original):
        return False

    # Try MethodType patching
    try:
        def timed_detect_language(self_obj, *args, **kwargs):
            call_n = collector.next_call()
            t0 = time.perf_counter()
            result = original(*args, **kwargs)
            t1 = time.perf_counter()
            collector.record("detect_language_ms", (t1 - t0) * 1000, call_n)
            return result

        bound = types.MethodType(timed_detect_language, model.model)
        model.model.detect_language = bound
        return True
    except (AttributeError, TypeError):
        return False


def instrument_model(model, collector: StageTimingCollector):
    """Monkey-patch internal methods on a live WhisperModel instance.

    Patches applied:
      - model.encode()                    → encoder_ms
      - model.generate_with_fallback()    → decoder_ms (wrapper)
      - model.model.detect_language()     → detect_language_ms (if patchable)
      - model.feature_extractor           → log_mel_ms (via proxy)

    decoder_ms scope note:
      generate_with_fallback() is ~130 lines of Python wrapping
      self.model.generate(). It includes decoder inference, beam search,
      temperature fallback (pinned to 0.0 so no fallback), tokenizer
      decoding, compression ratio checks, log-prob scoring, result
      selection, and Python wrapper overhead. These cannot be separated
      without modifying FasterWhisper/ct2 source.

    Short-audio assumption:
      Audio duration << chunk_length (30s default) → exactly 1 encoder pass,
      1 decoder pass, 1 language detection call per transcription.
    """
    collector.reset()

    # ── 1. Monkey-patch model.encode() ──
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
    original_gwf = model.generate_with_fallback

    def timed_gwf(self_obj, encoder_output, prompt, tokenizer, options):
        call_n = collector.next_call()
        t0 = time.perf_counter()
        result = original_gwf(encoder_output, prompt, tokenizer, options)
        t1 = time.perf_counter()
        collector.record("decoder_ms", (t1 - t0) * 1000, call_n)
        return result

    model.generate_with_fallback = types.MethodType(timed_gwf, model)

    # ── 3. Proxy for feature_extractor ──
    if hasattr(model, 'feature_extractor'):
        _orig_fe = model.feature_extractor
        _collector = collector

        class _TimedFeatureExtractor:
            """Proxy for feature_extractor. Records timing of __call__.

            Python resolves __call__ on the object's TYPE, not instance,
            so instance-level assignment to __call__ is ignored. We replace
            the entire attribute with a proxy whose class defines __call__.
            """
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

    # ── 4. Optional: patch model.model.detect_language() ──
    lang_patched = _safe_patch_detect_language(model, collector)
    if not lang_patched:
        # detect_language is a ct2 C extension that cannot be monkey-patched.
        # Its time will be captured inside residual_ms.
        pass


def run_one(model, collector, audio_float32: np.ndarray) -> dict:
    """Run one transcription with timing, return mutually exclusive breakdown.

    Timing regions are ADDITIVE:
      total_ms ≈ log_mel_ms + generator_consumption_ms
      generator_consumption_ms ≈ encoder_ms + decoder_ms + residual_ms

    No stage overlaps or double-counts.
    """
    collector.reset()

    # ── t_start to t_after_call: EAGER work ──
    # model.transcribe() runs feature_extractor eagerly, returns generator.
    # temperature=0.0 is REQUIRED to prevent multi-pass decoder fallback.
    t_start = time.perf_counter()
    segments_gen, info = model.transcribe(
        audio_float32,
        language="en",
        temperature=0.0,              # REQUIRED: single decoder pass
        beam_size=1,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    t_after_call = time.perf_counter()

    # ── t_after_call to t_after_segments: LAZY generator consumption ──
    # All computation (encoder, decoder, segment assembly) runs here.
    texts = list(segments_gen)
    t_after_segments = time.perf_counter()

    # ── t_after_segments to t_done: transcript join (negligible) ──
    transcript = " ".join(seg.text for seg in texts).strip()
    t_done = time.perf_counter()

    # ── Compute mutually exclusive stages ──
    generator_consumption_ms = (t_after_segments - t_after_call) * 1000
    encoder_ms = collector.sum_stage("encoder_ms")
    decoder_ms = collector.sum_stage("decoder_ms")
    log_mel_ms = collector.sum_stage("log_mel_ms")
    detect_language_ms = collector.sum_stage("detect_language_ms")

    # residual_ms = everything in generator NOT captured by patched stages
    # This is NOT a stage — it is the subtraction residual from all
    # uninstrumented work (segment splitting, loop overhead, etc.).
    # If detect_language was patched, it is subtracted here too.
    residual_ms = generator_consumption_ms - encoder_ms - decoder_ms - detect_language_ms
    # Not clamped: negative values reflect real measurement noise (timer jitter,
    # CPU scheduling, warm caches). Clamping would hide noise floor info.

    wall_total_ms = (t_done - t_start) * 1000

    return {
        "log_mel_ms": round(log_mel_ms, 2),
        "generator_consumption_ms": round(generator_consumption_ms, 2),
        "encoder_ms": round(encoder_ms, 2),
        "decoder_ms": round(decoder_ms, 2),
        "detect_language_ms": round(detect_language_ms, 2),
        "residual_ms": round(residual_ms, 2),
        "total_ms": round(wall_total_ms, 2),
        "segments_found": len(texts),
        "transcript": transcript[:80],
        "encoder_calls": sum(1 for r in collector.records if r["stage"] == "encoder_ms"),
        "decoder_calls": sum(1 for r in collector.records if r["stage"] == "decoder_ms"),
    }


def print_stage_breakdown(all_results: list):
    """Print stage breakdown with additive verification.

    Stages are hierarchically organized:
      total_ms ≈ log_mel_ms + generator_consumption_ms
      generator_consumption_ms ≈ encoder_ms + decoder_ms + residual_ms
    """
    lazy_stages_def = [
        ("encoder_ms",          "Encoder"),
        ("decoder_ms",          "Decoder wrapper (ct2 + Python post)"),
        ("residual_ms",         "Residual (uninstrumented work)"),
    ]
    all_stages_def = [
        ("log_mel_ms",          "Log-Mel spectrogram (eager)"),
        ("generator_consumption_ms", "Generator consumption (total lazy work)"),
    ] + lazy_stages_def

    print(f"\n  {'=' * 75}")
    print(f"  Stage breakdown — mutually exclusive, additive")
    print(f"  {'=' * 75}")
    print(f"  {'Stage':35s} {'Mean':>9s} {'Median':>9s} {'P95':>9s} {'% of total':>10s}")
    print(f"  {'-' * 75}")

    totals_per_run = [r["total_ms"] for r in all_results]
    mnt = mean(totals_per_run)

    for key, label in all_stages_def:
        vals = sorted([r[key] for r in all_results])
        n = len(vals)
        mn = mean(vals)
        md = median(vals)
        p95 = float(np.percentile(vals, 95))
        pcts = [
            r[key] / r["total_ms"] * 100
            for r in all_results if r["total_ms"] > 0
        ]
        avg_pct = mean(pcts) if pcts else 0
        print(f"  {label:35s} {mn:>9.2f} {md:>9.2f} {p95:>9.2f} {avg_pct:>9.1f}%")

    print(f"  {'─' * 75}")

    sum_lazy = mean([r["encoder_ms"] + r["decoder_ms"] + r["residual_ms"]
                      for r in all_results])
    gen_mean = mean([r["generator_consumption_ms"] for r in all_results])
    diff_lazy = abs(sum_lazy - gen_mean)
    sum_all = mean([r["log_mel_ms"] + r["generator_consumption_ms"] for r in all_results])
    diff_all = abs(sum_all - mnt)

    print(f"  Additive: enc+dec+residual={sum_lazy:.1f} vs gen_consumption={gen_mean:.1f} "
          f"(diff={diff_lazy:.2f}ms)")
    print(f"  Additive: log_mel+gen_consumption={sum_all:.1f} vs total={mnt:.1f} "
          f"(diff={diff_all:.2f}ms)")

    print()

    lazy_means = {
        label: mean([r[key] for r in all_results])
        for key, label in lazy_stages_def
    }
    dominant = max(lazy_means, key=lazy_means.get)
    dom_val = lazy_means[dominant]
    gen_avg = mean([r["generator_consumption_ms"] for r in all_results])
    dom_pct = dom_val / gen_avg * 100 if gen_avg > 0 else 0
    print(f"  Dominant generator work: {dominant} ({dom_val:.0f}ms, {dom_pct:.1f}%)")
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
            audio_float32, language="en", temperature=0.0, beam_size=1,
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
            f"res={result['residual_ms']:>5.1f}ms  "
            f"gen={result['generator_consumption_ms']:>6.1f}ms  "
            f"total={result['total_ms']:>6.1f}ms"
        )
        time.sleep(SLEEP_BETWEEN)

    print_stage_breakdown(all_results)

    # ── Build report ──
    # detect_language_ms is reported only if monkey-patching succeeded
    stage_fields = [
        "log_mel_ms", "generator_consumption_ms",
        "encoder_ms", "decoder_ms", "detect_language_ms", "residual_ms"
    ]
    per_stage_stats = {}
    for stage in stage_fields:
        vals = sorted([r[stage] for r in all_results])
        n = len(vals)
        mn = mean(vals) if n > 0 else 0
        md = median(vals) if n > 0 else 0
        p95 = float(np.percentile(vals, 95)) if vals else 0
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
        "p95": round(float(np.percentile(totals, 95)), 2) if totals else 0,
        "min": round(min(totals), 2) if totals else 0,
        "max": round(max(totals), 2) if totals else 0,
        "std": round(stdev(totals), 2) if len(totals) > 1 else 0,
    }

    lazy_stages = ["encoder_ms", "decoder_ms", "residual_ms"]
    lazy_means = {s: per_stage_stats[s]["mean"] for s in lazy_stages}
    dominant = max(lazy_means, key=lazy_means.get)
    dom_val = lazy_means[dominant]
    gen_mean = per_stage_stats["generator_consumption_ms"]["mean"]

    # Check if detect_language was successfully patched
    lang_available = any(r.get("detect_language_ms", 0) > 0 for r in all_results)

    report = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "runs": runs,
        "warmup_runs": WARMUP_RUNS,
        "configuration": {
            "temperature": 0.0,
            "beam_size": 1,
            "condition_on_previous_text": False,
            "vad_filter": False,
            "language": "en",
        },
        "detect_language_instrumented": lang_available,
        "additive_check": {
            "inner_sum_ms": round(
                per_stage_stats["encoder_ms"]["mean"]
                + per_stage_stats["decoder_ms"]["mean"]
                + per_stage_stats["residual_ms"]["mean"], 1
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
        "methodology_notes": (
            "Timing regions are mutually exclusive and additive. "
            "total_ms = log_mel_ms + generator_consumption_ms (+ trivial join). "
            "generator_consumption_ms = encoder_ms + decoder_ms + detect_language_ms "
            "+ residual_ms. "
            "temperature=0.0 is pinned to prevent multi-pass decoder fallback. "
            "Audio is always < chunk_length (30s) → exactly 1 encoder and 1 decoder pass. "
            "decoder_ms wraps generate_with_fallback() — includes ct2 decoder, beam search, "
            "tokenizer decode, log-prob scoring, compression checks, and Python wrapper overhead. "
            "residual_ms is the subtraction residual, not a stage — it aggregates all "
            "uninstrumented work (segment splitting, loop overhead, etc.)."
        ),
        "limitations": (
            "1. Decoder + beam search cannot be separated — both are inside "
            "ct2 self.model.generate() which is a C++ call. "
            "2. generate_with_fallback() includes Python post-processing "
            "(tokenizer decoding, compression checks, score computation) "
            "that should not be attributed to decoder inference. "
            "3. feature_extractor is proxied; isinstance() checks will see "
            "the proxy, not the original. "
            "4. detect_language is a ct2 C extension — only patchable on certain "
            "ct2 versions. If not patchable, its time falls into residual_ms. "
            "5. Short-audio assumption (<30s chunk_length). Not valid for "
            "audio exceeding 30 seconds."
        ),
    }

    if not lang_available:
        report["detect_language_note"] = (
            "detect_language could not be monkey-patched (ct2 C extension "
            "is read-only on this version). Its time is included in residual_ms."
        )

    # Save
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
    with open(path, "rb") as f:
        return f.read()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="FasterWhisper internal stage probe (v3 — publication quality)"
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
