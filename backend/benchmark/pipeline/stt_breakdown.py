"""stt_breakdown.py — Measure STT latency in detail.

Timing breakdown:
  audio_load_ms      — reading WAV file from disk (0 if using --record)
  audio_prepare_ms   — any conversion before API call (currently negligible for local provider)
  api_request_ms     — time inside STT transcription call (for local provider: in-process inference)
  inference_ms       — alias for api_request_ms with clearer name for local inference
  response_parse_ms  — extracting transcript text from response
  total_ms           — sum of all stages

Usage:
    cd backend
    python benchmark/pipeline/stt_breakdown.py --wav path.wav          # single file 20x
    python benchmark/pipeline/stt_breakdown.py --record                # record then transcribe 20x
    python benchmark/pipeline/stt_breakdown.py --runs 50 --wav path    # 50 iterations
    python benchmark/pipeline/stt_breakdown.py \
        --librispeech LibriSpeech/ \
        --max-utterances 50 --runs-per-utterance 3                    # multi-utterance mode

Output:
    benchmark_results/stt_breakdown.jsonl
    (also prints summary to stdout)
"""

import json
import os
import sys
import time
import random
from statistics import mean, median
from io import BytesIO
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from services.stt_factory import get_stt_provider
from services.stt_provider import STTProvider

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmark_results"
)
SLEEP_BETWEEN = 0.5


# ── LibriSpeech helpers (self-contained, not imported) ──


def _discover_librispeech(root_dir: str) -> list:
    """Recursively discover all valid LibriSpeech utterances.

    Expected directory structure:
      LibriSpeech/<subset>/<speaker_id>/<chapter_id>/
        <speaker_id>-<chapter_id>-<utterance_id>.flac
        <speaker_id>-<chapter_id>.trans.txt

    Returns list of dicts:
      [{"flac_path": ..., "utt_id": ..., "transcript_text": ...}, ...]
    """
    results = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        trans_files = [f for f in filenames if f.endswith(".trans.txt")]
        flac_files = [f for f in filenames if f.endswith(".flac")]

        if not trans_files or not flac_files:
            continue

        trans_map = {}
        for tfn in trans_files:
            tp = os.path.join(dirpath, tfn)
            with open(tp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        trans_map[parts[0]] = parts[1]

        for flac_fn in flac_files:
            utt_id = flac_fn.replace(".flac", "")
            text = trans_map.get(utt_id)
            if text:
                results.append({
                    "flac_path": os.path.join(dirpath, flac_fn),
                    "utt_id": utt_id,
                    "transcript_text": text,
                })
    return results


def _load_flac(path: str) -> bytes:
    """Load a FLAC file and return WAV bytes (int16 PCM)."""
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32")
    int16_data = (data * 32767).astype(np.int16)
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        if int16_data.ndim == 1:
            wf.setnchannels(1)
        else:
            wf.setnchannels(int16_data.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_data.tobytes())
    return buf.getvalue()


# ── Benchmark logic ──


def benchmark_provider(
    stt: STTProvider,
    audio_bytes: bytes,
    filename: str,
    runs: int,
    warmup_runs: int = 2,
) -> list:
    """Run transcription N times with detailed timing.

    If warmup_runs > 0, performs that many un-timed decodes first to
    prime the model / CPU cache before collecting measurements.
    """
    # Warmup: un-timed decodes to prime model state
    if warmup_runs > 0:
        for _ in range(warmup_runs):
            try:
                stt.transcribe(audio_bytes, filename)
            except Exception:
                pass  # ignore errors during warmup

    # Provider metadata (constant across all rows)
    provider_class = type(stt).__name__
    provider_model = getattr(stt, "model_size", getattr(stt, "model_name", "unknown"))
    provider_device = getattr(stt, "device", "unknown")
    provider_compute = getattr(stt, "compute_type", "unknown")

    all_rows = []

    for i in range(runs):
        t_load_start = time.perf_counter()

        # Already loaded — audio_load_ms = 0 for pre-loaded bytes
        t_load_done = time.perf_counter()

        # audio_prepare_ms: any conversion before API (today: none)
        t_prep_start = time.perf_counter()
        t_prep_done = time.perf_counter()

        # api_request_ms: the STT call (local inference for local provider)
        t_api_start = time.perf_counter()
        try:
            transcription = stt.transcribe(audio_bytes, filename)
        except Exception as exc:
            t_api_end = time.perf_counter()
            api_ms = round((t_api_end - t_api_start) * 1000, 2)
            row = {
                "audio_load_ms": round((t_load_done - t_load_start) * 1000, 2),
                "audio_prepare_ms": round((t_prep_done - t_prep_start) * 1000, 2),
                "api_request_ms": api_ms,
                "inference_ms": api_ms,
                "response_parse_ms": 0,
                "total_ms": round((t_api_end - t_load_start) * 1000, 2),
                "transcript_length_words": 0,
                "error": str(exc)[:200],
                "provider": provider_class,
                "model": provider_model,
                "device": provider_device,
                "compute_type": provider_compute,
            }
            all_rows.append(row)
            time.sleep(SLEEP_BETWEEN)
            continue

        t_api_end = time.perf_counter()

        # response_parse_ms: extract text from response
        t_parse_start = time.perf_counter()
        transcript = transcription.strip()
        t_parse_end = time.perf_counter()

        api_ms = round((t_api_end - t_api_start) * 1000, 2)
        row = {
            "audio_load_ms": round((t_load_done - t_load_start) * 1000, 2),
            "audio_prepare_ms": round((t_prep_done - t_prep_start) * 1000, 2),
            "api_request_ms": api_ms,
            "inference_ms": api_ms,
            "response_parse_ms": round((t_parse_end - t_parse_start) * 1000, 2),
            "total_ms": round((t_parse_end - t_load_start) * 1000, 2),
            "transcript_length_words": len(transcript.split()),
            "transcript": transcript[:100],
            "provider": provider_class,
            "model": provider_model,
            "device": provider_device,
            "compute_type": provider_compute,
        }

        all_rows.append(row)
        time.sleep(SLEEP_BETWEEN)

    return all_rows


def load_audio(path: str) -> bytes:
    if not os.path.isfile(path):
        abs_path = os.path.abspath(path)
        raise FileNotFoundError(
            f"WAV file not found: {abs_path}\n"
            f"  - Make sure the path is correct (run from backend/ dir)\n"
            f"  - Or use --record to capture from mic instead"
        )
    with open(path, "rb") as f:
        return f.read()


def record_audio(duration: float = 5.0) -> bytes:
    from services.audio import record_chunk
    print(f"  Recording {duration}s...")
    return record_chunk(duration)


def print_summary(all_rows: list, label: str):
    """Print timing breakdown summary.

    Uses np.percentile for correct P95 calculation (linear interpolation).
    """
    stages = ["audio_load_ms", "audio_prepare_ms", "api_request_ms",
              "inference_ms", "response_parse_ms", "total_ms"]
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
        p95 = round(float(np.percentile(vals, 95)), 1)
        lo = min(vals)
        hi = max(vals)

        # Percentage of total (using this row's own total)
        if stage == "total_ms":
            pct = "100%"
        else:
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
    parser.add_argument("--warmup", type=int, default=2, help="Untimed warmup decodes before measurements (default: 2)")
    parser.add_argument("--librispeech", type=str, default=None, help="LibriSpeech dataset root directory")
    parser.add_argument("--max-utterances", type=int, default=50, help="Max utterances to sample from LibriSpeech (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducing utterance sampling (default: 42)")
    parser.add_argument("--runs-per-utterance", type=int, default=3, help="Number of timed runs per utterance in LibriSpeech mode (default: 3)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stt = get_stt_provider()
    provider_class = type(stt).__name__
    print(f"  Provider: {provider_class}")

    rows = []

    if args.librispeech:
        # ── LibriSpeech multi-utterance mode ──
        utterances = _discover_librispeech(args.librispeech)
        if not utterances:
            print("ERROR: No valid LibriSpeech utterances found in", args.librispeech)
            sys.exit(1)
        discovered = len(utterances)
        sampled = utterances
        if discovered > args.max_utterances:
            random.seed(args.seed)
            sampled = random.sample(utterances, args.max_utterances)
        print(f"  LibriSpeech: {len(sampled)} utterances sampled from {discovered} discovered (seed={args.seed})")
        print(f"  Runs per utterance: {args.runs_per_utterance}")
        print(f"  Warmup: {args.warmup} runs (discarded) — before utterance loop")

        for utt_idx, utt in enumerate(sampled):
            print(f"  [{utt_idx+1}/{len(sampled)}] {utt['utt_id']} ...", end=" ", flush=True)
            wav_bytes = _load_flac(utt["flac_path"])
            # Warmup runs once using first utterance's audio; subsequent utterances skip warmup
            warmup = args.warmup if utt_idx == 0 else 0
            utt_rows = benchmark_provider(
                stt, wav_bytes, utt["utt_id"] + ".wav",
                runs=args.runs_per_utterance,
                warmup_runs=warmup,
            )
            for r in utt_rows:
                r["utt_id"] = utt["utt_id"]
            rows.extend(utt_rows)
            print(f"done ({len(utt_rows)} rows)")

        print(f"\n  Utterances: {len(sampled)} sampled from {discovered} discovered (seed={args.seed})")
        label = f"librispeech_{len(sampled)}utt"

    elif args.wav:
        print(f"  Loading WAV: {os.path.abspath(args.wav)}")
        audio = load_audio(args.wav)
        filename = os.path.basename(args.wav)
        label = os.path.basename(args.wav)
        print(f"  Audio: {len(audio)} bytes, {len(audio) / 32000:.1f}s estimated")
        print(f"  Runs: {args.runs}")
        if args.warmup > 0:
            print(f"  Warmup: {args.warmup} runs (discarded)")
        print()
        rows = benchmark_provider(stt, audio, filename, args.runs, warmup_runs=args.warmup)

    elif args.record:
        audio = record_audio(args.duration)
        filename = "recorded_chunk.wav"
        label = f"recorded_{args.duration}s"
        print(f"  Audio: {len(audio)} bytes, {len(audio) / 32000:.1f}s estimated")
        print(f"  Runs: {args.runs}")
        if args.warmup > 0:
            print(f"  Warmup: {args.warmup} runs (discarded)")
        print()
        rows = benchmark_provider(stt, audio, filename, args.runs, warmup_runs=args.warmup)

    else:
        print("Need --wav path.wav, --record, or --librispeech <path>")
        sys.exit(1)

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
