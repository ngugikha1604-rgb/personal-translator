"""transcript_stability_runner.py — Entry point for the Transcript Stability Benchmark.

Measures how streaming transcripts evolve before converging to the final transcript.
Evaluates transcript STABILITY AFTER merging (the transcript a downstream LLM sees).

Usage:
    cd backend
    python benchmark/streaming/transcript_stability_runner.py \
        --librispeech LibriSpeech/ \
        --max-utterances 50 \
        --buffer-sizes 500 1000 2000 3000 4000

Output:
    benchmark_results/stability_results.jsonl
    benchmark_results/stability_report.json
    benchmark_results/stable_prefix_growth.png
    benchmark_results/stabilization_cdf.png
    benchmark_results/rollback_heatmap.png
    benchmark_results/waterfall.png
    benchmark_results/edit_distance_curve.png
"""

from __future__ import annotations
import json
import os
import sys
import time
import random
from io import BytesIO
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from faster_whisper import WhisperModel

import alignment as align_mod
import tracking as track_mod
import metrics as metrics_mod
import aggregation as agg_mod
import report as rpt_mod
import visualization as vis_mod
import merging as merge_mod


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmark_results"
)
SAMPLE_RATE = 16000
STREAMING_INTERVAL_MS = 500
STREAMING_INTERVAL_S = STREAMING_INTERVAL_MS / 1000.0

DEFAULT_BUFFER_SIZES_MS = [500, 1000, 1500, 2000, 3000, 4000]
DEFAULT_MODEL = "tiny.en"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE = "int8"
WARMUP_RUNS = 2
SLEEP_BETWEEN_UTTERANCES = 0.5


# ── LibriSpeech helpers ──


def _discover_librispeech(root_dir: str) -> list:
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


def _load_wav(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _wav_bytes_to_float32(wav_bytes: bytes) -> np.ndarray:
    with wave.open(BytesIO(wav_bytes), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    return audio / 32768.0


def _simulate_streaming_windows(
    audio_duration_s: float, buffer_s: float, interval_s: float
) -> list[tuple[float, float]]:
    windows = []
    t = 0.0
    while t < audio_duration_s:
        end = t + buffer_s
        end = min(end, audio_duration_s)
        windows.append((t, end))
        t += interval_s
    return windows


# ── Core benchmark ──


def benchmark_utterance(
    model: WhisperModel,
    audio_f32: np.ndarray,
    buffer_ms: int,
    utt_id: str,
    final_words: list[str],
    interval_ms: int = STREAMING_INTERVAL_MS,
) -> list[metrics_mod.UtteranceMetrics]:
    """Decode raw windows once. Then apply ALL merge strategies.
    Returns one UtteranceMetrics per strategy.
    """
    duration_s = len(audio_f32) / SAMPLE_RATE
    buffer_s = buffer_ms / 1000.0
    interval_s = interval_ms / 1000.0

    windows = _simulate_streaming_windows(duration_s, buffer_s, interval_s)
    if not windows:
        return []

    # ── Step 1: Decode each streaming window (raw, independent) ──
    raw_windows: list[metrics_mod.StreamingWindow] = []
    for win_idx, (win_start, win_end) in enumerate(windows):
        start_frame = int(win_start * SAMPLE_RATE)
        end_frame = min(int(win_end * SAMPLE_RATE), len(audio_f32))
        window_audio = audio_f32[start_frame:end_frame]
        if len(window_audio) == 0:
            raw_windows.append(metrics_mod.StreamingWindow(
                window_index=win_idx, buffer_ms=buffer_ms,
                start_time=win_start, end_time=win_end, transcript_words=[],
            ))
            continue
        gen, _ = model.transcribe(
            window_audio, language="en", temperature=0.0,
            beam_size=1, condition_on_previous_text=False, vad_filter=False,
        )
        texts = list(gen)
        transcript_text = " ".join(seg.text for seg in texts).strip()
        transcript_words = transcript_text.split() if transcript_text else []
        raw_windows.append(metrics_mod.StreamingWindow(
            window_index=win_idx, buffer_ms=buffer_ms,
            start_time=win_start, end_time=win_end, transcript_words=transcript_words,
        ))
    if not raw_windows:
        return []

    # ── Step 2: First raw appearance for each final word (for commit latency) ──
    first_raw: dict[int, int] = {}
    for win_idx, rw in enumerate(raw_windows):
        al = align_mod.align(final_words, rw.transcript_words)
        for a_idx, b_idx in al.pairs:
            if a_idx is not None and b_idx is not None and a_idx not in first_raw:
                first_raw[a_idx] = win_idx

    # ── Step 3: For each strategy, merge + track + compute metrics ──
    strategies: dict[str, merge_mod.MergeStrategy | None] = {
        "raw": None,
        **{k: v for k, v in merge_mod.all_strategies().items()},
    }
    results: list[metrics_mod.UtteranceMetrics] = []

    for strat_name, strategy in strategies.items():
        if strategy is not None:
            strategy.reset()

        # Build merged transcript sequence from raw windows
        merged_windows: list[metrics_mod.StreamingWindow] = []
        current_merged: list[str] = []
        for i, rw in enumerate(raw_windows):
            if strategy is None:
                merged_words = list(rw.transcript_words)
            else:
                merged_words = strategy.merge(rw.transcript_words, current_merged)
                current_merged = merged_words
            merged_windows.append(metrics_mod.StreamingWindow(
                window_index=i, buffer_ms=buffer_ms,
                start_time=rw.start_time, end_time=rw.end_time,
                transcript_words=merged_words,
            ))

        # Align merged transcripts to final
        merged_aligned: list[metrics_mod.AlignedWindow] = []
        for mw in merged_windows:
            al = align_mod.align(final_words, mw.transcript_words)
            merged_aligned.append(metrics_mod.AlignedWindow(window=mw, alignment=al))

        # Track word lifecycles + causal SPR
        tracker = track_mod.WordTracker(final_words)
        causal_spl, causal_spr = [], []
        for i, aw in enumerate(merged_aligned):
            tracker.update(aw.alignment, i)
            # Set raw first appearance on each lifecycle
            for a_idx, raw_win in first_raw.items():
                if a_idx < len(final_words):
                    wl = tracker.final_states()[a_idx]
                    if wl.first_raw_appearance_window is None:
                        wl.first_raw_appearance_window = raw_win
            cur = tracker.final_states()
            spl, spr = metrics_mod._stable_prefix(aw.alignment, cur)
            causal_spl.append(spl)
            causal_spr.append(spr)
            aw.stable_prefix_length = spl
            aw.stable_prefix_ratio = spr

        lifecycles = tracker.final_states()
        metrics = metrics_mod.compute_utterance_metrics(
            utt_id=utt_id, buffer_ms=buffer_ms, audio_duration_s=duration_s,
            final_words=final_words, windows=merged_windows,
            lifecycles=lifecycles, aligned_windows=merged_aligned,
            precomputed_stable_prefix_lengths=causal_spl,
            precomputed_stable_prefix_ratios=causal_spr,
            raw_windows=raw_windows, strategy_name=strat_name,
        )
        metrics.hallucination_count = tracker.hallucination_count
        results.append(metrics)

    return results


# ── Warmup ──


def _warmup(model: WhisperModel, audio_f32: np.ndarray, n: int = WARMUP_RUNS) -> None:
    if n <= 0:
        return
    segment = audio_f32[:SAMPLE_RATE] if len(audio_f32) >= SAMPLE_RATE else audio_f32
    print(f"  Warmup: {n} decode(s) ...", end=" ", flush=True)
    for _ in range(n):
        gen, _ = model.transcribe(
            segment, language="en", temperature=0.0,
            beam_size=1, condition_on_previous_text=False, vad_filter=False,
        )
        list(gen)
    print("done")
    time.sleep(0.5)


# ── Main ──


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Transcript Stability Benchmark (merged evaluation)"
    )
    parser.add_argument("--librispeech", type=str, default=None,
                        help="LibriSpeech dataset root directory")
    parser.add_argument("--wav", type=str, default=None, help="Single WAV (debug)")
    parser.add_argument("--transcript", type=str, default=None,
                        help="Word-level transcript JSON (requires --wav)")
    parser.add_argument("--max-utterances", type=int, default=50,
                        help="Max utterances to sample (default: 50)")
    parser.add_argument("--buffer-sizes", type=int, nargs="+",
                        default=DEFAULT_BUFFER_SIZES_MS,
                        help="Rolling buffer sizes in ms")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for utterance sampling")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Whisper model size (default: tiny.en)")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                        help="Device (cpu/cuda)")
    parser.add_argument("--compute", type=str, default=DEFAULT_COMPUTE,
                        help="Compute type (int8/float16/float32)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Resolve audio source ──
    audio_source: list[dict] = []
    if args.librispeech:
        utterances = _discover_librispeech(args.librispeech)
        if not utterances:
            print("ERROR: No valid LibriSpeech utterances found")
            sys.exit(1)
        discovered = len(utterances)
        if discovered > args.max_utterances:
            utterances = random.sample(utterances, args.max_utterances)
        print(f"  LibriSpeech: {len(utterances)} utterances from {discovered}")
        audio_source = utterances
    elif args.wav:
        audio_source = [{"wav_path": args.wav, "utt_id": os.path.basename(args.wav), "transcript_text": None}]
    else:
        print("Need --librispeech or --wav")
        sys.exit(1)

    # ── Init model ──
    print(f"  Model: {args.model} ({args.device}, {args.compute})")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute)

    first_src = audio_source[0]
    if "flac_path" in first_src:
        first_f32 = _flac_to_float32(first_src["flac_path"])
    else:
        first_f32 = _wav_bytes_to_float32(_load_wav(first_src.get("wav_path", args.wav)))
    _warmup(model, first_f32, WARMUP_RUNS)

    # ── Process utterances ──
    all_metrics: list[metrics_mod.UtteranceMetrics] = []
    for src_idx, src in enumerate(audio_source):
        utt_id = src["utt_id"]
        print(f"\n  [{src_idx+1}/{len(audio_source)}] {utt_id}")

        # Load audio
        if "flac_path" in src:
            audio_f32 = _wav_bytes_to_float32(_load_flac(src["flac_path"]))
        else:
            audio_f32 = _wav_bytes_to_float32(_load_wav(src.get("wav_path", args.wav)))

        # Final transcript (reference)
        gen, _ = model.transcribe(
            audio_f32, language="en", temperature=0.0,
            beam_size=1, condition_on_previous_text=False,
            vad_filter=False, word_timestamps=True,
        )
        final_text = " ".join(seg.text for seg in list(gen)).strip()
        final_words = final_text.split() if final_text else []
        if not final_words:
            print("    [skipped — empty]")
            continue
        print(f"    {len(final_words)} words: {final_text[:80]}...")

        for bs in args.buffer_sizes:
            all_results = benchmark_utterance(
                model, audio_f32, bs, utt_id, final_words,
                interval_ms=STREAMING_INTERVAL_MS,
            )
            for mr in all_results:
                all_metrics.append(mr)
                n_stable = sum(1 for wl in mr.word_lifecycles if wl.stabilization_window is not None)
                print(f"    {bs:>5d}ms [{mr.strategy:>20s}] "
                      f"wins={len(mr.windows)} "
                      f"stable={n_stable}/{len(mr.word_lifecycles)} "
                      f"churn={mr.total_churn} "
                      f"rollback_max={mr.max_rollback}")
            time.sleep(0.2)
        time.sleep(SLEEP_BETWEEN_UTTERANCES)

    # ── Aggregate ──
    print(f"\n  Aggregating {len(all_metrics)} results...")
    aggregates = agg_mod.aggregate_utterances(all_metrics, args.buffer_sizes)
    baseline = agg_mod.add_offline_baseline(aggregates, all_metrics)

    config_params = {
        "model": args.model, "device": args.device, "compute_type": args.compute,
        "streaming_interval_ms": STREAMING_INTERVAL_MS, "warmup_runs": WARMUP_RUNS,
        "seed": args.seed, "merge_strategies": list(merge_mod.all_strategies().keys()),
    }

    rpt_mod.write_jsonl(all_metrics)
    rpt_mod.write_report(all_metrics, aggregates, args.buffer_sizes, config_params)
    rpt_mod.print_summary(all_metrics, aggregates, baseline)
    vis_mod.generate_plots(all_metrics, aggregates, args.buffer_sizes, args.seed)


def _flac_to_float32(path: str) -> np.ndarray:
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32)


if __name__ == "__main__":
    main()
