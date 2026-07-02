"""window_edge_debug.py — Debug tool: visualize per-window transcripts for ONE
utterance and test the "sliding window truncates context" hypothesis.

Two things this script does that the full benchmark doesn't show you:
  1. Prints the raw transcript of every streaming window, one after another,
     so you can SEE the churn happen instead of just reading a churn number.
  2. For every word in the final (reference) transcript, computes how much
     PRECEDING audio context that word had in each window it appeared in
     (word_start_time - window_start_time), and whether the word was decoded
     correctly in that window. If correctness drops as preceding context
     shrinks, that's direct evidence the sliding window (not the model) is
     the problem — later windows literally give the model less to work with
     for words near the start of the buffer.

Usage:
    cd backend
    python benchmark/streaming/window_edge_debug.py \
        --librispeech LibriSpeech-dataset --index 0 \
        --buffer-ms 4000 --interval-ms 500

    # Or pick by utt_id directly (e.g. one from the run you already saw):
    python benchmark/streaming/window_edge_debug.py \
        --librispeech LibriSpeech-dataset --utt-id 84-121550-0007 \
        --buffer-ms 4000 --interval-ms 500

    # Or a single WAV file:
    python benchmark/streaming/window_edge_debug.py --wav myclip.wav \
        --buffer-ms 4000 --interval-ms 500
"""

from __future__ import annotations
import argparse
import os
import random
import sys
from io import BytesIO
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))  # so `import alignment` works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from faster_whisper import WhisperModel
from alignment import align, normalize_word


SAMPLE_RATE = 16000


# ── Reuse the same discovery/loading logic as the other streaming scripts ──

def _discover_librispeech(root_dir: str) -> list:
    results = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        trans_files = [f for f in filenames if f.endswith(".trans.txt")]
        flac_files = [f for f in filenames if f.endswith(".flac")]
        if not trans_files or not flac_files:
            continue
        trans_map = {}
        for tfn in trans_files:
            with open(os.path.join(dirpath, tfn), "r", encoding="utf-8") as f:
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


def _flac_to_float32(path: str) -> np.ndarray:
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32)


def _wav_to_float32(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    return audio / 32768.0


def _simulate_streaming_windows(duration_s: float, buffer_s: float, interval_s: float):
    windows = []
    t = 0.0
    while t < duration_s:
        end = min(t + buffer_s, duration_s)
        windows.append((t, end))
        t += interval_s
    return windows


# ── Main debug logic ──

def main() -> None:
    p = argparse.ArgumentParser(description="Debug: per-window transcripts + context-vs-correctness")
    p.add_argument("--librispeech", type=str, default=None)
    p.add_argument("--wav", type=str, default=None)
    p.add_argument("--index", type=int, default=0, help="Which sampled utterance to use (0-based)")
    p.add_argument("--utt-id", type=str, default=None, help="Pick a specific utt_id instead of --index")
    p.add_argument("--seed", type=int, default=42, help="Must match the seed used in the main run to reproduce the same 50-utterance sample")
    p.add_argument("--max-utterances", type=int, default=50, help="Must match the main run's sample size")
    p.add_argument("--buffer-ms", type=int, default=4000)
    p.add_argument("--interval-ms", type=int, default=500)
    p.add_argument("--model", type=str, default="tiny.en")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--compute", type=str, default="int8")
    p.add_argument("--context-threshold-s", type=float, default=1.0,
                    help="Words with less than this much preceding context are labeled 'edge' in the summary table")
    args = p.parse_args()

    # ── Resolve audio source (reproduces the same sample as the main run if --seed/--max-utterances match) ──
    if args.librispeech:
        utterances = _discover_librispeech(args.librispeech)
        discovered = len(utterances)
        random.seed(args.seed)
        if discovered > args.max_utterances:
            utterances = random.sample(utterances, args.max_utterances)
        if args.utt_id:
            match = [u for u in utterances if u["utt_id"] == args.utt_id]
            if not match:
                print(f"ERROR: utt_id '{args.utt_id}' not found in the {len(utterances)}-utterance sample "
                      f"(seed={args.seed}, max_utterances={args.max_utterances}). "
                      f"Make sure these match the main benchmark run.")
                sys.exit(1)
            src = match[0]
        else:
            if args.index >= len(utterances):
                print(f"ERROR: index {args.index} out of range (sample has {len(utterances)} utterances)")
                sys.exit(1)
            src = utterances[args.index]
        print(f"  Utterance: {src['utt_id']}")
        audio_f32 = _flac_to_float32(src["flac_path"])
    elif args.wav:
        print(f"  WAV file: {args.wav}")
        audio_f32 = _wav_to_float32(args.wav)
    else:
        print("Need --librispeech <path> [--index N | --utt-id ID] or --wav <path>")
        sys.exit(1)

    duration_s = len(audio_f32) / SAMPLE_RATE
    print(f"  Duration: {duration_s:.2f}s")
    print(f"  Model: {args.model} ({args.device}, {args.compute})")
    print(f"  Buffer: {args.buffer_ms}ms, Interval: {args.interval_ms}ms\n")

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute)

    # Warmup
    seg = audio_f32[:SAMPLE_RATE] if len(audio_f32) >= SAMPLE_RATE else audio_f32
    list(model.transcribe(seg, language="en", temperature=0.0, beam_size=1,
                           condition_on_previous_text=False, vad_filter=False)[0])

    # ── Full-audio decode WITH word timestamps (reference + word timing) ──
    gen, _ = model.transcribe(
        audio_f32, language="en", temperature=0.0, beam_size=1,
        condition_on_previous_text=False, vad_filter=False, word_timestamps=True,
    )
    segments = list(gen)
    final_words: list[str] = []
    word_times: list[tuple[float, float]] = []  # (start, end) per final word
    for seg_obj in segments:
        if seg_obj.words:
            for w in seg_obj.words:
                final_words.append(w.word.strip())
                word_times.append((w.start, w.end))
        else:
            # Fallback: no word-level timestamps available, split segment text
            for tok in seg_obj.text.strip().split():
                final_words.append(tok)
                word_times.append((seg_obj.start, seg_obj.end))

    print(f"  Final transcript ({len(final_words)} words):")
    print(f"    {' '.join(final_words)}\n")

    # ── Run streaming windows ──
    buffer_s = args.buffer_ms / 1000.0
    interval_s = args.interval_ms / 1000.0
    windows = _simulate_streaming_windows(duration_s, buffer_s, interval_s)

    window_texts: list[str] = []
    window_word_lists: list[list[str]] = []
    print("=" * 100)
    print("  PER-WINDOW TRANSCRIPTS (watch how much changes between consecutive windows)")
    print("=" * 100)
    for i, (ws, we) in enumerate(windows):
        start_frame = int(ws * SAMPLE_RATE)
        end_frame = min(int(we * SAMPLE_RATE), len(audio_f32))
        chunk = audio_f32[start_frame:end_frame]
        if len(chunk) == 0:
            text = ""
        else:
            g, _ = model.transcribe(chunk, language="en", temperature=0.0, beam_size=1,
                                     condition_on_previous_text=False, vad_filter=False)
            text = " ".join(s.text for s in g).strip()
        window_texts.append(text)
        words = text.split() if text else []
        window_word_lists.append(words)

        # Diff vs previous window (word-level edit distance) for a quick "churn" signal per line
        if i == 0:
            diff_note = ""
        else:
            al = align(window_word_lists[i - 1], words)
            diff_note = f"  [Δ vs prev: {al.edit_distance} edits]"

        print(f"  [{i:>2d}] t=[{ws:>5.2f}s, {we:>5.2f}s)  {text!r}{diff_note}")

    # ── Context-vs-correctness table ──
    print()
    print("=" * 100)
    print(f"  CONTEXT vs CORRECTNESS  (edge = less than {args.context_threshold_s:.1f}s of preceding audio in that window)")
    print("=" * 100)
    print(f"  {'word':<18} {'win':>4} {'preceding_s':>12} {'following_s':>12} {'correct?':>9} {'edge?':>6}")

    edge_correct = 0
    edge_total = 0
    deep_correct = 0
    deep_total = 0

    for w_idx, (word, (wstart, wend)) in enumerate(zip(final_words, word_times)):
        for win_idx, (ws, we) in enumerate(windows):
            if not (ws <= wstart < we):
                continue  # this word's audio doesn't start inside this window
            preceding = wstart - ws
            following = we - wend
            # Determine whether this word was decoded correctly in this window
            # by aligning the window's word list to the final transcript and
            # checking the aligned surface at this reference index.
            al = align(final_words, window_word_lists[win_idx])
            surface = None
            for a, b in al.pairs:
                if a == w_idx and b is not None:
                    surface = al.b_words[b]
                    break
            is_correct = surface is not None and normalize_word(surface) == normalize_word(word)
            is_edge = preceding < args.context_threshold_s

            if is_edge:
                edge_total += 1
                if is_correct:
                    edge_correct += 1
            else:
                deep_total += 1
                if is_correct:
                    deep_correct += 1

            print(f"  {word:<18} {win_idx:>4} {preceding:>12.2f} {following:>12.2f} "
                  f"{'yes' if is_correct else 'no':>9} {'EDGE' if is_edge else '':>6}")

    print()
    print("=" * 100)
    print("  SUMMARY — does correctness drop when a word has little preceding context?")
    print("=" * 100)
    edge_rate = (edge_correct / edge_total * 100) if edge_total else float("nan")
    deep_rate = (deep_correct / deep_total * 100) if deep_total else float("nan")
    print(f"  Edge occurrences  (<{args.context_threshold_s:.1f}s preceding context): "
          f"{edge_correct}/{edge_total} correct ({edge_rate:.1f}%)")
    print(f"  Deep occurrences (>={args.context_threshold_s:.1f}s preceding context): "
          f"{deep_correct}/{deep_total} correct ({deep_rate:.1f}%)")
    print()
    if edge_total > 0 and deep_total > 0:
        if deep_rate - edge_rate > 15:
            print("  -> STRONG evidence the sliding window (truncated context) is causing instability,")
            print("     not the model itself. A word gets MUCH less reliable when it has little")
            print("     preceding audio in that window, regardless of which model would decode it.")
        elif abs(deep_rate - edge_rate) <= 15:
            print("  -> WEAK/no correlation between preceding context and correctness.")
            print("     This suggests the model itself (not the windowing scheme) may be the")
            print("     dominant source of instability — worth benchmarking a stronger model.")
        else:
            print("  -> Unexpected: edge words are MORE correct than deep words. Investigate further")
            print("     before drawing conclusions (could be an artifact of this one utterance).")
    else:
        print("  -> Not enough data points in one or both buckets to draw a conclusion.")
        print("     Try a longer utterance or a smaller --context-threshold-s.")


if __name__ == "__main__":
    main()
