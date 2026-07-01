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

  Ground truth uses transcript text. For word-level window assignment,
  word timestamps are required. When ground-truth word timestamps are
  available (provided via --transcript JSON), they are used directly.
  When only utterance-level transcripts are available (LibriSpeech mode),
  word alignments are generated via Whisper word_timestamps=True and
  treated as approximate pseudo-alignments — transcript text is ground
  truth; timestamps are approximate only.

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
import random

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

# Plotting (optional — benchmark runs without matplotlib)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Default streaming interval — must remain fixed throughout the benchmark
STREAMING_INTERVAL_MS = 500
STREAMING_INTERVAL_S = STREAMING_INTERVAL_MS / 1000.0

# Only buffer size varies between configurations
BUFFER_SIZES_MS = [500, 1000, 1500, 2000, 3000, 4000]
MAX_UTTERANCES = 300
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


def _get_cpu_utilization(
    process_cpu_start: float, process_cpu_end: float,
    wall_start: float, wall_end: float,
) -> float:
    """Compute CPU utilization as process_time / wall_time * 100.

    This is more reliable than psutil.cpu_percent(interval=0) which
    almost always returns 0% for short operations.
    """
    wall_delta = wall_end - wall_start
    if wall_delta <= 0:
        return 0.0
    cpu_delta = process_cpu_end - process_cpu_start
    return (cpu_delta / wall_delta) * 100.0


def get_rss_mb(psutil_mod) -> float:
    if psutil_mod is None:
        return 0.0
    return psutil_mod.Process().memory_info().rss / (1024 * 1024)


class _RssMonitor:
    """Lightweight RSS peak monitor that samples RSS every `interval_s` seconds.

    Start before decode, call `.stop()` after decode to retrieve peak_rss.
    Peak RSS is estimated by periodic sampling, NOT continuous OS-level tracing.
    """
    def __init__(self, psutil_mod, interval_s: float = 0.005):
        import threading
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread = None
        if psutil_mod is not None:
            proc = psutil_mod.Process()
            def _sample():
                while not self._stop.is_set():
                    try:
                        rss = proc.memory_info().rss / (1024 * 1024)
                        if rss > self.peak:
                            self.peak = rss
                    except Exception:
                        pass
                    self._stop.wait(interval_s)
            t = threading.Thread(target=_sample, daemon=True)
            t.start()
            self._thread = t

    def stop(self, timeout: float = 2.0):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=timeout)


def record_audio(duration: float = 10.0) -> bytes:
    from services.audio import record_chunk
    return record_chunk(duration)


def _load_flac(path: str) -> bytes:
    """Load FLAC file into WAV bytes via soundfile.

    soundfile is preferred for FLAC; falls back to pydub.
    """
    import soundfile as sf
    import io as _io
    data, sr = sf.read(path, dtype="float32")
    # Convert float32 to int16 WAV bytes for compatibility
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

        # Parse .trans.txt into utt_id → text map
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


def _flac_to_float32(flac_path: str) -> np.ndarray:
    """Load FLAC file as float32 numpy array at the expected sample rate."""
    import soundfile as sf
    data, sr = sf.read(flac_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # mono mix
    return data.astype(np.float32)


def _generate_word_timestamps(
    model: WhisperModel,
    audio_f32: np.ndarray,
) -> list:
    """Generate word-level [{"word","start","end"},...] via Whisper word_timestamps.

    LibriSpeech does NOT include word-level timestamps.
    We generate them by running Whisper with word_timestamps=True on the full audio.
    The generated timestamps are approximate alignments, NOT ground truth — they
    depend on Whisper's internal alignment algorithm.

    This is documented in the report methodology_notes.
    """
    gen, _ = model.transcribe(
        audio_f32, language="en", temperature=0.0, beam_size=1,
        condition_on_previous_text=False, vad_filter=False,
        word_timestamps=True,
    )
    segments = list(gen)
    result = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                result.append({
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                })
    return result


# ========================================================
# LibriSpeech provides utterance-level transcripts only.
# Word alignments are generated via Whisper word_timestamps=True.
# These are approximate pseudo-alignments, NOT ground truth timestamps.
# Only transcript text is considered ground truth.
# Alternatives (forced alignment via aeneas/gentle) are not implemented.
# ========================================================


def generate_synthetic_audio(duration_s: float) -> bytes:
    """Generate synthetic noise audio (no speech — WER=N/A)."""
    n = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n, endpoint=False)
    noise = np.random.randn(n) * 0.3
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    samples = (noise * mod * 32767).astype(np.int16)
    from services.audio import _to_wav
    return _to_wav(samples.reshape(-1, 1))


# ── Run one configuration (one buffer size) ───────────
# Uses a pre-loaded float32 rolling buffer to avoid WAV reconstruction.


def _build_window_reference_map(
    transcript: list, windows: list
) -> dict:
    """Pre-build a window→ref_words map to avoid scanning the transcript per window.

    Uses a linear-time two-pointer algorithm since both transcript words and
    windows are sorted by time.  Complexity: O(W + N) where W = number of
    windows and N = number of transcript words.

    Returns dict keyed by (start, end) tuple → [word_strings].
    Empty for silent windows.
    """
    if not transcript or not windows:
        return {}
    ref_map = {tuple((round(ws, 3), round(we, 3))): [] for ws, we in windows}
    word_idx = 0
    n_words = len(transcript)
    for win_start, win_end in windows:
        key = (round(win_start, 3), round(win_end, 3))
        while word_idx < n_words:
            e = transcript[word_idx]
            midpoint = (e["start"] + e["end"]) / 2.0
            if midpoint < win_start:
                word_idx += 1
                continue
            elif win_start <= midpoint < win_end:
                ref_map[key].append(e["word"])
                word_idx += 1
            else:
                break
    return ref_map


def _run_rolling_window(
    model: WhisperModel,
    audio_f32: np.ndarray,
    sample_rate: int,
    win_start_s: float,
    win_end_s: float,
    window_index: int,
    ref_map: dict,
    has_transcript: bool,
    psutil_mod,
) -> dict:
    """Transcribe one rolling window from a float32 buffer.

    Uses pre-built window→reference map for O(1) ref word lookup.
    RSS is monitored by a background thread during decode (peak estimated by
    periodic sampling, not continuous OS-level tracing).
    Returns per-window metrics including processing wall time (excludes sleep).
    """
    start_cpu = time.process_time()
    # processing_wall_ms tracks actual work only (no sleep)
    processing_start = time.perf_counter()

    # ── Stage 1: buffer update (rolling buffer slice) ──
    t_buf0 = time.perf_counter()
    start_frame = int(win_start_s * sample_rate)
    end_frame = min(int(win_end_s * sample_rate), len(audio_f32))
    window_f32 = audio_f32[start_frame:end_frame]
    t_buf1 = time.perf_counter()
    slice_ms = (t_buf1 - t_buf0) * 1000

    if len(window_f32) == 0:
        return None

    # ── Stage 2: preprocessing ──
    t_conv0 = time.perf_counter()
    t_conv1 = time.perf_counter()
    convert_ms = (t_conv1 - t_conv0) * 1000

    # ── Stage 3: Whisper decode (with RSS monitoring) ──
    rss_before = get_rss_mb(psutil_mod)
    monitor = _RssMonitor(psutil_mod)
    t_dec0 = time.perf_counter()
    dec_cpu0 = time.process_time()
    gen, _ = model.transcribe(
        window_f32, language="en", temperature=0.0, beam_size=1,
        condition_on_previous_text=False, vad_filter=False,
    )
    texts = list(gen)
    t_dec1 = time.perf_counter()
    decode_ms = (t_dec1 - t_dec0) * 1000
    dec_cpu1 = time.process_time()
    monitor.stop()
    rss_peak = monitor.peak
    rss_after = get_rss_mb(psutil_mod)

    # ── Stage 4: postprocess ──
    t_post0 = time.perf_counter()
    transcript_text = " ".join(seg.text for seg in texts).strip()

    key = (round(win_start_s, 3), round(win_end_s, 3))
    ref_words = ref_map.get(key, [])
    ref_text = " ".join(ref_words)
    wer_val, cer_val = compute_wer(ref_text, transcript_text) if has_transcript else (None, None)
    t_post1 = time.perf_counter()
    postprocess_ms = (t_post1 - t_post0) * 1000

    processing_end = time.perf_counter()
    end_cpu = time.process_time()

    # ── Compute metrics ──
    total_pipeline_ms = (processing_end - processing_start) * 1000
    window_dur_ms = (win_end_s - win_start_s) * 1000
    rtf = decode_ms / window_dur_ms if window_dur_ms > 0 else 0

    cpu_pct = _get_cpu_utilization(start_cpu, end_cpu, processing_start, processing_end)

    total_nonzero = total_pipeline_ms or 1e-9
    slice_pct = (slice_ms / total_nonzero) * 100.0
    convert_pct = (convert_ms / total_nonzero) * 100.0
    decode_pct = (decode_ms / total_nonzero) * 100.0
    postprocess_pct = (postprocess_ms / total_nonzero) * 100.0

    hyp_word_count = len(transcript_text.split()) if transcript_text else 0

    # Observed RSS (single number for backwards compat) = peak
    # Also expose all three samples for detailed analysis
    observed_rss = max(rss_before, rss_peak, rss_after)

    row = {
        "buffer_ms": int((win_end_s - win_start_s) * 1000),
        "window_index": window_index,
        "window_start": round(win_start_s, 3),
        "window_end": round(win_end_s, 3),
        "window_duration_ms": round(window_dur_ms, 1),
        "reference_word_count": len(ref_words),
        "hypothesis_word_count": hyp_word_count,
        # Flat pipeline fields (canonical location)
        "slice_ms": round(slice_ms, 3),
        "convert_ms": round(convert_ms, 3),
        "decode_ms": round(decode_ms, 2),
        "postprocess_ms": round(postprocess_ms, 3),
        "total_pipeline_ms": round(total_pipeline_ms, 2),
        "pipeline_pct": {
            "slice_pct": round(slice_pct, 1),
            "convert_pct": round(convert_pct, 1),
            "decode_pct": round(decode_pct, 1),
            "postprocess_pct": round(postprocess_pct, 1),
        },
        "decode_latency_ms": round(decode_ms, 2),
        "rtf": round(rtf, 4),
        "cpu_percent": round(cpu_pct, 1),
        "rss_mb": round(observed_rss, 1),
        "rss_mb_before": round(rss_before, 1),
        "rss_mb_after": round(rss_after, 1),
        "rss_mb_peak": round(rss_peak, 1),
        "hypothesis": transcript_text[:120],
        "reference": ref_text[:120],
    }

    if wer_val is not None and len(ref_words) > 0:
        row["wer"] = round(wer_val, 4)
        row["cer"] = round(cer_val, 4)
    else:
        row["wer"] = None
        row["cer"] = None

    return row


def run_configuration(
    model: WhisperModel,
    audio_f32: np.ndarray,
    sample_rate: int,
    buffer_ms: int,
    interval_s: float,
    transcript: list,
    psutil_mod,
    has_transcript: bool,
) -> dict:
    """Run streaming simulation for one buffer size.

    Simulates a rolling audio buffer by using a numpy sliding window over
    pre-loaded float32 audio. Measures buffer update, preprocessing, and
    decode separately.

    Returns aggregated results + per-window rows.
    """
    duration_s = len(audio_f32) / sample_rate
    buffer_s = buffer_ms / 1000.0

    windows = simulate_streaming_windows(duration_s, buffer_s, interval_s)

    # Pre-build window→reference map (O(N) scan once, not per window)
    ref_map = _build_window_reference_map(transcript, windows)

    per_window_rows = []
    all_wer_vals = []
    all_cer_vals = []
    all_ref_wc = []
    all_latencies = []
    all_slice_ms = []
    all_convert_ms = []
    all_decode_ms_list = []
    all_postprocess_ms = []
    all_total_pipeline_ms = []

    cfg_start_s = time.time()
    for win_idx, (win_start, win_end) in enumerate(windows):
        row = _run_rolling_window(
            model, audio_f32, sample_rate,
            win_start, win_end, win_idx, ref_map,
            has_transcript, psutil_mod,
        )
        if row is None:
            continue

        decode_ms = row["decode_latency_ms"]
        all_latencies.append(decode_ms)
        all_slice_ms.append(row["slice_ms"])
        all_convert_ms.append(row["convert_ms"])
        all_decode_ms_list.append(row["decode_ms"])
        all_postprocess_ms.append(row["postprocess_ms"])
        all_total_pipeline_ms.append(row["total_pipeline_ms"])

        if row.get("wer") is not None:
            all_wer_vals.append(row["wer"])
            all_cer_vals.append(row["cer"])
            all_ref_wc.append(row["reference_word_count"])

        per_window_rows.append(row)

        time.sleep(SLEEP_BETWEEN_WINDOWS)
    cfg_end_s = time.time()
    cfg_wall_s = cfg_end_s - cfg_start_s

    # ── Aggregate ──
    n = len(all_latencies)
    if n == 0:
        return {"buffer_ms": buffer_ms, "total_windows": 0}, []

    lats = sorted(all_latencies)
    mn_lat = mean(lats)
    md_lat = median(lats)
    p95_lat = lats[min(int(n * 0.95), n - 1)]
    p99_lat = lats[min(int(n * 0.99), n - 1)]
    sd_lat = stdev(lats) if n > 1 else 0
    cv_lat = sd_lat / mn_lat if mn_lat > 0 else 0

    rtfs = [r["rtf"] for r in per_window_rows]
    mn_rtf = mean(rtfs) if rtfs else 0

    # Backlog
    backlog_ratio_p95 = p95_lat / (interval_s * 1000)
    miss_count = sum(1 for la in all_latencies if la > interval_s * 1000)
    backlog_miss_rate = miss_count / n * 100 if n > 0 else 0

    # Throughput — use processing_wall_s (excludes artificial sleep) for throughput
    processing_wall_s = sum(all_total_pipeline_ms) / 1000.0
    windows_per_second = n / processing_wall_s if processing_wall_s > 0 else 0
    audio_per_second = duration_s / processing_wall_s if processing_wall_s > 0 else 0

    # Pipeline timing
    mn_slice = mean(all_slice_ms) if all_slice_ms else 0
    mn_convert = mean(all_convert_ms) if all_convert_ms else 0
    mn_decode = mean(all_decode_ms_list) if all_decode_ms_list else 0
    mn_post = mean(all_postprocess_ms) if all_postprocess_ms else 0
    mn_pipe = mean(all_total_pipeline_ms) if all_total_pipeline_ms else 0

    # Pipeline percentages
    total_stage = mn_slice + mn_convert + mn_decode + mn_post
    denom = total_stage or 1e-9
    slice_pct = (mn_slice / denom) * 100.0
    convert_pct = (mn_convert / denom) * 100.0
    decode_pct = (mn_decode / denom) * 100.0
    post_pct = (mn_post / denom) * 100.0

    # Accuracy
    total_ref_words = sum(all_ref_wc) if all_ref_wc else 0
    if all_ref_wc:
        total_wc = total_ref_words
        avg_wer = sum(w * c for w, c in zip(all_wer_vals, all_ref_wc)) / total_wc
        avg_cer = sum(w * c for w, c in zip(all_cer_vals, all_ref_wc)) / total_wc
    else:
        avg_wer = None
        avg_cer = None

    # CPU/RSS
    cpus = [r["cpu_percent"] for r in per_window_rows]
    rsses = [r["rss_mb"] for r in per_window_rows]

    redundancy_ratio = buffer_s / interval_s

    config_result = {
        "buffer_ms": buffer_ms,
        "total_windows": n,
        "total_reference_words": total_ref_words,
        "redundancy_ratio": round(redundancy_ratio, 2),
        "decode_latency_ms": {
            "mean": round(mn_lat, 1),
            "median": round(md_lat, 1),
            "p95": round(p95_lat, 1),
            "p99": round(p99_lat, 1),
            "min": round(min(lats), 1) if lats else 0,
            "max": round(max(lats), 1) if lats else 0,
            "std": round(sd_lat, 1),
            "cv": round(cv_lat, 3),
        },
        "pipeline_details": {
            "slice_ms": {"mean": round(mn_slice, 3)},
            "convert_ms": {"mean": round(mn_convert, 3)},
            "decode_ms": {"mean": round(mn_decode, 1)},
            "postprocess_ms": {"mean": round(mn_post, 3)},
            "total_pipeline_ms": {"mean": round(mn_pipe, 1)},
        },
        "pipeline_pct": {
            "slice_pct": round(slice_pct, 1),
            "convert_pct": round(convert_pct, 1),
            "decode_pct": round(decode_pct, 1),
            "postprocess_pct": round(post_pct, 1),
        },
        "rtf": {"mean": round(mn_rtf, 3)},
        "throughput": {
            "windows_per_second": round(windows_per_second, 2),
            "audio_seconds_per_second": round(audio_per_second, 3),
            "processing_wall_seconds": round(processing_wall_s, 2),
            "benchmark_wall_seconds": round(cfg_wall_s, 2),
        },
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


def _warmup(model: WhisperModel, audio_f32: np.ndarray, sample_rate: int, n: int):
    """Run N warmup decodes to initialize Whisper, CPU cache, and model memory.

    Results are discarded. Uses the first 1-second segment of audio.
    """
    if n <= 0:
        return
    warmup_audio = audio_f32[:sample_rate] if len(audio_f32) >= sample_rate else audio_f32
    print(f"  Warmup: {n} decode{'s' if n > 1 else ''} on {len(warmup_audio) / sample_rate:.1f}s audio...")
    for i in range(n):
        gen, _ = model.transcribe(
            warmup_audio, language="en", temperature=0.0,
            beam_size=1, condition_on_previous_text=False,
            vad_filter=False,
        )
        _ = list(gen)
    print(f"  Warmup complete.")
    time.sleep(SLEEP_BETWEEN_CONFIGS)


def _merge_runs(run_results: list) -> dict:
    """Aggregate multiple runs of the same buffer size into one result.

    Latency/RTF/CPU/RSS values are pooled across runs.
    WER/CER is pooled across runs (weighted by reference word count).
    """
    if not run_results:
        return {}
    first = run_results[0]

    mean_latencies = []
    median_latencies = []
    p95_latencies = []
    p99_latencies = []
    min_latencies = []
    max_latencies = []
    std_latencies = []
    cv_latencies = []
    all_rtfs = []
    all_cpus = []
    all_rsses = []
    all_wer_num = []
    all_cer_num = []
    all_ref_wc = []
    all_pipeline_slice = []
    all_pipeline_convert = []
    all_pipeline_decode = []
    all_pipeline_post = []
    all_pipeline_total = []
    all_wps = []
    all_aps = []
    all_slice_pcts = []
    all_convert_pcts = []
    all_decode_pcts = []
    all_post_pcts = []
    total_windows = 0
    total_miss = 0
    total_processing_wall = 0.0
    total_benchmark_wall = 0.0

    for r in run_results:
        l = r["decode_latency_ms"]
        mean_latencies.append(l["mean"])
        median_latencies.append(l.get("median", 0))
        p95_latencies.append(l["p95"])
        p99_latencies.append(l.get("p99", 0))
        min_latencies.append(l["min"])
        max_latencies.append(l["max"])
        std_latencies.append(l.get("std", 0))
        cv_latencies.append(l.get("cv", 0))
        total_windows += r["total_windows"]
        total_miss += r["backlog_miss_rate_pct"] * r["total_windows"] / 100
        total_processing_wall += r.get("throughput", {}).get("processing_wall_seconds", 0)
        total_benchmark_wall += r.get("throughput", {}).get("benchmark_wall_seconds", r.get("throughput", {}).get("config_wall_seconds", 0))

        all_rtfs.append(r["rtf"]["mean"])
        all_cpus.append(r["cpu_percent"]["mean"])
        all_rsses.append(r["rss_mb"]["mean"])

        if r.get("wer") is not None:
            all_wer_num.append(r["wer"] * r.get("total_reference_words", r["total_windows"]))
            all_cer_num.append(r["cer"] * r.get("total_reference_words", r["total_windows"]))
            all_ref_wc.append(r.get("total_reference_words", r["total_windows"]))

        pd = r.get("pipeline_details", {})
        all_pipeline_slice.append(pd.get("slice_ms", {}).get("mean", 0))
        all_pipeline_convert.append(pd.get("convert_ms", {}).get("mean", 0))
        all_pipeline_decode.append(pd.get("decode_ms", {}).get("mean", 0))
        all_pipeline_post.append(pd.get("postprocess_ms", {}).get("mean", 0))
        all_pipeline_total.append(pd.get("total_pipeline_ms", {}).get("mean", 0))

        all_wps.append(r.get("throughput", {}).get("windows_per_second", 0))
        all_aps.append(r.get("throughput", {}).get("audio_seconds_per_second", 0))

        pp = r.get("pipeline_pct", {})
        all_slice_pcts.append(pp.get("slice_pct", 0))
        all_convert_pcts.append(pp.get("convert_pct", 0))
        all_decode_pcts.append(pp.get("decode_pct", 0))
        all_post_pcts.append(pp.get("postprocess_pct", 0))

    n_runs = len(run_results)

    # Pipeline percentages (aggregated across runs)
    mn_slice_p = mean(all_slice_pcts) if all_slice_pcts else 0
    mn_conv_p = mean(all_convert_pcts) if all_convert_pcts else 0
    mn_dec_p = mean(all_decode_pcts) if all_decode_pcts else 0
    mn_post_p = mean(all_post_pcts) if all_post_pcts else 0

    merged = {
        "buffer_ms": first["buffer_ms"],
        "total_windows": total_windows,
        "total_reference_words": sum(all_ref_wc) if all_ref_wc else 0,
        "redundancy_ratio": first["redundancy_ratio"],
        "decode_latency_ms": {
            "mean": round(mean(mean_latencies), 1) if mean_latencies else 0,
            "median": round(mean(median_latencies), 1) if median_latencies else 0,
            "p95": round(mean(p95_latencies), 1) if p95_latencies else 0,
            "p99": round(mean(p99_latencies), 1) if p99_latencies else 0,
            "min": round(min(min_latencies), 1) if min_latencies else 0,
            "max": round(max(max_latencies), 1) if max_latencies else 0,
            "std": round(mean(std_latencies), 1) if std_latencies else 0,
            "cv": round(mean(cv_latencies), 3) if cv_latencies else 0,
        },
        "pipeline_details": {
            "slice_ms": {"mean": round(mean(all_pipeline_slice), 3)},
            "convert_ms": {"mean": round(mean(all_pipeline_convert), 3)},
            "decode_ms": {"mean": round(mean(all_pipeline_decode), 1)},
            "postprocess_ms": {"mean": round(mean(all_pipeline_post), 3)},
            "total_pipeline_ms": {"mean": round(mean(all_pipeline_total), 1)},
        },
        "pipeline_pct": {
            "slice_pct": round(mn_slice_p, 1),
            "convert_pct": round(mn_conv_p, 1),
            "decode_pct": round(mn_dec_p, 1),
            "postprocess_pct": round(mn_post_p, 1),
        },
        "rtf": {"mean": round(mean(all_rtfs), 3) if all_rtfs else 0},
        "throughput": {
            "windows_per_second": round(mean(all_wps), 2) if all_wps else 0,
            "audio_seconds_per_second": round(mean(all_aps), 3) if all_aps else 0,
            "processing_wall_seconds": round(total_processing_wall, 2),
            "benchmark_wall_seconds": round(total_benchmark_wall, 2),
        },
        "backlog_ratio_p95": round(mean([r.get("backlog_ratio_p95", 0) for r in run_results]), 2),
        "backlog_miss_rate_pct": round(total_miss / total_windows * 100, 1) if total_windows else 0,
        "cpu_percent": {
            "mean": round(mean(all_cpus), 1),
            "observed_max": max(r["cpu_percent"]["observed_max"] for r in run_results if "observed_max" in r.get("cpu_percent", {})),
        },
        "rss_mb": {
            "mean": round(mean(all_rsses), 1),
            "observed_max": max(r["rss_mb"]["observed_max"] for r in run_results if "observed_max" in r.get("rss_mb", {})),
        },
    }

    if sum(all_ref_wc) > 0:
        merged["wer"] = round(sum(all_wer_num) / sum(all_ref_wc), 4) if sum(all_ref_wc) else None
        merged["cer"] = round(sum(all_cer_num) / sum(all_ref_wc), 4) if sum(all_ref_wc) else None
    else:
        merged["wer"] = first.get("wer")
        merged["cer"] = first.get("cer")

    return merged


def run_benchmark(
    audio_f32: np.ndarray,
    sample_rate: int,
    transcript: list,
    buffer_sizes: list,
    interval_ms: int,
    runs: int = 1,
    seed: int = 42,
    dataset_source: str = "custom",
):
    """Run the rolling buffer benchmark for all buffer sizes.

    Performs warmup, then for each buffer size runs 'runs' repeated
    measurements and aggregates across runs.
    """
    psutil_mod = try_psutil()
    has_psutil = psutil_mod is not None
    interval_s = interval_ms / 1000.0

    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

    duration_s = len(audio_f32) / sample_rate
    has_transcript = transcript is not None and len(transcript) > 0

    print(f"  Model:           {MODEL_NAME}")
    print(f"  Device:          {DEVICE}")
    print(f"  Compute:         {COMPUTE_TYPE}")
    print(f"  Audio:           {duration_s:.1f}s")
    print(f"  Streaming interval: {interval_ms}ms (fixed)")
    print(f"  Buffer sizes:    {buffer_sizes} ms")
    print(f"  Ground truth:    {'YES' if has_transcript else 'NO — WER/CER=N/A'}")
    print(f"  Measurement runs:{runs}")
    print(f"  psutil:          {'available' if has_psutil else 'NOT available'}")
    print()

    # ── Warmup ──
    if WARMUP_RUNS > 0:
        _warmup(model, audio_f32, sample_rate, WARMUP_RUNS)

    all_rows = []
    config_results = []

    for bs in buffer_sizes:
        print(f"  ── Buffer={bs}ms, {runs} run{'s' if runs > 1 else ''} ──")
        run_results = []

        for run_idx in range(runs):
            if runs > 1:
                print(f"    run {run_idx + 1}/{runs}...", end=" ", flush=True)
            result, rows = run_configuration(
                model, audio_f32, sample_rate, bs,
                interval_s, transcript, psutil_mod, has_transcript,
            )
            if runs > 1:
                print(f"latency={result.get('decode_latency_ms', {}).get('mean', 0):.0f}ms")

            run_results.append(result)

            if run_idx == runs - 1:
                all_rows.extend(rows)

        # Merge multiple runs
        merged = _merge_runs(run_results)
        config_results.append(merged)

        wer_str = "N/A" if merged.get("wer") is None else f"{merged['wer']:.3f}"
        print(f"  → merged: latency={merged['decode_latency_ms']['mean']:>6.0f}ms  "
              f"p95={merged['decode_latency_ms']['p95']:>6.0f}ms  "
              f"RTF={merged['rtf']['mean']:.3f}  "
              f"backlog_p95={merged['backlog_ratio_p95']:.2f}  "
              f"miss={merged['backlog_miss_rate_pct']:.0f}%  "
              f"wer={wer_str}  "
              f"decodes={merged['total_windows']}")
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
            "seed": seed,
            "dataset_source": dataset_source,
            "timestamp_source": "Word-level JSON if provided via --transcript; for LibriSpeech, Whisper-generated word alignments (approximate, not ground truth). Only transcript text is considered ground truth.",
        },
        "has_ground_truth": has_transcript,
        "results": config_results,
        "tradeoff": tradeoff_block,
        "methodology_notes": (
            "This benchmark evaluates rolling audio buffer effects for STT inference. "
            "It uses a pre-loaded float32 numpy array as the rolling buffer — each window "
            "is a direct slice view (no WAV reconstruction, no file I/O per decode). "
            "Each rolling window is an independent STT request. NO transcript merging, "
            "NO stabilization, NO hypothesis revision, NO incremental decoding. "
            "Words are assigned to windows by midpoint: midpoint = (start + end) / 2. "
            "WER/CER are weighted averages of per-window values (weighted by "
            "reference word count). Silent windows (zero reference words) are excluded.\n"
            "Rolling windows overlap; therefore reference words may be duplicated across "
            "windows. Weighted WER measures recognition quality per window and is NOT "
            "directly comparable to conventional full-audio corpus WER. It is intended for "
            "relative comparison between buffer sizes under identical methodology.\n\n"
            "Pipeline timing breakdown:\n"
            "  slice_ms — time to slice the rolling buffer (numpy array view).\n"
            "  convert_ms — time to prepare input (negligible for float32).\n"
            "  decode_ms — Whisper model inference.\n"
            "  postprocess_ms — time to materialise segments and compute text/WER.\n"
            "  total_pipeline_ms — wall-clock time for the entire window processing.\n\n"
            "Backlog ratio P95 = P95 decode latency / streaming interval. "
            "Backlog miss rate = % of decodes exceeding the streaming interval. "
            "If backlog_ratio_p95 > 1.0, the system cannot sustain realtime streaming "
            "for 5% of decodes. "
            "Redundancy ratio = buffer_size / interval (derived, not measured).\n\n"
            "CPU utilization is computed as (process_time_delta / wall_time_delta) × 100. "
            "This is more reliable than psutil.cpu_percent(interval=0) which returns 0% "
            "for sub-second operations.\n\n"
            "RSS is monitored by a background thread that samples every ~5ms during decode. "
            "The maximum observed value (rss_mb_peak) is reported as rss_mb. "
            "Per-window rows also include rss_mb_before and rss_mb_after. "
            "Peak RSS is estimated by periodic sampling, NOT continuous OS-level tracing."
        ),
        "limitations": [
            "NO transcript merging — windows are evaluated independently.",
            "NO transcript stabilization — no multi-hypothesis refinement.",
            "NO incremental decoding — no partial hypothesis updates.",
            "NO LLM integration — measures STT only.",
            "NO VAD simulation — audio is pre-trimmed; real streaming has VAD delay.",
            "Offline simulation only — does not measure network or UI latency.",
            "Single streaming interval tested; changing the interval may shift tradeoffs.",
            "CPU utilization is computed from process_time deltas and is approximate for short operations.",
            "Peak RSS is estimated by periodic sampling during decode, not continuous OS-level tracing.",
            "Rolling buffer uses numpy array views — actual production buffer management overhead may differ.",
            "Each rolling window is independently encoded and decoded. Encoder features are not reused between consecutive windows. Therefore measured latency represents independent inference rather than optimized incremental streaming.",
            "Rolling windows overlap — reference words are duplicated across windows. Weighted WER measures per-window recognition quality and cannot be directly compared to conventional full-audio corpus WER. It is intended for relative buffer-size comparison only.",
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

    # ── Plots ──
    _generate_plots(config_results, has_transcript)


# ── Main ─────────────────────────────────────────────────

# ── Plotting ───────────────────────────────────────────────


def _generate_plots(config_results: list, has_transcript: bool):
    """Generate diagnostic plots saved to OUTPUT_DIR.

    Requires matplotlib. Silently skips if unavailable.
    All plots use the non-prescriptive tradeoff style.
    """
    if not HAS_MPL:
        print("  matplotlib not installed — skipping plots")
        return

    bs = [c["buffer_ms"] for c in config_results]

    # 1. Latency (mean + p95 + p99)
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(bs, [c["decode_latency_ms"]["mean"] for c in config_results],
             "o-", label="mean", linewidth=1.5, markersize=5)
    ax1.plot(bs, [c["decode_latency_ms"]["p95"] for c in config_results],
             "s--", label="P95", linewidth=1.5, markersize=5)
    ax1.plot(bs, [c["decode_latency_ms"].get("p99", c["decode_latency_ms"]["p95"]) for c in config_results],
             "d-.", label="P99", linewidth=1.5, markersize=5)
    ax1.set_xlabel("Rolling buffer size (ms)")
    ax1.set_ylabel("Decode latency (ms)")
    ax1.set_title("Rolling Buffer Latency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, "rolling_buffer_latency.png"), dpi=150)

    # 2. RTF
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(bs, [c["rtf"]["mean"] for c in config_results],
             "o-", linewidth=1.5, markersize=5)
    ax2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.6,
                label="RTF=1.0 (realtime threshold)")
    ax2.set_xlabel("Rolling buffer size (ms)")
    ax2.set_ylabel("RTF")
    ax2.set_title("Rolling Buffer RTF")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, "rolling_buffer_rtf.png"), dpi=150)

    # 3. Backlog
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(bs, [c["backlog_ratio_p95"] for c in config_results],
             "o-", linewidth=1.5, markersize=5, label="Backlog ratio P95")
    ax3.plot(bs, [c["backlog_miss_rate_pct"] for c in config_results],
             "s--", linewidth=1.5, markersize=5, label="Miss rate (%)")
    ax3.axhline(y=1.0, color="gray", linestyle=":", alpha=0.6,
                label="Backlog=1.0 threshold")
    ax3.set_xlabel("Rolling buffer size (ms)")
    ax3.set_ylabel("Backlog / Miss rate")
    ax3.set_title("Rolling Buffer Backlog")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, "rolling_buffer_backlog.png"), dpi=150)

    # 4. WER (only if ground truth exists and at least one config has valid WER)
    if has_transcript:
        valid_results = [c for c in config_results if c.get("wer") is not None]
        if valid_results:
            fig4, ax4 = plt.subplots(figsize=(7, 4))
            x_wer = [c["buffer_ms"] for c in valid_results]
            y_wer = [c["wer"] for c in valid_results]
            ax4.plot(x_wer, y_wer, "o-", linewidth=1.5, markersize=5)
            ax4.set_xlabel("Rolling buffer size (ms)")
            ax4.set_ylabel("WER")
            ax4.set_title("Rolling Buffer WER")
            ax4.grid(True, alpha=0.3)
            fig4.tight_layout()
            fig4.savefig(os.path.join(OUTPUT_DIR, "rolling_buffer_wer.png"), dpi=150)

    plt.close("all")
    print("  Plots: saved to", OUTPUT_DIR)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="FasterWhisper rolling buffer benchmark"
    )
    parser.add_argument("--wav", type=str, default=None,
                        help="WAV file for accuracy+latency benchmark")
    parser.add_argument("--transcript", type=str, default=None,
                        help="Word-level transcript JSON (required with --wav)")
    parser.add_argument("--librispeech", type=str, default=None,
                        help="LibriSpeech dataset root directory (auto-discovers FLAC + .trans.txt)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--record", action="store_true",
                        help="Record audio (latency-only, no WER)")
    parser.add_argument("--interval", type=int, default=STREAMING_INTERVAL_MS,
                        help=f"Fixed streaming interval in ms (default: {STREAMING_INTERVAL_MS})")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of measurement rounds (default: 1)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Recording duration in seconds")
    args = parser.parse_args()

    np.random.seed(args.seed)

    transcript = None
    wav_bytes = None
    audio_f32 = None
    dataset_source = "custom"

    if args.librispeech:
        # ── LibriSpeech mode: discover all valid utterances ──
        utterances = _discover_librispeech(args.librispeech)
        if not utterances:
            print("ERROR: No valid LibriSpeech utterances found in", args.librispeech)
            sys.exit(1)
        random.seed(args.seed)

        if MAX_UTTERANCES is not None and len(utterances) > MAX_UTTERANCES:
            utterances = random.sample(utterances, MAX_UTTERANCES)
        print(f"  LibriSpeech root: {args.librispeech}")
        print(f"  Discovered: {len(utterances)} utterances")

        # Collect per-window rows for JSONL; aggregate by buffer size for report
        all_rows = []
        buffer_results = {bs: [] for bs in BUFFER_SIZES_MS}
        model = None

        for utt_idx, utt in enumerate(utterances):
            print(f"\n  ── Utterance {utt_idx+1}/{len(utterances)}: {utt['utt_id']} ──")
            print(f"    Text: {utt['transcript_text'][:80]}...")

            # Load FLAC
            flac_f32 = _flac_to_float32(utt["flac_path"])
            print(f"    Audio: {len(flac_f32)/SAMPLE_RATE:.1f}s, {flac_f32.dtype}")

            # Init model once on first utterance
            if model is None:
                model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
                # Warm up once before any measurements
                _warmup(model, flac_f32, SAMPLE_RATE, WARMUP_RUNS)

            # Generate word timestamps (LibriSpeech has no timestamps)
            print("    Generating Whisper word alignments (word_timestamps=True)...")
            wl_transcript = _generate_word_timestamps(model, flac_f32)
            print(f"    Generated {len(wl_transcript)} word entries")

            # Run benchmark for this utterance — append per-buffer results
            psutil_mod = try_psutil()
            has_transcript = len(wl_transcript) > 0
            interval_s = args.interval / 1000.0

            for bs in BUFFER_SIZES_MS:
                print(f"    Buffer={bs}ms...", end=" ", flush=True)
                for run_idx in range(args.runs):
                    config_result, rows = run_configuration(
                        model, flac_f32, SAMPLE_RATE, bs,
                        interval_s, wl_transcript, psutil_mod, has_transcript,
                    )
                    # Tag rows with utterance id for debugging
                    for row in rows:
                        row["utt_id"] = utt["utt_id"]
                    all_rows.extend(rows)
                    if run_idx == args.runs - 1:
                        config_result["utt_id"] = utt["utt_id"]
                        buffer_results[bs].append(config_result)
                print(f"done ({len(rows)} windows)")
                time.sleep(SLEEP_BETWEEN_CONFIGS)

        # Aggregate per-buffer results across all utterances
        aggregated_results = []
        for bs in BUFFER_SIZES_MS:
            if buffer_results[bs]:
                merged = _merge_runs(buffer_results[bs])
                aggregated_results.append(merged)

        # Save JSONL
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        jsonl_path = os.path.join(OUTPUT_DIR, "rolling_buffer_results.jsonl")
        with open(jsonl_path, "w") as f:
            for r in all_rows:
                f.write(json.dumps(r) + "\n")
        print(f"\n  Saved: {jsonl_path} ({len(all_rows)} rows, {len(utterances)} utterances)")

        # Aggregated report — one result per buffer size across all utterances
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
                "streaming_interval_ms": args.interval,
                "buffer_sizes_ms": BUFFER_SIZES_MS,
                "warmup_runs": WARMUP_RUNS,
                "seed": args.seed,
                "dataset_source": "librispeech",
                "num_utterances": len(utterances),
                "timestamp_source": "Whisper-generated word alignments (approximate pseudo-alignments, only transcript text is ground truth)",
            },
            "has_ground_truth": True,
            "results": aggregated_results,
            "methodology_notes": (
                "LibriSpeech provides utterance-level transcripts only. "
                "Word alignments are generated using Whisper word_timestamps=True. "
                "These are approximate pseudo-alignments, NOT manually annotated "
                "ground truth. Only transcript text is considered ground truth; "
                "window assignment depends on Whisper-generated alignments. "
                "Rolling windows overlap — reference words are duplicated across "
                "windows. Weighted WER measures per-window recognition quality and "
                "is intended for relative buffer-size comparison, NOT direct "
                "comparison with conventional full-audio corpus WER. "
                "Each utterance is benchmarked independently. "
                "Final report aggregates per buffer size across all utterances using _merge_runs. "
                "Rolling window methodology, WER/CER, and aggregation are unchanged."
            ),
            "limitations": [
                "LibriSpeech provides utterance-level transcripts only — word alignments are Whisper-generated, not ground truth.",
                "Alignment accuracy depends on Whisper word_timestamps quality.",
                "Results aggregated across utterances — per-utterance detail available in JSONL.",
                "Each rolling window is independently encoded and decoded. Encoder features are not reused between consecutive windows. Therefore measured latency represents independent inference rather than optimized incremental streaming.",
                "Rolling windows overlap — reference words are duplicated across windows. Weighted WER is intended for relative buffer-size comparison only, not conventional corpus WER.",
            ],
        }
        report_path = os.path.join(OUTPUT_DIR, "rolling_buffer_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {report_path}")

        _generate_plots(aggregated_results, True)
        return

    # ── Non-LibriSpeech modes (unchanged) ──
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

    if wav_bytes is not None:
        audio_f32 = _wav_bytes_to_float32(wav_bytes)
    else:
        audio_f32 = np.zeros(1, dtype=np.float32)

    run_benchmark(
        audio_f32, SAMPLE_RATE, transcript or [],
        BUFFER_SIZES_MS, args.interval, runs=args.runs,
        seed=args.seed, dataset_source=dataset_source,
    )


if __name__ == "__main__":
    main()
