# STT Pipeline Benchmark Findings — Production Reference

**Last updated:** from benchmark runs on `tiny.en` / CPU / int8 (Faster Whisper, `cpu_threads=4`)
**Purpose:** Single source of truth for what's been measured, what's reliable, and what to fix before shipping the glasses HUD pipeline. Read this before touching STT/streaming config again.

---

## 1. Core inference speed (single-shot, no streaming)

**Source:** `stt_breakdown.py` on 50 LibriSpeech utterances, 3 runs each (150 total), with warmup.

| Metric | Run A | Run B | Note |
|---|---|---|---|
| Mean | 920.9 ms | 665.4 ms | Same config, ~28% gap — attributed to system noise/background load between runs, not a code bug (verified: sampling is deterministic with seed=42, code unchanged) |
| Median | 920.2 ms | 635.3 ms | |
| P95 | 1218.2 ms | 978.2 ms | |
| Min | 371.1 ms | 354.8 ms | Shortest utterance (~2 words) |
| Max | 2463.1 ms | 1971.5 ms | Longest utterance (~77 words) |
| Transcript length | mean 19 words | — | |

**Takeaway:** treat **~650–920ms mean, ~980–1220ms P95** as the realistic range for a single tiny.en/CPU/int8 decode on LibriSpeech-length utterances. Don't trust a single run's number as ground truth — rerun 2-3x and look at the range, this machine has real run-to-run variance.

**Stage breakdown:** `audio_load_ms`, `audio_prepare_ms`, `response_parse_ms` are all **0ms** — 100% of latency is model inference (`api_request_ms` / `inference_ms`, same value, kept as two field names for backward compatibility). **Nothing to optimize outside the model itself** — no point tuning I/O or audio prep code.

---

## 2. Internal stage probe (encoder vs decoder split)

**Source:** `stt_internal_stage_probe` (fixed short audio clip, not LibriSpeech corpus).

| Stage | Time | % of total |
|---|---|---|
| Encoder | 382.6 ms | 64% |
| Decoder | 214.0 ms | 36% |
| **Total** | **612.5 ms** | — |

Additive check passed (610.2ms measured vs 612.5ms sum — within 0.5% error). CV low (0.05–0.11), this number is trustworthy for its specific short-clip test case. Encoder dominates — if optimizing the model itself is ever on the table, encoder is the bigger lever.

---

## 3. Thread scaling

**Source:** `stt_thread_scaling`.

- **`cpu_threads=4` is optimal**: 617ms mean, RTF 0.123.
- 4→8 threads regresses — consistent with this machine having **4 physical cores**. Don't increase thread count expecting a speedup; already at the ceiling.
- **Current production setting (`cpu_threads=4`) is correct — no action needed.**

---

## 4. Streaming chunk size (static, non-rolling test)

**Source:** `streaming_chunk_size` benchmark — tests fixed chunk durations independently, not the live rolling-buffer scenario.

| Chunk size | first_result_delay | RTF |
|---|---|---|
| 250ms | 997ms | 2.99 (overloaded — can't keep up) |
| 4000ms | 4847ms | 0.212 (good throughput, bad latency) |

**Takeaway:** there is no single "good" fixed chunk size — it's a hard tradeoff between responsiveness and CPU headroom. This is exactly the problem VAD-based dynamic chunking (P1 roadmap item) is meant to solve: cut on speech boundaries instead of a fixed timer, so short utterances resolve fast and long ones don't starve the CPU.

---

## 5. Rolling buffer / backlog analysis (the real production bottleneck)

**Source:** `rolling_buffer_benchmark` (25,824 windows) + custom `backlog_diagnostic.py` analysis on top of it. Streaming interval tested was fixed at **500ms**.

### 5.1 Numbers that matter (large-sample buckets only — ignore any bucket with n<100, those are noise)

| Buffer size | Samples (n) | decode_mean | Miss rate (decode > 500ms interval) |
|---|---|---|---|
| 500ms | 4,028 | 738 ms | **71.9%** |
| 1000ms | 3,724 | 726 ms | **71.3%** |
| 1500ms | 3,420 | 745 ms | **73.9%** |
| 2000ms | 3,116 | 753 ms | **76.6%** |
| 3000ms | 2,534 | 785 ms | **82.6%** |
| 4000ms | 2,039 | 787 ms | **83.4%** |

**This confirms the original concern from `v_report`/`rolling_buffer_report`: the system is not real-time capable at a 500ms streaming interval, at ANY buffer size.** Even the smallest realistic buffer (500ms) already has decode_mean (738ms) exceeding the interval by ~1.5x.

### 5.2 Root cause — structural, not a bug

- decode_mean is essentially **flat around 700–790ms regardless of buffer size** (500ms through 4000ms). This means the bottleneck is the **fixed per-decode inference cost**, not the audio duration being decoded — consistent with Section 1's single-shot numbers.
- **Needed interval to reach ~0% miss (P95-based):** roughly **1200–1800ms**, not 500ms, across all buffer sizes tested.
- ⚠️ A naive automated read of the raw data (971 micro-buckets, including tiny buffers like 4ms with n=6) will misleadingly suggest "small buffers keep up fine" — that's an artifact of near-empty startup buffers, not a real operating point. **Always filter to buckets with large sample counts (n>100) before drawing conclusions from this dataset.**

### 5.3 Worst-case utterances

Top 10 worst utterances all show **100% miss rate** regardless of buffer size, with decode_mean 1088–1254ms across 42–144 decode attempts each. These are not one-off outliers — they're consistently slow regardless of configuration, meaning **no buffer/interval tuning alone fixes them**. Worth inspecting later whether they share traits (longer speech, faster speaking rate, more complex phonetics) once VAD chunking is in place.

---

## 6. Model comparison (70B vs 8B) — NOT reliable, don't use

`model_comparison/report.json`: 70B parse rate was 23.2% (13 samples) vs 8B's 100% (56 samples). Sample sizes are too imbalanced to trust the latency comparison. **Irrelevant going forward anyway** — plan is to switch to a local model requiring on-device testing once hardware is available. Ignore this report for future decisions.

---

## 7. Verification benchmark — broken, unusable

`v_report.json`: `parse_ok: 0` across all 240 turns, both baseline and candidate. No usable data. Still on the backlog to debug (P1) if verification quality benchmarking becomes a priority again.

---

## 8. What to actually change in production (priority order)

1. **Increase the streaming interval.** 500ms is not achievable with tiny.en/CPU/int8 — realistic floor is ~750-800ms (mean-based) to ~1200-1800ms (P95-safe). This is the single highest-leverage, lowest-effort fix.
2. **Implement VAD-based dynamic chunking (P1, already on roadmap).** Confirmed by data: chunk size barely affects decode time (flat ~700-790ms across 500ms-4000ms buffers), so the win isn't from buffer tuning — it's from cutting on actual speech boundaries so short utterances resolve fast without waiting on a fixed timer, and so the system doesn't decode padding/silence.
3. **Do not spend time optimizing audio I/O or parsing code** — confirmed 0ms contribution, that's not where the time goes.
4. **Do not increase `cpu_threads` past 4** — confirmed regression past the physical core count.
5. **Treat single benchmark runs with suspicion.** This machine has ~20-30% run-to-run variance on identical configs (Section 1). Always run 2-3x before trusting a number, especially before/after a config change.
6. **Revisit worst-case utterances** once VAD chunking lands, to see if the 100%-miss cases persist — if they do, that's a sign tiny.en itself (not the buffering strategy) is the ceiling, and local model upgrade (once hardware allows) becomes the next lever.

---

## 9. Glossary / field names (avoid confusion later)

- `api_request_ms` / `inference_ms` — same value, two names. Historic name (`api_request_ms`) kept for backward compatibility with old Groq-cloud-era reports; `inference_ms` is the accurate name for the current local Faster Whisper provider.
- `decode_ms` (rolling buffer benchmark) — time to decode one rolling-buffer window; comparable in meaning to `inference_ms` above.
- `backlog_miss_rate` / `miss_pct_vs_500ms` — % of decode windows where `decode_ms` exceeded the fixed streaming interval (500ms in the tested config). High = system falls behind real-time.
- `ms_per_audio_ms` — decode time divided by buffer audio duration. >1.0 means slower than real-time for that buffer size.
