"""backlog_diagnostic.py — Diagnose root cause of high backlog_miss_rate.

Reads the EXISTING rolling_buffer_results.jsonl (already generated) instead of
re-running the benchmark. Answers:

  1. Is decode_ms scaling proportionally with buffer_ms (expected), or is
     something else inflating latency (e.g. RSS growth, thermal throttling
     over the course of the run)?
  2. For each buffer size, what streaming interval WOULD be needed to hit
     0% backlog miss? (i.e. is 500ms simply too aggressive for CPU tiny.en,
     independent of any bug?)
  3. Does miss rate correlate with WHEN in the run the window occurred
     (early vs late) -> would suggest thermal/contention drift, not a
     structural issue.
  4. Does miss rate correlate with which utterance / audio characteristics?

Usage:
    cd backend
    python benchmark/streaming/backlog_diagnostic.py benchmark_results/rolling_buffer_results.jsonl
"""

import json
import sys
from collections import defaultdict
from statistics import mean, median

try:
    import numpy as np
    def percentile(vals, p):
        return float(np.percentile(vals, p))
except ImportError:
    def percentile(vals, p):
        s = sorted(vals)
        k = (len(s) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(s) - 1)
        if f == c:
            return s[f]
        return s[f] + (s[c] - s[f]) * (k - f)

STREAMING_INTERVAL_MS = 500.0  # fixed interval used by rolling_buffer_benchmark.py


def load_rows(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main():
    if len(sys.argv) < 2:
        print("Usage: python backlog_diagnostic.py <rolling_buffer_results.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading {path} ...")
    rows = load_rows(path)
    print(f"Loaded {len(rows)} per-window rows.\n")

    # ── 1. Group by buffer_ms ──
    by_buffer = defaultdict(list)
    for r in rows:
        by_buffer[r["buffer_ms"]].append(r)

    print("=" * 100)
    print("1) DECODE TIME vs BUFFER SIZE — is scaling proportional (expected) or worse?")
    print("=" * 100)
    print(f"{'buffer_ms':>10} {'n':>6} {'decode_mean':>12} {'decode_p95':>12} "
          f"{'ms_per_audio_ms':>16} {'miss_vs_500ms':>14} {'needed_interval_p95':>20}")

    buffer_summary = []
    for bs in sorted(by_buffer.keys()):
        group = by_buffer[bs]
        decodes = [r["decode_ms"] for r in group if "decode_ms" in r]
        if not decodes:
            continue
        d_mean = mean(decodes)
        d_p95 = percentile(decodes, 95)
        miss = sum(1 for d in decodes if d > STREAMING_INTERVAL_MS) / len(decodes) * 100
        ms_per_audio_ms = d_mean / bs  # >1.0 means slower than realtime for this buffer
        needed_interval = d_p95  # interval must be >= p95 decode time to avoid backlog at p95
        buffer_summary.append({
            "buffer_ms": bs, "n": len(decodes), "decode_mean": d_mean,
            "decode_p95": d_p95, "ms_per_audio_ms": ms_per_audio_ms,
            "miss_pct": miss, "needed_interval_p95": needed_interval,
        })
        print(f"{bs:>10} {len(decodes):>6} {d_mean:>12.1f} {d_p95:>12.1f} "
              f"{ms_per_audio_ms:>16.3f} {miss:>13.1f}% {needed_interval:>20.1f}")

    print()
    print("Interpretation:")
    print("  ms_per_audio_ms < 1.0  -> decode is FASTER than the audio it processes (can keep up)")
    print("  ms_per_audio_ms > 1.0  -> decode is SLOWER than realtime for that buffer size (cannot keep up)")
    print("  needed_interval_p95    -> streaming interval that would give ~0% miss at P95 for this buffer")
    print()

    # ── 2. Is there drift over time within a run (thermal / contention)? ──
    print("=" * 100)
    print("2) DECODE TIME DRIFT OVER TIME — early vs late windows in the run")
    print("=" * 100)
    for bs in sorted(by_buffer.keys()):
        group = sorted(by_buffer[bs], key=lambda r: r.get("window_index", 0))
        decodes = [r["decode_ms"] for r in group if "decode_ms" in r]
        if len(decodes) < 10:
            continue
        n = len(decodes)
        first_third = decodes[: n // 3]
        last_third = decodes[-(n // 3):]
        drift_pct = (mean(last_third) - mean(first_third)) / mean(first_third) * 100 if mean(first_third) else 0
        flag = "  <-- POSSIBLE DRIFT" if abs(drift_pct) > 15 else ""
        print(f"  buffer={bs:>5}ms  first_third_mean={mean(first_third):>8.1f}ms  "
              f"last_third_mean={mean(last_third):>8.1f}ms  drift={drift_pct:>+6.1f}%{flag}")
    print()

    # ── 3. Miss rate by utterance (if utt_id present) — flags specific bad utterances ──
    if rows and "utt_id" in rows[0]:
        print("=" * 100)
        print("3) MISS RATE BY UTTERANCE — top 10 worst utterances (any buffer size)")
        print("=" * 100)
        by_utt = defaultdict(list)
        for r in rows:
            if "decode_ms" in r and "utt_id" in r:
                by_utt[r["utt_id"]].append(r["decode_ms"])
        utt_miss = []
        for utt_id, decodes in by_utt.items():
            miss = sum(1 for d in decodes if d > STREAMING_INTERVAL_MS) / len(decodes) * 100
            utt_miss.append((utt_id, miss, mean(decodes), len(decodes)))
        utt_miss.sort(key=lambda x: -x[1])
        for utt_id, miss, d_mean, n in utt_miss[:10]:
            print(f"  {utt_id:<30} miss={miss:>6.1f}%  decode_mean={d_mean:>8.1f}ms  n={n}")
        print()

    # ── 4. Overall root-cause verdict ──
    print("=" * 100)
    print("4) ROOT CAUSE VERDICT")
    print("=" * 100)
    small_buf = min(buffer_summary, key=lambda x: x["buffer_ms"])
    large_buf = max(buffer_summary, key=lambda x: x["buffer_ms"])

    if small_buf["ms_per_audio_ms"] > 0.5 and small_buf["miss_pct"] > 50:
        print(f"  -> Even the SMALLEST buffer ({small_buf['buffer_ms']}ms) has {small_buf['miss_pct']:.0f}% miss rate")
        print(f"     with decode_mean={small_buf['decode_mean']:.0f}ms against a fixed 500ms interval.")
        print(f"     This means: the 500ms streaming interval is STRUCTURALLY too aggressive")
        print(f"     for tiny.en/CPU/int8 -- not a bug, a hardware/model-speed ceiling.")
        print(f"     Fix options (not mutually exclusive):")
        print(f"       a) Increase streaming interval (fewer decodes/sec, less redundant work)")
        print(f"       b) Use VAD to only decode on speech end, not fixed intervals")
        print(f"       c) Smaller/faster model or GPU device")
        print(f"       d) Accept partial results are 'best effort', decouple emit rate from decode rate")
    else:
        print(f"  -> Smallest buffer keeps up ({small_buf['miss_pct']:.0f}% miss). Miss rate grows with buffer size.")
        print(f"     This is EXPECTED behavior (larger buffer = more audio to decode per window),")
        print(f"     not a bug. The tradeoff is inherent: bigger buffer = better accuracy/context,")
        print(f"     but slower per-window decode.")

    print()
    print("  Needed interval to reach near-0% miss, by buffer size:")
    for s in buffer_summary:
        print(f"    buffer={s['buffer_ms']:>5}ms  -> interval >= {s['needed_interval_p95']:.0f}ms "
              f"(vs fixed 500ms used in benchmark)")

    # ── Write JSON report alongside the jsonl input ──
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "benchmark_results")
    out_path = os.path.join(out_dir, "backlog_diagnostic_report.json")

    drift_summary = []
    for bs in sorted(by_buffer.keys()):
        group = sorted(by_buffer[bs], key=lambda r: r.get("window_index", 0))
        decodes = [r["decode_ms"] for r in group if "decode_ms" in r]
        if len(decodes) < 10:
            continue
        n = len(decodes)
        first_third = decodes[: n // 3]
        last_third = decodes[-(n // 3):]
        ft_mean = mean(first_third)
        lt_mean = mean(last_third)
        drift_pct = (lt_mean - ft_mean) / ft_mean * 100 if ft_mean else 0
        drift_summary.append({
            "buffer_ms": bs, "first_third_mean_ms": round(ft_mean, 1),
            "last_third_mean_ms": round(lt_mean, 1), "drift_pct": round(drift_pct, 1),
            "possible_drift": abs(drift_pct) > 15,
        })

    worst_utterances = []
    if rows and "utt_id" in rows[0]:
        by_utt = defaultdict(list)
        for r in rows:
            if "decode_ms" in r and "utt_id" in r:
                by_utt[r["utt_id"]].append(r["decode_ms"])
        utt_miss = []
        for utt_id, decodes in by_utt.items():
            miss = sum(1 for d in decodes if d > STREAMING_INTERVAL_MS) / len(decodes) * 100
            utt_miss.append({"utt_id": utt_id, "miss_pct": round(miss, 1),
                              "decode_mean_ms": round(mean(decodes), 1), "n": len(decodes)})
        utt_miss.sort(key=lambda x: -x["miss_pct"])
        worst_utterances = utt_miss[:10]

    is_structural_limit = small_buf["ms_per_audio_ms"] > 0.5 and small_buf["miss_pct"] > 50

    report = {
        "streaming_interval_ms_fixed": STREAMING_INTERVAL_MS,
        "total_rows_analyzed": len(rows),
        "buffer_summary": [
            {
                "buffer_ms": s["buffer_ms"], "n": s["n"],
                "decode_mean_ms": round(s["decode_mean"], 1),
                "decode_p95_ms": round(s["decode_p95"], 1),
                "ms_per_audio_ms": round(s["ms_per_audio_ms"], 3),
                "miss_pct_vs_500ms": round(s["miss_pct"], 1),
                "needed_interval_p95_ms": round(s["needed_interval_p95"], 1),
            }
            for s in buffer_summary
        ],
        "drift_over_time": drift_summary,
        "worst_utterances_top10": worst_utterances,
        "root_cause_verdict": {
            "is_structural_interval_too_aggressive": is_structural_limit,
            "smallest_buffer_ms": small_buf["buffer_ms"],
            "smallest_buffer_miss_pct": round(small_buf["miss_pct"], 1),
            "smallest_buffer_decode_mean_ms": round(small_buf["decode_mean"], 1),
            "explanation": (
                "Even the smallest buffer cannot keep up with a fixed 500ms interval; "
                "this is a structural/hardware ceiling for tiny.en on CPU, not a bug. "
                "Fix options: (a) increase streaming interval, (b) use VAD-based decoding "
                "instead of fixed intervals, (c) faster model/device, (d) decouple emit "
                "rate from decode rate and treat partial results as best-effort."
            ) if is_structural_limit else (
                "Smallest buffer keeps up fine; miss rate grows with buffer size, which is "
                "the expected accuracy-vs-latency tradeoff, not a bug."
            ),
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved JSON report: {out_path}")


if __name__ == "__main__":
    main()
