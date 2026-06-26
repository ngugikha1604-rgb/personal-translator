"""report_stt_breakdown.py — Summarize STT breakdown results as a bottleneck report.

Reads benchmark_results/stt_breakdown.jsonl
Produces benchmark_results/stt_breakdown_report.json

Usage:
    cd backend
    python benchmark/pipeline/report_stt_breakdown.py

Output:
    benchmark_results/stt_breakdown_report.json
"""

import json
import os
import sys
from statistics import mean, median

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

INPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmark_results", "stt_breakdown.jsonl"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmark_results", "stt_breakdown_report.json"
)


def compute_stats(values: list) -> dict:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return {"mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0}
    return {
        "mean": round(mean(s), 1),
        "median": round(median(s), 1),
        "p95": round(s[int(n * 0.95)], 1),
        "min": round(min(s), 1),
        "max": round(max(s), 1),
    }


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Run stt_breakdown.py first.")
        print(f"  Missing: {INPUT_PATH}")
        sys.exit(1)

    with open(INPUT_PATH) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    ok_rows = [r for r in rows if "error" not in r]
    total = len(rows)
    ok = len(ok_rows)
    failed = total - ok

    stages = ["audio_load_ms", "audio_prepare_ms", "api_request_ms",
              "response_parse_ms", "total_ms"]

    # ── Per-stage stats ──
    stage_stats = {}
    for stage in stages:
        vals = [float(r[stage]) for r in ok_rows]
        stage_stats[stage] = compute_stats(vals)

    # ── Stage percentages (relative to each row's total) ──
    stage_pcts = {}
    for stage in stages:
        if stage == "total_ms":
            continue
        ratios = [
            float(r[stage]) / float(r["total_ms"]) * 100
            for r in ok_rows if float(r["total_ms"]) > 0
        ]
        stage_pcts[stage] = compute_stats(ratios)

    # ── Dominant bottleneck analysis ──
    # For each row, identify which stage consumed the most latency
    dominant_counts = {}
    for stage in stages:
        if stage == "total_ms":
            continue
        dominant_counts[stage] = 0

    for r in ok_rows:
        best_stage = max(
            (s for s in stages if s != "total_ms"),
            key=lambda s: float(r[s]),
        )
        if best_stage in dominant_counts:
            dominant_counts[best_stage] += 1

    dominant_pcts = {
        k: round(v / len(ok_rows) * 100, 1)
        for k, v in sorted(dominant_counts.items(), key=lambda x: -x[1])
    }

    # ── Build report ──
    report = {
        "source": "stt_breakdown.jsonl",
        "total_runs": total,
        "successful_runs": ok,
        "failed_runs": failed,
        "stage_timing_ms": stage_stats,
        "stage_percentage_of_total": stage_pcts,
        "dominant_bottleneck": {
            "stage": list(dominant_pcts.keys())[0] if dominant_pcts else None,
            "pct_of_runs": list(dominant_pcts.values())[0] if dominant_pcts else 0,
            "all_stages": dominant_pcts,
        },
        "bottleneck_summary": (
            f"Stage '{list(dominant_pcts.keys())[0]}' is the dominant bottleneck "
            f"in {list(dominant_pcts.values())[0]}% of runs"
        ) if dominant_pcts else "No data",
        "recommendation": (
            "Focus optimization on the stage with the highest mean timing "
            "and highest dominance percentage."
        ),
    }

    # Write
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    # Print
    print(f"\n{'=' * 60}")
    print(f"  STT Breakdown Report")
    print(f"{'=' * 60}")
    print(f"  Runs: {ok} successful, {failed} failed")

    print(f"\n  — Stage timing (ms) —")
    print(f"  {'Stage':25s} {'mean':>8s} {'median':>8s} {'p95':>8s} {'min':>8s} {'max':>8s}")
    for stage in stages:
        s = stage_stats[stage]
        print(f"  {stage:25s} {s['mean']:>8.1f} {s['median']:>8.1f} {s['p95']:>8.1f} {s['min']:>8.1f} {s['max']:>8.1f}")

    print(f"\n  — Stage percentage of total —")
    print(f"  {'Stage':25s} {'mean':>8s} {'median':>8s} {'p95':>8s}")
    for stage in stages:
        if stage == "total_ms":
            continue
        p = stage_pcts[stage]
        print(f"  {stage:25s} {p['mean']:>7.1f}% {p['median']:>7.1f}% {p['p95']:>7.1f}%")

    print(f"\n  — Dominant bottleneck distribution —")
    for stage, pct in dominant_pcts.items():
        bar = "█" * max(1, int(pct / 5))
        print(f"  {bar} {stage:20s} {pct:.0f}%")

    print(f"\n  Report saved: {OUTPUT_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
