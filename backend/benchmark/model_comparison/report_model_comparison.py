"""report_model_comparison.py — Compare 70B vs 8B results, produce decision report.

Usage:
    cd backend
    python benchmark/model_comparison/report_model_comparison.py

Output:
    benchmark_results/model_comparison/report.json
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from statistics import mean, median

REPORT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmark_results", "model_comparison"
)


def load(path: str) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def compute_latency_stats(values: list) -> dict:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return {"mean": 0, "median": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
    return {
        "mean": round(mean(s), 1),
        "median": round(median(s), 1),
        "p95": round(s[int(n * 0.95)], 1),
        "p99": round(s[int(n * 0.99)], 1),
        "min": round(min(s), 1),
        "max": round(max(s), 1),
    }


def compare_field(baselines: list, candidates: list, field: str) -> dict:
    """Compare one output field across matched turns."""
    identical = 0
    mismatches = 0
    degraded = []

    # Match by conversation_id + turn_index
    b_map = {(r["conversation_id"], r["turn_index"]): r for r in baselines}
    c_map = {(r["conversation_id"], r["turn_index"]): r for r in candidates}
    common = set(b_map.keys()) & set(c_map.keys())

    for key in common:
        b = b_map[key]
        c = c_map[key]
        if not b.get("parse_ok") or not c.get("parse_ok"):
            continue
        b_val = str(b.get(field, "")).strip()
        c_val = str(c.get(field, "")).strip()

        if b_val == c_val or (not b_val and not c_val):
            identical += 1
        else:
            mismatches += 1
            degraded.append({
                "conversation_id": b["conversation_id"],
                "turn_index": b["turn_index"],
                "turn_text": b.get("turn_text", ""),
                "baseline": b_val,
                "candidate": c_val,
            })

    total = identical + mismatches
    return {
        "identical": identical,
        "mismatches": mismatches,
        "match_rate_pct": round(identical / total * 100, 1) if total else 0,
        "total_compared": total,
        "degraded_examples": degraded[:10],  # cap at 10
    }


def main():
    path_a = os.path.join(REPORT_DIR, "70b_baseline.jsonl")
    path_b = os.path.join(REPORT_DIR, "8b_candidate.jsonl")

    if not os.path.exists(path_a) or not os.path.exists(path_b):
        print(f"Error: Run run_model_comparison.py first.")
        print(f"  Missing: {path_a if not os.path.exists(path_a) else ''}")
        print(f"  Missing: {path_b if not os.path.exists(path_b) else ''}")
        sys.exit(1)

    baseline = load(path_a)
    candidate = load(path_b)

    # ── Latency ──────────────────────────────────────────────────
    b_lat = [r["total_ms"] for r in baseline if r.get("parse_ok")]
    c_lat = [r["total_ms"] for r in candidate if r.get("parse_ok")]
    b_ttft = [r["ttft_ms"] for r in baseline if r.get("parse_ok")]
    c_ttft = [r["ttft_ms"] for r in candidate if r.get("parse_ok")]

    # ── Parse success ────────────────────────────────────────────
    b_ok = sum(1 for r in baseline if r.get("parse_ok"))
    c_ok = sum(1 for r in candidate if r.get("parse_ok"))

    # ── Field comparison ─────────────────────────────────────────
    fields = ["intent", "social_signal", "understanding_check", "reply"]
    field_results = {}
    for field in fields:
        field_results[field] = compare_field(baseline, candidate, field)

    # ── Build report ─────────────────────────────────────────────
    report = {
        "model_a": "llama-3.3-70b-versatile",
        "model_b": "llama-3.1-8b-instant",
        "samples": {
            "model_a": len([r for r in baseline if r.get("parse_ok")]),
            "model_b": len([r for r in candidate if r.get("parse_ok")]),
        },
        "latency": {
            "model_a": {
                "ttft_ms": compute_latency_stats(b_ttft),
                "total_ms": compute_latency_stats(b_lat),
            },
            "model_b": {
                "ttft_ms": compute_latency_stats(c_ttft),
                "total_ms": compute_latency_stats(c_lat),
            },
            "latency_improvement_pct": round(
                ((mean(b_ttft) - mean(c_ttft)) / mean(b_ttft)) * 100, 1
            ) if b_ttft and c_ttft else 0,
        },
        "parse_success_rate": {
            "model_a": round(b_ok / len(baseline) * 100, 1),
            "model_b": round(c_ok / len(candidate) * 100, 1),
        },
        "fields": field_results,
        "degraded_examples": [],
    }

    # Collect all degraded examples across fields
    for field, res in field_results.items():
        for ex in res.get("degraded_examples", []):
            report["degraded_examples"].append(ex)

    # ── Write report ─────────────────────────────────────────────
    out_path = os.path.join(REPORT_DIR, "report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Print summary ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Model Comparison Report")
    print(f"{'=' * 60}")
    print(f"  Model A: {report['model_a']}")
    print(f"  Model B: {report['model_b']}")
    print(f"\n  — Parse success —")
    print(f"    {report['model_a']}: {report['parse_success_rate']['model_a']}%")
    print(f"    {report['model_b']}: {report['parse_success_rate']['model_b']}%")
    print(f"\n  — TTFT (ms) —")
    for m in ["model_a", "model_b"]:
        t = report['latency'][m]['ttft_ms']
        print(f"    {m}: mean={t['mean']}  median={t['median']}  p95={t['p95']}  min={t['min']}  max={t['max']}")
    print(f"\n  — Latency improvement: {report['latency']['latency_improvement_pct']}% —")
    print(f"\n  — Field match rates —")
    for field, res in field_results.items():
        print(f"    {field:25s}: {res['identical']:>3}/{res['total_compared']:>3} identical "
              f"({res['match_rate_pct']}%)  degraded={len(res['degraded_examples']):>3}")
    print(f"\n  — Total degraded examples across all fields: {len(report['degraded_examples'])}")
    print(f"\n  Report saved: {out_path}")
    print(f"{'=' * 60}")

    return report


if __name__ == "__main__":
    main()
