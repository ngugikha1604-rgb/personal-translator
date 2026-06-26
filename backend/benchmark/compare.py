"""compare.py — Compare baseline and candidate JSONL results.

Usage:
    cd backend
    python benchmark/compare.py benchmark_results/baseline.jsonl benchmark_results/no_example_1.jsonl
    python benchmark/compare.py benchmark_results/verification_baseline.jsonl benchmark_results/verification_minimal.jsonl

Output:
    report.json — structured comparison with human-review rows
"""

import json
import os
import sys
from collections import Counter


def load_results(path: str) -> list:
    """Load JSONL file."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_stats(latencies: list) -> dict:
    """Compute latency statistics."""
    if not latencies:
        return {"mean": 0, "median": 0, "p50": 0, "p95": 0, "p99": 0, "count": 0}
    sorted_lats = sorted(latencies)
    n = len(sorted_lats)
    return {
        "mean": round(sum(sorted_lats) / n, 1),
        "median": round(sorted_lats[n // 2], 1),
        "p50": round(sorted_lats[int(n * 0.50)], 1),
        "p95": round(sorted_lats[int(n * 0.95)], 1),
        "p99": round(sorted_lats[int(n * 0.99)], 1),
        "count": n,
    }


def compare_analyzer(baseline: list, candidate: list, baseline_path: str = "", candidate_path: str = "") -> dict:
    """Compare two Analyzer result sets."""
    # Build lookup by conversation_id + turn_index
    b_map = {}
    for r in baseline:
        key = (r["conversation_id"], r["turn_index"])
        b_map[key] = r

    c_map = {}
    for r in candidate:
        key = (r["conversation_id"], r["turn_index"])
        c_map[key] = r

    common_keys = set(b_map.keys()) & set(c_map.keys())
    fields = ["intent", "social_signal", "understanding_check", "reply"]

    identical = {f: 0 for f in fields}
    mismatches = {f: 0 for f in fields}
    mismatch_details = []
    b_only_parse_ok = sum(1 for r in baseline if r.get("parse_ok"))
    c_only_parse_ok = sum(1 for r in candidate if r.get("parse_ok"))
    both_parse_ok = sum(1 for k in common_keys if b_map[k].get("parse_ok") and c_map[k].get("parse_ok"))

    for key in common_keys:
        b = b_map[key]
        c = c_map[key]

        # Skip if either failed parsing
        if not b.get("parse_ok") or not c.get("parse_ok"):
            continue

        for field in fields:
            b_val = str(b.get(field, "")).strip()
            c_val = str(c.get(field, "")).strip()
            if b_val == c_val:
                identical[field] += 1
            else:
                mismatches[field] += 1
                mismatch_details.append({
                    "conversation_id": b["conversation_id"],
                    "category": b.get("category", ""),
                    "label": b.get("label", ""),
                    "turn_index": b["turn_index"],
                    "turn_text": b.get("turn_text", ""),
                    "field": field,
                    "baseline": b_val,
                    "candidate": c_val,
                })

    total_turns = len(common_keys)
    total_parsed = both_parse_ok

    # Latency stats
    b_lats = [b_map[k].get("total_ms", 0) for k in common_keys if b_map[k].get("parse_ok")]
    c_lats = [c_map[k].get("total_ms", 0) for k in common_keys if c_map[k].get("parse_ok")]

    report = {
        "pipeline": "analyzer",
        "total_turns": total_turns,
        "total_parsed_both": total_parsed,
        "baseline": {
            "path": baseline_path,
            "turns": len(baseline),
            "parse_ok": b_only_parse_ok,
            "latency_ms": compute_stats(b_lats),
        },
        "candidate": {
            "path": candidate_path,
            "turns": len(candidate),
            "parse_ok": c_only_parse_ok,
            "latency_ms": compute_stats(c_lats),
        },
        "fields": {},
        "total_identical": sum(identical.values()),
        "total_mismatches": sum(mismatches.values()),
        "mismatch_details": mismatch_details,
        "human_review_required": len(mismatch_details),
    }

    for field in fields:
        total = identical[field] + mismatches[field]
        report["fields"][field] = {
            "identical": identical[field],
            "mismatches": mismatches[field],
            "match_rate_pct": round(identical[field] / total * 100, 1) if total else 0,
        }

    return report


def compare_verification(baseline: list, candidate: list, baseline_path: str = "", candidate_path: str = "") -> dict:
    """Compare two Verification result sets."""
    b_map = {}
    for r in baseline:
        key = (r["conversation_id"], r["turn_index"])
        b_map[key] = r

    c_map = {}
    for r in candidate:
        key = (r["conversation_id"], r["turn_index"])
        c_map[key] = r

    common_keys = set(b_map.keys()) & set(c_map.keys())

    identical = 0
    mismatches = 0
    mismatch_details = []
    both_parse_ok = 0

    for key in common_keys:
        b = b_map[key]
        c = c_map[key]
        if not b.get("parse_ok") or not c.get("parse_ok"):
            continue
        both_parse_ok += 1

        b_val = b.get("understanding_correct")
        c_val = c.get("understanding_correct")
        if b_val == c_val:
            identical += 1
        else:
            mismatches += 1
            mismatch_details.append({
                "conversation_id": b["conversation_id"],
                "category": b.get("category", ""),
                "label": b.get("label", ""),
                "turn_index": b["turn_index"],
                "question": b.get("question", ""),
                "user_response": b.get("user_response", ""),
                "field": "understanding_correct",
                "baseline": b_val,
                "candidate": c_val,
                "baseline_warning": b.get("warning"),
                "candidate_warning": c.get("warning"),
            })

    b_lats = [b_map[k].get("total_ms", 0) for k in common_keys if b_map[k].get("parse_ok")]
    c_lats = [c_map[k].get("total_ms", 0) for k in common_keys if c_map[k].get("parse_ok")]

    report = {
        "pipeline": "verification",
        "total_turns": len(common_keys),
        "total_parsed_both": both_parse_ok,
        "baseline": {
            "path": baseline_path,
            "turns": len(baseline),
            "parse_ok": sum(1 for r in baseline if r.get("parse_ok")),
            "latency_ms": compute_stats(b_lats),
        },
        "candidate": {
            "path": candidate_path,
            "turns": len(candidate),
            "parse_ok": sum(1 for r in candidate if r.get("parse_ok")),
            "latency_ms": compute_stats(c_lats),
        },
        "understanding_correct": {
            "identical": identical,
            "mismatches": mismatches,
            "match_rate_pct": round(identical / (identical + mismatches) * 100, 1) if (identical + mismatches) else 0,
        },
        "mismatch_details": mismatch_details,
        "human_review_required": len(mismatch_details),
    }

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("baseline", type=str, help="Baseline JSONL file")
    parser.add_argument("candidate", type=str, help="Candidate JSONL file")
    parser.add_argument("--output", type=str, default="report.json", help="Output report path")
    args = parser.parse_args()

    print(f"Loading baseline: {args.baseline}")
    baseline = load_results(args.baseline)
    print(f"Loading candidate: {args.candidate}")
    candidate = load_results(args.candidate)

    # Detect pipeline type
    is_verification = "verification" in args.baseline or (baseline and "understanding_correct" in baseline[0])

    if is_verification:
        report = compare_verification(baseline, candidate, args.baseline, args.candidate)
    else:
        report = compare_analyzer(baseline, candidate, args.baseline, args.candidate)

    # Write report
    with open(args.output, "w", encoding="utf-8") as f:
        pretty = json.dumps(report, indent=2, ensure_ascii=False, default=str)
        f.write(pretty)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Pipeline: {report['pipeline']}")
    print(f"  Total turns compared: {report['total_turns']}")
    print(f"  Both parsed OK: {report['total_parsed_both']}")
    print(f"{'=' * 60}")

    if report["pipeline"] == "analyzer":
        for field, stats in report["fields"].items():
            match_pct = stats["match_rate_pct"]
            print(f"  {field:25s} identical={stats['identical']:>4}  mismatches={stats['mismatches']:>4}  ({match_pct}%)")
    else:
        s = report["understanding_correct"]
        print(f"  understanding_correct  identical={s['identical']:>4}  mismatches={s['mismatches']:>4}  ({s['match_rate_pct']}%)")

    print(f"\n  Baseline latency: {report['baseline']['latency_ms']['mean']:.0f}ms mean")
    print(f"  Candidate latency: {report['candidate']['latency_ms']['mean']:.0f}ms mean")
    print(f"\n  Human review required: {report['human_review_required']} items")
    print(f"  Report saved: {args.output}")


if __name__ == "__main__":
    main()
