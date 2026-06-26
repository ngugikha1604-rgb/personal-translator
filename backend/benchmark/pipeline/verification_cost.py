"""verification_cost.py — Measure and compare Analyzer vs Verification cost.

Determines whether the separate Verification LLM call is worth its latency + tokens.

Usage:
    cd backend
    python benchmark/pipeline/verification_cost.py

Output:
    benchmark_results/verification_cost.json
"""

import json
import os
import sys
import time
from statistics import mean, median

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from services.analyzer import Analyzer
from services.llm import call_verification_llm, LLMResult
from services.copilot import _safe_parse_json

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmark_results"
)
SLEEP_BETWEEN = 0.5


# ── Test data: prompts where questions need verification ──────

TEST_CASES = [
    # (question, user_response)
    ("Why are you interested in AI?", "I study software engineering."),
    ("Why are you interested in AI?", "I find LLMs fascinating — they changed how I think about code."),
    ("What are your career goals?", "I work at a tech startup."),
    ("What are your career goals?", "I want to lead an AI research team in 3 years."),
    ("How long have you been coding?", "I know Python and JavaScript."),
    ("How long have you been coding?", "About 5 years now."),
    ("Where did you study?", "I studied AI and machine learning."),
    ("Where did you study?", "At Stanford University."),
    ("Do you have experience with React?", "I've worked with Vue and Angular mostly."),
    ("Do you have experience with React?", "Yes, I've been using it for about 2 years."),
    ("What's your experience with Python?", "I've been using Python for 6 years in production."),
    ("How would you design a recommendation system?", "I'd start with the data pipeline and user embeddings."),
    ("What do you think about the new privacy policy?", "The policy was announced last week."),
    ("What do you think about the new privacy policy?", "I think it's a step forward, but needs work on data sharing."),
    ("Are you from Vietnam?", "Yeah, I'm from Vietnam, been there my whole life."),
    ("Tell me about a challenge project you've worked on.", "I designed a real-time fraud detection system."),
]


def build_conversation(question: str, user_response: str) -> list:
    """Build turns list for verification: other asks, user answers."""
    return [
        {"speaker": "other", "text": question},
        {"speaker": "user", "text": user_response},
    ]


def run_analyzer(question: str, user_response: str) -> dict:
    """Run Analyzer (gets intent + understanding_check)."""
    turns = [{"speaker": "other", "text": question}]
    t0 = time.perf_counter()
    try:
        result = Analyzer().analyze(turns)
    except Exception as exc:
        return {
            "question": question,
            "user_response": user_response,
            "error": str(exc)[:200],
            "llm_ms": round((time.perf_counter() - t0) * 1000),
            "total_ms": round((time.perf_counter() - t0) * 1000),
            "pipeline": "analyzer",
        }
    t1 = time.perf_counter()

    return {
        "question": question,
        "user_response": user_response,
        "parse_ok": True,
        "pipeline": "analyzer",
        "intent": result.intent,
        "understanding_check": result.understanding_check or None,
        "reply": result._parsed.get("reply", "") if result._parsed else "",
        "llm_ms": result.llm_ms,
        "ttft_ms": result.ttft_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "total_ms": round((t1 - t0) * 1000),
    }


def run_verification(question: str, user_response: str) -> dict:
    """Run Verification pipeline (checks user answer alignment)."""
    turns = build_conversation(question, user_response)
    t0 = time.perf_counter()
    try:
        llm = call_verification_llm(turns)
        parsed = _safe_parse_json(llm.text)
    except Exception as exc:
        return {
            "question": question,
            "user_response": user_response,
            "error": str(exc)[:200],
            "llm_ms": round((time.perf_counter() - t0) * 1000),
            "total_ms": round((time.perf_counter() - t0) * 1000),
            "pipeline": "verification",
        }
    t1 = time.perf_counter()

    return {
        "question": question,
        "user_response": user_response,
        "parse_ok": parsed is not None,
        "pipeline": "verification",
        "understanding_correct": parsed.get("understanding_correct") if parsed else None,
        "warning": parsed.get("warning") if parsed else None,
        "llm_ms": llm.total_ms,
        "ttft_ms": llm.ttft_ms,
        "prompt_tokens": llm.prompt_tokens,
        "completion_tokens": llm.completion_tokens,
        "total_tokens": llm.total_tokens,
        "total_ms": round((t1 - t0) * 1000),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "verification_cost.json")

    print(f"  Running {len(TEST_CASES)} test cases × 2 (Analyzer + Verification) = {len(TEST_CASES)*2} LLM calls")
    print()

    analyzer_results = []
    verif_results = []

    for i, (question, user_response) in enumerate(TEST_CASES):
        print(f"  [{i+1}/{len(TEST_CASES)}] \"{question[:50]}...\"")

        # Analyzer
        a = run_analyzer(question, user_response)
        analyzer_results.append(a)
        time.sleep(SLEEP_BETWEEN)

        # Verification
        v = run_verification(question, user_response)
        verif_results.append(v)
        time.sleep(SLEEP_BETWEEN)

    # ── Compute aggregates ──
    a_ok = [r for r in analyzer_results if r.get("parse_ok")]
    v_ok = [r for r in verif_results if r.get("parse_ok")]

    console_msg = ""
    if len(a_ok) < len(analyzer_results) or len(v_ok) < len(verif_results):
        console_msg = f" (analyzer: {len(a_ok)}/{len(analyzer_results)} ok, " \
                       f"verification: {len(v_ok)}/{len(verif_results)} ok)"

    a_llm = sorted([r["llm_ms"] for r in a_ok])
    v_llm = sorted([r["llm_ms"] for r in v_ok])
    a_tok = [r.get("total_tokens", 0) for r in a_ok]
    v_tok = [r.get("total_tokens", 0) for r in v_ok]

    report = {
        "test_cases": len(TEST_CASES),
        "analyzer_ok": len(a_ok),
        "verification_ok": len(v_ok),
        "latency_ms": {
            "analyzer": {
                "mean": round(mean(a_llm), 1) if a_llm else 0,
                "median": round(median(a_llm), 1) if a_llm else 0,
                "p95": round(a_llm[int(len(a_llm)*0.95)], 1) if a_llm else 0,
                "min": round(min(a_llm), 1) if a_llm else 0,
                "max": round(max(a_llm), 1) if a_llm else 0,
                "samples": len(a_llm),
            },
            "verification": {
                "mean": round(mean(v_llm), 1) if v_llm else 0,
                "median": round(median(v_llm), 1) if v_llm else 0,
                "p95": round(v_llm[int(len(v_llm)*0.95)], 1) if v_llm else 0,
                "min": round(min(v_llm), 1) if v_llm else 0,
                "max": round(max(v_llm), 1) if v_llm else 0,
                "samples": len(v_llm),
            },
        },
        "tokens": {
            "analyzer": {
                "mean_prompt": round(mean([r.get("prompt_tokens", 0) for r in a_ok]), 1) if a_ok else 0,
                "mean_completion": round(mean([r.get("completion_tokens", 0) for r in a_ok]), 1) if a_ok else 0,
                "mean_total": round(mean(a_tok), 1) if a_tok else 0,
            },
            "verification": {
                "mean_prompt": round(mean([r.get("prompt_tokens", 0) for r in v_ok]), 1) if v_ok else 0,
                "mean_completion": round(mean([r.get("completion_tokens", 0) for r in v_ok]), 1) if v_ok else 0,
                "mean_total": round(mean(v_tok), 1) if v_tok else 0,
            },
        },
    }

    # Compute cross-pipeline stats only for test cases where both succeeded
    paired = []
    for a, v in zip(analyzer_results, verif_results):
        if a.get("parse_ok") and v.get("parse_ok"):
            paired.append({
                "question": a["question"],
                "intent": a.get("intent"),
                "check": a.get("understanding_check"),
                "v_correct": v.get("understanding_correct"),
                "v_warning": v.get("warning"),
                "analyzer_ms": a.get("llm_ms", 0),
                "verification_ms": v.get("llm_ms", 0),
            })

    report["paired_cases"] = len(paired)
    if paired:
        a_ms = [p["analyzer_ms"] for p in paired]
        v_ms = [p["verification_ms"] for p in paired]
        total = sum(a_ms) + sum(v_ms)
        report["comparison"] = {
            "avg_analyzer_ms": round(mean(a_ms), 1),
            "avg_verification_ms": round(mean(v_ms), 1),
            "verification_pct_of_total": round(sum(v_ms) / total * 100, 1) if total else 0,
            "verification_token_pct": round(
                report["tokens"]["verification"]["mean_total"] / (
                    report["tokens"]["analyzer"]["mean_total"] +
                    report["tokens"]["verification"]["mean_total"]
                ) * 100, 1
            ) if (report["tokens"]["analyzer"]["mean_total"] + report["tokens"]["verification"]["mean_total"]) else 0,
        }

    # ── Print summary ──
    print(f"\n{'=' * 60}")
    print(f"  Verification Cost Report{console_msg}")
    print(f"{'=' * 60}")
    print(f"  Test cases: {report['test_cases']} (analyzer ok: {report['analyzer_ok']}, verification ok: {report['verification_ok']})")
    print(f"\n  — Latency —")
    for pipe in ["analyzer", "verification"]:
        l = report["latency_ms"][pipe]
        print(f"    {pipe:15s}: mean={l['mean']:>6.0f}ms  median={l['median']:>6.0f}ms  p95={l['p95']:>6.0f}ms  n={l['samples']}")
    if "comparison" in report:
        c = report["comparison"]
        print(f"\n  — Cross-pipeline comparison ({report['paired_cases']} paired cases) —")
        print(f"    Avg Analyzer ms:     {c['avg_analyzer_ms']:.0f}ms")
        print(f"    Avg Verification ms: {c['avg_verification_ms']:.0f}ms")
        print(f"    Verification % total: {c['verification_pct_of_total']}%")
        print(f"    Verification token %: {c['verification_token_pct']}%")

    # ── Save ──
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
