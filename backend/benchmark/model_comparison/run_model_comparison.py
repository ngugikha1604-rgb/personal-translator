"""run_model_comparison.py — Run Analyzer with two different models, collect results.

Usage:
    cd backend
    python benchmark/model_comparison/run_model_comparison.py

Output:
    benchmark_results/model_comparison/70b_baseline.jsonl
    benchmark_results/model_comparison/8b_candidate.jsonl
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from services.analyzer import Analyzer
from benchmark.corpus import benchmark_corpus
from benchmark.model_comparison.model_test_set import MODEL_TEST_IDS, CUSTOM_PROBES

# ── Models to compare ────────────────────────────────────────────
MODEL_A = "llama-3.3-70b-versatile"   # baseline
MODEL_B = "llama-3.1-8b-instant"      # candidate

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmark_results", "model_comparison"
)
SLEEP_BETWEEN = 0.5  # seconds, to avoid Groq rate limits


def build_conversations() -> list:
    """Build the test set: corpus picks + custom probes."""
    by_id = {c["id"]: c for c in benchmark_corpus}
    result = []

    for cid in MODEL_TEST_IDS:
        if cid in by_id:
            result.append(by_id[cid])
        else:
            print(f"  [WARN] conversation {cid} not found in corpus, skipping")

    result.extend(CUSTOM_PROBES)
    return result


def run_one(analyzer: Analyzer, conversation: dict, model: str,
            max_turns: int = None) -> list:
    """Run Analyzer on every 'other' turn, return result list."""
    buffer = []
    results = []
    turns = conversation["turns"]
    if max_turns:
        turns = turns[:max_turns]

    for turn in turns:
        buffer.append(turn)
        if turn["speaker"] != "other":
            continue

        t0 = time.perf_counter()
        try:
            result = analyzer.analyze(buffer, model_override=model)
        except Exception as exc:
            t1 = time.perf_counter()
            results.append({
                "conversation_id": conversation["id"],
                "category": conversation.get("category", "?"),
                "label": conversation.get("label", "?"),
                "turn_index": len(results),
                "turn_text": turn["text"],
                "error": str(exc)[:200],
                "total_ms": round((t1 - t0) * 1000),
                "parse_ok": False,
            })
            time.sleep(SLEEP_BETWEEN)
            continue

        t1 = time.perf_counter()
        results.append({
            "conversation_id": conversation["id"],
            "category": conversation.get("category", "?"),
            "label": conversation.get("label", "?"),
            "turn_index": len(results),
            "turn_text": turn["text"],
            "intent": result.intent,
            "social_signal": result.social_signal,
            "understanding_check": result.understanding_check,
            "reply": result._parsed.get("reply", "") if result._parsed else "",
            "raw": result.raw,
            "llm_ms": result.llm_ms,
            "ttft_ms": result.ttft_ms,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "total_ms": round((t1 - t0) * 1000),
            "parse_ok": True,
        })

        time.sleep(SLEEP_BETWEEN)

    return results


def run_all(conversations: list, model_name: str, model_label: str, output_path: str):
    """Run all conversations with a given model, save JSONL."""
    analyzer = Analyzer()
    all_results = []
    total_calls = 0

    print(f"\n  Model: {model_name} ({model_label})")
    print(f"  Conversations: {len(conversations)}")

    for i, conv in enumerate(conversations):
        tag = f"  [{i+1}/{len(conversations)}] {conv['id']} — {conv.get('label', '?')[:40]}"
        print(tag)

        turn_results = run_one(analyzer, conv, model_name)
        all_results.extend(turn_results)
        total_calls += len(turn_results)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = sum(1 for r in all_results if r.get("parse_ok"))
    lats = [r["total_ms"] for r in all_results if r.get("parse_ok")]
    mean_lat = sum(lats) / len(lats) if lats else 0

    print(f"  → Saved: {output_path}")
    print(f"  → Calls: {total_calls}, Parse OK: {ok}/{len(all_results)}")
    print(f"  → Mean latency: {mean_lat:.0f}ms")

    return total_calls


def main():
    print("=" * 60)
    print("  Model Comparison Benchmark")
    print("=" * 60)

    conversations = build_conversations()
    print(f"\n  Test set: {len(conversations)} conversations")
    corpus_count = sum(1 for c in conversations if "id" in c and c["id"] in MODEL_TEST_IDS)
    custom_count = len(conversations) - corpus_count
    print(f"  Corpus picks: {corpus_count}, Custom probes: {custom_count}")

    # Run 70B baseline
    out_a = os.path.join(OUTPUT_DIR, "70b_baseline.jsonl")
    calls_a = run_all(conversations, MODEL_A, "70B baseline", out_a)

    # Run 8B candidate
    out_b = os.path.join(OUTPUT_DIR, "8b_candidate.jsonl")
    calls_b = run_all(conversations, MODEL_B, "8B candidate", out_b)

    print(f"\n{'=' * 60}")
    print(f"  Complete: {calls_a + calls_b} total API calls")
    print(f"  Run the comparison report:")
    print(f"    python benchmark/model_comparison/report_model_comparison.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
