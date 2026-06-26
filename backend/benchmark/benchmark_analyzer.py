"""benchmark_analyzer.py — Run Analyzer at different prompt variants and save results.

Usage:
    cd backend
    python benchmark/benchmark_analyzer.py [--max 10] [--output results/run1]

Output: JSONL files with turn-by-turn results.
"""

import json
import os
import sys
import time
from collections import Counter

# ── Add backend to path ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.analyzer import Analyzer, AnalysisResult, ANALYZER_PROMPT
from benchmark.corpus import benchmark_corpus

# ── Prompt variants to test ──────────────────────────────────────
# Define candidate prompts here BEFORE running.
# Example: remove Example 1 (studying) from ANALYZER_PROMPT
PROMPT_NO_EXAMPLE_1 = ANALYZER_PROMPT.replace(
    'Conversation:\nOther: What are you studying?\n\nOutput:\n{{\"intent\": \"trying to understand educational background\", \"social_signal\": \"curious\", \"understanding_check\": null, \"reply\": \"studying AI, mostly building LLM stuff\"}}\n\n---\n\n',
    ""
)

PROMPT_VARIANTS = {
    "baseline": ANALYZER_PROMPT,
    "no_example_1": PROMPT_NO_EXAMPLE_1,
    # Add more variants here as needed:
    # "no_examples_1_2": ...,
}


def run_conversation(conversation: dict, prompt: str, max_turns: int = None) -> list:
    """Run Analyzer on every 'other' turn in a conversation.
    
    Returns list of turn results.
    """
    analyzer = Analyzer()
    buffer = []
    results = []
    turns = conversation["turns"]

    if max_turns:
        turns = turns[:max_turns]

    for turn in turns:
        buffer.append(turn)
        if turn["speaker"] != "other":
            continue  # Only analyze other-speaker turns

        t0 = time.perf_counter()
        try:
            result = analyzer.analyze(buffer, prompt_template=prompt)
        except Exception as exc:
            t1 = time.perf_counter()
            results.append({
                "conversation_id": conversation["id"],
                "category": conversation["category"],
                "label": conversation["label"],
                "turn_index": len(results),
                "turn_text": turn["text"],
                "error": str(exc),
                "total_ms": round((t1 - t0) * 1000),
                "parse_ok": False,
            })
            continue
        t1 = time.perf_counter()

        results.append({
            "conversation_id": conversation["id"],
            "category": conversation["category"],
            "label": conversation["label"],
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

    return results


def run_benchmark(prompt_variants: dict, output_dir: str, max_convs: int = None):
    """Run all prompt variants on all conversations, save results."""
    os.makedirs(output_dir, exist_ok=True)

    conversations = benchmark_corpus
    if max_convs:
        conversations = conversations[:max_convs]

    for variant_name, prompt_text in prompt_variants.items():
        print(f"\n{'=' * 60}")
        print(f"  Running: {variant_name}")
        print(f"{'=' * 60}")

        all_results = []
        n_conv = len(conversations)
        for i, conv in enumerate(conversations):
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{n_conv}] {conv['category']} — {conv['label'][:40]}")
            turn_results = run_conversation(conv, prompt_text)
            all_results.extend(turn_results)

        # Save
        out_path = os.path.join(output_dir, f"{variant_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Summary
        total = len(all_results)
        ok = sum(1 for r in all_results if r.get("parse_ok"))
        errors = total - ok
        latencies = [r["total_ms"] for r in all_results if r.get("parse_ok")]
        mean_lat = sum(latencies) / len(latencies) if latencies else 0

        print(f"\n  Results for {variant_name}:")
        print(f"    Turns: {total}")
        print(f"    Parsed OK: {ok}")
        print(f"    Errors: {errors}")
        print(f"    Mean latency: {mean_lat:.0f}ms")
        print(f"    Saved to: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark Analyzer prompt variants")
    parser.add_argument("--max", type=int, default=None, help="Max conversations to process")
    parser.add_argument("--output", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--variant", type=str, default=None,
                        help="Specific variant to run (default: all)")
    args = parser.parse_args()

    variants = PROMPT_VARIANTS
    if args.variant:
        if args.variant not in variants:
            print(f"Unknown variant: {args.variant}. Available: {list(variants.keys())}")
            sys.exit(1)
        variants = {args.variant: variants[args.variant]}

    run_benchmark(variants, args.output, args.max)

    print(f"\n{'=' * 60}")
    print("  Done. Run compare.py to analyze differences.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
