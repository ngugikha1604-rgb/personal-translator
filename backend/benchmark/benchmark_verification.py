"""benchmark_verification.py — Run Verification at different prompt variants and save results.

Usage:
    cd backend
    python benchmark/benchmark_verification.py [--max 10] [--output results/run1]

Output: JSONL files with turn-by-turn results.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.llm import call_verification_llm, VERIFICATION_PROMPT
from services.copilot import _safe_parse_json
from benchmark.corpus import benchmark_corpus

# ── Prompt variants to test ──────────────────────────────────────
# These are candidate reduced versions of VERIFICATION_PROMPT.
# Define by removing example pairs.
PROMPT_MINIMAL = """You are a conversation alignment checker. Your job is to verify whether the user's spoken response actually addresses the speaker's intent and question — not just whether it's factually accurate.

Primary check: Does the user answer WHAT was asked?
Secondary check: Is the user factually consistent with the profile?

User profile:
- Interests: {interests}
- Communication style: {style}
{context_block}

Check the user's last message ("You" speaker) against the conversation history and the user profile. Return ONLY a valid JSON object:
{{
  "understanding_correct": <true if the user's response addresses the speaker's intent AND is factually consistent, false otherwise>,
  "factual_error": "<if understanding_correct is false, describe what went wrong — either intent mismatch or factual inconsistency. null if everything is correct>",
  "warning": "<brief user-facing warning if there's an issue (5-10 words). Describes WHY the answer missed the mark. null if everything is correct>"
}}

Rules:
- Return ONLY raw JSON. No markdown, no code fences, no extra text.
- understanding_correct must be boolean.
- PRIMARY: Check if the user answered the right question. If the speaker asks WHY and the user answers WHAT, that's understanding_correct = false.
- SECONDARY: Check factual consistency against the profile and earlier conversation.
- factual_error: null if correct, or 1 sentence explaining what was wrong (intent mismatch or factual issue).
- warning: null if correct, or a short, natural phrase (5-10 words, e.g. "They asked WHY, you answered WHAT you study").
- Do not add fields. The only allowed keys are understanding_correct, factual_error, and warning.
- False positives are better than false negatives — when in doubt, warn the user.

---

Examples:

Conversation:
Other: Why are you interested in AI?
You: I study software engineering.

Output:
{{"understanding_correct": false, "factual_error": "Speaker asked WHY the user is interested in AI, but user answered WHAT they study — does not address the question.", "warning": "They asked WHY, you answered WHAT you study"}}

---

Conversation:
Other: Why are you interested in AI?
You: I find LLMs fascinating — they changed how I think about building software.

Output:
{{"understanding_correct": true, "factual_error": null, "warning": null}}

---

Conversation:
Other: Are you from Vietnam?
You: Yeah, I'm from Vietnam, been there my whole life.
Profile: home_country = "USA", years_in_vietnam = 2

Output:
{{"understanding_correct": false, "factual_error": "User said they're from Vietnam and been there whole life, but profile shows they're from USA, only 2 years in Vietnam.", "warning": "Wait — you said Vietnam your whole life, but you're from USA"}}"""


PROMPT_VARIANTS = {
    "baseline": VERIFICATION_PROMPT,
    "minimal": PROMPT_MINIMAL,
    # Add more variants by removing different sets of example pairs
}


# ── Verification helper ─────────────────────────────────────────
def _patch_verification_prompt(prompt_text: str):
    """Temporarily replace VERIFICATION_PROMPT in the llm module."""
    import services.llm as llm_module
    llm_module.VERIFICATION_PROMPT = prompt_text


def _restore_verification_prompt():
    """Restore original VERIFICATION_PROMPT."""
    import services.llm as llm_module
    llm_module.VERIFICATION_PROMPT = VERIFICATION_PROMPT


def build_verification_turns(conversation: dict) -> list:
    """Build a conversation with user responses for verification testing.

    For each "other" turn, we need an adjacent "user" response to verify.
    We create user responses that may or may not match the question.
    """
    turns = conversation["turns"]
    # We need user responses for verification. Create simple filler responses.
    result = []
    for i, turn in enumerate(turns):
        if turn["speaker"] == "other":
            result.append({"speaker": "other", "text": turn["text"]})
            # Add a user response after each other turn
            user_text = _infer_user_response(turn["text"])
            result.append({"speaker": "user", "text": user_text})
    return result


def _infer_user_response(question: str) -> str:
    """Generate a plausible user response for verification testing.
    
    Some are intentionally mismatched to trigger verification warnings.
    """
    # Simple heuristic: match vs mismatch
    question_lower = question.lower()
    
    # ~40% chance of intentional mismatch
    import random
    random.seed(hash(question) % 10000)
    
    if random.random() < 0.4:
        # Intentional mismatch: answer a different question
        if "why" in question_lower:
            return "I study software engineering."
        elif "where" in question_lower:
            return "I studied AI and machine learning."
        elif "how long" in question_lower:
            return "I know Python and JavaScript."
        elif "what" in question_lower and "think" in question_lower:
            return "The policy was announced last week."
        elif "do you have experience" in question_lower:
            return "I've worked with Vue and Angular mostly."
        elif "goals" in question_lower or "future" in question_lower:
            return "I work at a tech startup."
        elif "how" in question_lower:
            return "It's very fast compared to others."
        else:
            return "Yes, definitely."
    else:
        # Matching answer
        if "why" in question_lower:
            return "I find it fascinating, especially the problem-solving aspect."
        elif "what" in question_lower:
            return "I work on AI applications mostly, building language tools."
        elif "where" in question_lower:
            return "At a university in the Bay Area."
        elif "how long" in question_lower:
            return "About 5 years now."
        elif "do you" in question_lower or "have you" in question_lower:
            return "Yes, I've been doing that for a while."
        elif "how" in question_lower:
            return "By breaking it down into smaller steps and analyzing each part."
        else:
            return "That sounds good to me."


def verify_turn(conversation_turns: list) -> dict:
    """Run verification on the last user turn in a conversation."""
    try:
        llm_result = call_verification_llm(conversation_turns)
        parsed = _safe_parse_json(llm_result.text)
        if parsed is None:
            return {
                "parse_ok": False,
                "error": f"Unparseable: {llm_result.text[:200]}",
                "raw": llm_result.text,
                "llm_ms": llm_result.total_ms,
            }
        return {
            "parse_ok": True,
            "understanding_correct": parsed.get("understanding_correct", False),
            "factual_error": parsed.get("factual_error"),
            "warning": parsed.get("warning"),
            "raw": llm_result.text,
            "llm_ms": llm_result.total_ms,
        }
    except Exception as exc:
        return {
            "parse_ok": False,
            "error": str(exc),
            "llm_ms": 0,
        }


def run_benchmark(prompt_variants: dict, output_dir: str, max_convs: int = None):
    """Run verification with each prompt variant on all conversations."""
    os.makedirs(output_dir, exist_ok=True)

    conversations = benchmark_corpus
    if max_convs:
        conversations = conversations[:max_convs]

    for variant_name, prompt_text in prompt_variants.items():
        print(f"\n{'=' * 60}")
        print(f"  Running Verification: {variant_name}")
        print(f"{'=' * 60}")

        _patch_verification_prompt(prompt_text)

        all_results = []
        n_conv = len(conversations)
        for i, conv in enumerate(conversations):
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{n_conv}] {conv['category']} — {conv['label'][:40]}")

            # Build conversation with user responses
            ver_turns = build_verification_turns(conv)
            
            # Each pair of (other, user) turns is a verification opportunity
            for j in range(0, len(ver_turns) - 1, 2):
                if ver_turns[j]["speaker"] != "other":
                    continue
                # Use conversation up to this point
                context = ver_turns[:j + 2]
                t0 = time.perf_counter()
                result = verify_turn(context)
                t1 = time.perf_counter()

                all_results.append({
                    "conversation_id": conv["id"],
                    "category": conv["category"],
                    "label": conv["label"],
                    "turn_index": j // 2,
                    "question": ver_turns[j]["text"],
                    "user_response": ver_turns[j + 1]["text"],
                    "total_ms": round((t1 - t0) * 1000),
                    **result,
                })

        _restore_verification_prompt()

        out_path = os.path.join(output_dir, f"verification_{variant_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        total = len(all_results)
        ok = sum(1 for r in all_results if r.get("parse_ok"))
        latencies = [r["total_ms"] for r in all_results if r.get("parse_ok")]
        mean_lat = sum(latencies) / len(latencies) if latencies else 0

        print(f"\n  Results for {variant_name}:")
        print(f"    Turns: {total}")
        print(f"    Parsed OK: {ok}")
        print(f"    Errors: {total - ok}")
        print(f"    Mean latency: {mean_lat:.0f}ms")
        print(f"    Saved to: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark Verification prompt variants")
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

    _restore_verification_prompt()
    print(f"\n{'=' * 60}")
    print("  Done. Run compare.py to analyze differences.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
