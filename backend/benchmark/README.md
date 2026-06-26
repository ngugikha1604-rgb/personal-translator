# Benchmark Framework — Conversation Copilot

## Structure

```
benchmark/
├── corpus.py                   # 100 benchmark conversations (25 per category)
├── benchmark_analyzer.py       # Run Analyzer with prompt variants
├── benchmark_verification.py   # Run Verification with prompt variants
├── compare.py                  # Compare results and generate report
└── README.md                   # This file
```

## Setup

```bash
cd backend
# Ensure GROQ_API_KEY is set (in .env file)
```

The scripts import from `services/` and use the real Groq API.

## Running

### 1. Quick test (10 conversations)

```bash
python benchmark/benchmark_analyzer.py --max 10 --output benchmark_results/smoke
```

### 2. Full Analyzer benchmark (100 conversations, ~500 turns)

```bash
# Baseline (current prompt)
python benchmark/benchmark_analyzer.py --output benchmark_results/analyzer_baseline

# Candidate: no example 1
python benchmark/benchmark_analyzer.py \
    --variant no_example_1 \
    --output benchmark_results/analyzer_no_ex1
```

### 3. Full Verification benchmark

```bash
# Baseline
python benchmark/benchmark_verification.py --output benchmark_results/v_baseline

# Candidate: minimal (3 examples)
python benchmark/benchmark_verification.py \
    --variant minimal \
    --output benchmark_results/v_minimal
```

### 4. Compare results

```bash
# Analyzer comparison
python benchmark/compare.py \
    benchmark_results/analyzer_baseline/baseline.jsonl \
    benchmark_results/analyzer_no_ex1/no_example_1.jsonl \
    --output benchmark_results/analyzer_no_ex1_report.json

# Verification comparison
python benchmark/compare.py \
    benchmark_results/v_baseline/verification_baseline.jsonl \
    benchmark_results/v_minimal/verification_minimal.jsonl \
    --output benchmark_results/v_minimal_report.json
```

## Adding new prompt variants

### For Analyzer:

Edit `benchmark/benchmark_analyzer.py`, add to `PROMPT_VARIANTS` dict:

```python
PROMPT_VARIANTS = {
    "baseline": ANALYZER_PROMPT,
    "no_example_1": PROMPT_NO_EXAMPLE_1,
    "my_variant": ANALYZER_PROMPT.replace(...),  # ← add here
}
```

### For Verification:

Edit `benchmark/benchmark_verification.py`, add to `PROMPT_VARIANTS` dict.

You can create reduced prompts by removing example pairs from the string constant.

## Output format

### Analyzer JSONL (per turn)

```json
{
  "conversation_id": "A1",
  "category": "networking",
  "label": "opening with small talk",
  "turn_index": 0,
  "turn_text": "Hey, nice to meet you! ...",
  "intent": "opening a networking conversation",
  "social_signal": "friendly",
  "understanding_check": null,
  "reply": "just here to meet people",
  "raw": "{\"intent\": ...}",
  "llm_ms": 450,
  "ttft_ms": 200,
  "prompt_tokens": 300,
  "completion_tokens": 50,
  "total_tokens": 350,
  "total_ms": 480,
  "parse_ok": true
}
```

### Verification JSONL (per question-response pair)

```json
{
  "conversation_id": "B1",
  "turn_index": 0,
  "question": "Tell me about yourself...",
  "user_response": "I'm from a software background...",
  "understanding_correct": true,
  "factual_error": null,
  "warning": null,
  "llm_ms": 350,
  "total_ms": 370,
  "parse_ok": true
}
```

### Compare report.json

```json
{
  "pipeline": "analyzer",
  "total_turns": 480,
  "total_parsed_both": 475,
  "baseline": { "latency_ms": { "mean": 450, "p95": 780, "p99": 920 } },
  "candidate": { "latency_ms": { "mean": 430, "p95": 750, "p99": 900 } },
  "fields": {
    "intent": { "identical": 430, "mismatches": 45, "match_rate_pct": 90.5 },
    "social_signal": { ... },
    "understanding_check": { ... },
    "reply": { ... }
  },
  "mismatch_details": [
    {
      "conversation_id": "A5",
      "field": "intent",
      "baseline": "probing technical background",
      "candidate": "asking a question"
    }
  ],
  "human_review_required": 137
}
```

## Human review workflow

Every mismatch is stored in `mismatch_details[]`. The report does NOT attempt semantic equivalence. A human must review each mismatch and classify as:

- **Identical** (exact match — should never appear here)
- **Semantically equivalent** (different words, same meaning)
- **Degraded** (candidate is worse)
- **Improved** (candidate is better)

To extract all mismatches for review:

```bash
python -c "
import json
with open('report.json') as f:
    r = json.load(f)
with open('review_items.jsonl', 'w') as f:
    for item in r['mismatch_details']:
        f.write(json.dumps(item) + '\n')
print(f'{len(r[\"mismatch_details\"])} items written to review_items.jsonl')
"
```

## Commands summary

```bash
# Analyzer
python benchmark/benchmark_analyzer.py --output benchmark_results/analyzer_baseline
python benchmark/benchmark_analyzer.py --variant no_example_1 --output benchmark_results/analyzer_no_ex1
python benchmark/compare.py benchmark_results/analyzer_baseline/baseline.jsonl benchmark_results/analyzer_no_ex1/no_example_1.jsonl --output analyzer_report.json

# Verification
python benchmark/benchmark_verification.py --output benchmark_results/v_baseline
python benchmark/benchmark_verification.py --variant minimal --output benchmark_results/v_minimal
python benchmark/compare.py benchmark_results/v_baseline/verification_baseline.jsonl benchmark_results/v_minimal/verification_minimal.jsonl --output v_report.json
```
