# Model Comparison Benchmark

Compare `llama-3.3-70b-versatile` (baseline) vs `llama-3.1-8b-instant` (candidate).

## Usage

```bash
cd backend

# Step 1 — Run both models on the 20-conversation test set
python benchmark/model_comparison/run_model_comparison.py

# Step 2 — Generate comparison report
python benchmark/model_comparison/report_model_comparison.py
```

## Output

```
benchmark_results/model_comparison/
├── 70b_baseline.jsonl    # 70B results (turn-by-turn)
├── 8b_candidate.jsonl    # 8B results (turn-by-turn)
└── report.json           # Decision report
```

## What it measures

| Metric | Source |
|---|---|
| TTFT (mean, median, p95) | `_run_groq_stream` internal timing |
| Parse success rate | JSON parse success after LLM call |
| Intent match rate | Field-by-field comparison per turn |
| Social signal match rate | Same |
| Understanding_check match rate | Same |
| Reply match rate | Same |

## Test set

20 conversations total:
- 3 understanding check edge cases (+ 2 custom probes)
- 5 interview conversations
- 4 technical discussions
- 4 networking conversations
- 4 casual + referent conversations

~60 API calls per model. ~5 minutes total with 0.5s inter-request delays.
