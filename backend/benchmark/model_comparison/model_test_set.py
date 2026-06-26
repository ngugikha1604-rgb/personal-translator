"""model_test_set.py — 20 high-information conversations for model comparison.

Each entry references a conversation from the main corpus by ID,
plus optional custom single-turn prompts for specific edge cases.
"""

# IDs from benchmark/corpus.py
MODEL_TEST_IDS = [
    # ── Understanding check cases (3) ────────────────────────────────
    "A5",   # "What got you into that?" → canonical WHY vs WHAT check
    "B4",   # "Why do you want to leave?" → motivation probing
    "B14",  # "How do you stay current?" → process vs examples check

    # ── Interview (5) ───────────────────────────────────────────────
    "B0",   # "Tell me about yourself" → open-ended intent
    "B2",   # "Behavioral: conflict resolution" → narrative tracking
    "B7",   # "Where in 5 years?" → future vs current distinction
    "B12",  # "Tell me about a time you failed" → narrative vs listing
    "B9",   # "Salary expectations" → direct factual probe

    # ── Technical discussion (4) ─────────────────────────────────────
    "C0",   # "Monolith vs microservices" → opinion seeking
    "C1",   # "Memory leak debugging" → multi-turn chain
    "C5",   # "SQL vs NoSQL" → tradeoff vs direct answer
    "C15",  # "Testing philosophy" → opinion probe

    # ── Networking (4) ───────────────────────────────────────────────
    "A2",   # "Indirect recruiting" → subtle social signal
    "A7",   # "Polite interest fading" → social signal nuance
    "A13",  # "Hobby crossover — jam session" → informal proposal
    "A20",  # "Polite closing" → fade-out detection

    # ── Casual + referent (4) ─────────────────────────────────────────
    "D3",   # "Travel stories" → referent resolution ("that place")
    "D6",   # "Movie discussion" → casual opinion vs formal review
    "D18",  # "Cooking attempt" → advice vs sympathy
    "D22",  # "New hobby discovery" → interest probe
]

# Custom single-turn prompts for hardened understanding check tests
CUSTOM_PROBES = [
    {
        "id": "custom_why_vs_what",
        "category": "understanding_check",
        "label": "canonical WHY vs WHAT",
        "turns": [
            {"speaker": "other", "text": "What got you interested in AI?"},
        ]
    },
    {
        "id": "custom_how_vs_what",
        "category": "understanding_check",
        "label": "HOW mechanism vs evaluation",
        "turns": [
            {"speaker": "other", "text": "How does this sorting algorithm work?"},
        ]
    },
    {
        "id": "custom_opinion_vs_fact",
        "category": "understanding_check",
        "label": "OPINION vs factual summary",
        "turns": [
            {"speaker": "other", "text": "What do you think about the new data privacy regulations?"},
        ]
    },
    {
        "id": "custom_goals_vs_current",
        "category": "understanding_check",
        "label": "FUTURE goals vs current job",
        "turns": [
            {"speaker": "other", "text": "What are your career goals for the next 5 years?"},
        ]
    },
]
