"""verify_merge_strategies.py — Sanity check: compare all merge strategies side by side."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from merging import all_strategies

raw_window_sequence = [
    ["I", "think", "he"],
    ["I", "think", "he", "is"],
    ["I", "think", "he", "was", "here"],   # revision: is → was
    ["I", "think", "he", "was", "here"],
    ["I", "think", "he", "was", "here", "today"],
]

strategies = {"raw": None, **all_strategies()}
for name, strat in strategies.items():
    if strat is not None:
        strat.reset()
    current = []
    print(f"\n{name}:")
    for i, raw in enumerate(raw_window_sequence):
        merged = list(raw) if strat is None else strat.merge(raw, current)
        current = merged
        print(f"  window {i}: {merged}")
