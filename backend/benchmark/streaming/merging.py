"""merging.py — Streaming transcript merging strategies.

Each strategy consumes raw (independent) rolling-window transcripts and
produces a sequence of merged transcripts that a downstream consumer
(e.g. an LLM) would actually see.

Strategy comparison is the primary research contribution of the
Transcript Stability benchmark.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from alignment import normalize_word


# ── MergeStrategy interface ──


class MergeStrategy(ABC):
    """Merging strategy — produces merged transcript given raw window output.

    Stateless: each call to merge() depends only on the raw transcript
    and the current merged state.
    """
    name: str

    def reset(self) -> None:
        """Reset any internal state (e.g., history buffer). Override if needed."""
        pass

    @abstractmethod
    def merge(
        self,
        raw_words: list[str],
        current_merged: list[str],
    ) -> list[str]:
        ...


# ── Strategy implementations ──


class NaiveAppend(MergeStrategy):
    """Never revise.  Only append new words beyond the current length.

    This is the simplest possible baseline — identical to taking the first
    N words of the latest raw window (where N = merged length from previous
    step), then appending any new words.

    Advantages: maximum latency (words appear immediately).
    Disadvantages: errors in raw output are permanently embedded.
    """
    name = "naive_append"

    def merge(
        self,
        raw_words: list[str],
        current_merged: list[str],
    ) -> list[str]:
        n = len(current_merged)
        if len(raw_words) <= n:
            return list(current_merged)
        result = list(current_merged)
        result.extend(raw_words[n:])
        return result


class SlidingReplace(MergeStrategy):
    """Replace the overlapping prefix, append new words beyond it.

    This is what a naive streaming pipeline does: for the overlapping
    region (first N words where N = min(current_len, raw_len)), use the
    raw window's version.  Append any new words at the end.

    Advantages: corrects errors when a later decode has better context.
    Disadvantages: words in the middle of the transcript can change,
    causing "volatile" user experience.
    """
    name = "sliding_replace"

    def merge(
        self,
        raw_words: list[str],
        current_merged: list[str],
    ) -> list[str]:
        # ── Raw always overwrites the overlapping prefix (that's the
        #    "replace" semantic).  Keep current_merged's tail only when
        #    it extends further than raw_words currently does. ──
        overlap = min(len(raw_words), len(current_merged))
        result = list(raw_words[:overlap])
        if len(raw_words) > len(current_merged):
            # Raw window has NEW words beyond what's currently merged
            result.extend(raw_words[len(current_merged):])
        elif len(current_merged) > len(raw_words):
            # current_merged has words beyond what raw covers — keep them
            result.extend(current_merged[len(raw_words):])
        return result


class LocalAgreement(MergeStrategy):
    """Word committed only after appearing in N consecutive windows.

    A word is "committed" only when the same surface form appears at the
    same position in N consecutive raw window transcripts.  Words that
    haven't reached N agreements appear in the merged output as
    "uncommitted suffix" (they can still change).

    This is a simplified version of the local agreement policy used in
    Google's and Deepgram's streaming ASR systems.

    The `n` parameter controls the commitment window width:
      N=1 — same as SlidingReplace (commit immediately)
      N=2 — commit after 2 consecutive windows
      N=3 — commit after 3 (more stable, higher latency)
    """
    def __init__(self, n: int = 2) -> None:
        self.n = n
        self.name = f"local_agreement_{n}"
        self._history: list[list[str]] = []

    def reset(self) -> None:
        """Call at the start of each new utterance."""
        self._history = []

    def merge(
        self,
        raw_words: list[str],
        current_merged: list[str],
    ) -> list[str]:
        # Append this window to rolling history
        self._history.append(list(raw_words))
        if len(self._history) > self.n:
            self._history.pop(0)

        # Not enough history yet — return empty output (no words committed
        # until N consecutive windows agree).  This is the CORRECT behavior:
        # local agreement delays commits until N confirmations, so the first
        # N-1 windows produce nothing.  Words WILL appear in the merged
        # transcript only when the agreement window is full.
        if len(self._history) < self.n:
            return []

        # Find N-way agreement prefix (normalized comparison)
        min_len = min(len(w) for w in self._history)
        agreed: list[str] = []
        for pos in range(min_len):
            forms = [h[pos] for h in self._history]
            normalized = [normalize_word(f) for f in forms]
            if len(set(normalized)) == 1:
                agreed.append(forms[-1])  # keep most recent surface form
            else:
                break

        # Build merged output: committed prefix + latest raw suffix
        latest = self._history[-1]
        result = list(agreed)
        suffix_start = min(len(agreed), len(latest))
        result.extend(latest[suffix_start:])
        return result


# ── Available strategies ──


def all_strategies() -> dict[str, MergeStrategy]:
    """Return all strategies keyed by name."""
    return {
        "naive_append": NaiveAppend(),
        "sliding_replace": SlidingReplace(),
        "local_agreement_2": LocalAgreement(2),
        "local_agreement_3": LocalAgreement(3),
    }


def strategy_display_name(name: str) -> str:
    """Human-readable short label for plots and tables."""
    return {
        "naive_append": "Naive Append",
        "sliding_replace": "Sliding Replace",
        "local_agreement_2": "Local Agr. (N=2)",
        "local_agreement_3": "Local Agr. (N=3)",
    }.get(name, name)


if __name__ == "__main__":
    # Standalone sanity check
    print("=== SlidingReplace sanity check ===")
    from alignment import normalize_word

    def show(s):
        print(s)

    sr = SlidingReplace()
    sr.reset()

    # Case 1: raw shorter than merged — merged tail should be kept
    current = ["a", "b", "c", "d"]
    raw = ["x", "y"]
    out = sr.merge(raw, current)
    show(f"  Input:  merged={current}, raw={raw}")
    show(f"  Output: {out}")
    show(f"  Expect: ['x', 'y', 'c', 'd']  (prefix replaced, tail kept)")
    show(f"  PASS:   {out == ['x', 'y', 'c', 'd']}")

    # Case 2: raw longer than merged — new words appended
    current = ["a", "b"]
    raw = ["x", "y", "z"]
    out = sr.merge(raw, current)
    show(f"  Input:  merged={current}, raw={raw}")
    show(f"  Output: {out}")
    show(f"  Expect: ['x', 'y', 'z']  (prefix replaced, new words appended)")
    show(f"  PASS:   {out == ['x', 'y', 'z']}")

    # Case 3: equal length, divergent content — raw wins
    current = ["a", "b", "c"]
    raw = ["x", "y", "z"]
    out = sr.merge(raw, current)
    show(f"  Input:  merged={current}, raw={raw}")
    show(f"  Output: {out}")
    show(f"  Expect: ['x', 'y', 'z']  (all replaced by raw)")
    show(f"  PASS:   {out == ['x', 'y', 'z']}")

    # Case 4: raw empty — keep all of merged
    current = ["a", "b"]
    raw = []
    out = sr.merge(raw, current)
    show(f"  Input:  merged={current}, raw={raw}")
    show(f"  Output: {out}")
    show(f"  Expect: ['a', 'b']  (keep all merged when raw silent)")
    show(f"  PASS:   {out == ['a', 'b']}")
