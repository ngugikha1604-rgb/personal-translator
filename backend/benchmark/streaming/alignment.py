"""alignment.py — Sequence alignment for streaming transcript analysis.

Provides Needleman-Wunsch global alignment between two word sequences.
Used for two purposes:
  A) Streaming transcript → Final transcript (for stable prefix, stabilization time)
  B) Window_i → Window_(i+1) (for incremental edits, rollback detection)

We use global alignment (NW) rather than local alignment (SW) because the
streaming output is expected to be a prefix-like rendering of the full
utterance — gaps at the end of the streaming output mean "not yet decoded",
not errors.
"""

from __future__ import annotations
from dataclasses import dataclass


# ── Pre-processing ──


def normalize_word(w: str) -> str:
    """Lower-case and strip leading/trailing punctuation for comparison.

    Whisper output is inconsistently capitalised and punctuated.  Stripping
    both ends keeps the core word intact while discarding surface variation
    that is not semantically meaningful.
    """
    return w.strip().lower().strip(""".,!?;:'"-""")


# ── Alignment data structure ──


@dataclass(frozen=True)
class Alignment:
    """Result of aligning two word sequences A (reference) and B (hypothesis).

    Each element in `pairs` is:
      (a_idx, b_idx) — both indices refer to that word in the respective list.
      (a_idx, None)  — word in A has no counterpart in B (deletion in B).
      (None, b_idx)  — word in B has no counterpart in A (insertion in B).
    """
    a_words: tuple[str, ...]
    b_words: tuple[str, ...]
    pairs: tuple[tuple[int | None, int | None], ...]

    def __post_init__(self) -> None:
        # Validate invariants
        assert len(self.pairs) > 0 or (len(self.a_words) == 0 and len(self.b_words) == 0)
        for a, b in self.pairs:
            if a is not None:
                assert 0 <= a < len(self.a_words), f"a_idx {a} out of range [0, {len(self.a_words)})"
            if b is not None:
                assert 0 <= b < len(self.b_words), f"b_idx {b} out of range [0, {len(self.b_words)})"

    # ── Derived properties ──

    @property
    def edit_distance(self) -> int:
        """Levenshtein distance = number of non-identity pairs."""
        return sum(
            1 for a, b in self.pairs
            if a is None or b is None or normalize_word(self.a_words[a]) != normalize_word(self.b_words[b])
        )

    @property
    def matches(self) -> int:
        """Number of identity pairs (same word, same position)."""
        return sum(
            1 for a, b in self.pairs
            if a is not None and b is not None
               and normalize_word(self.a_words[a]) == normalize_word(self.b_words[b])
        )

    @property
    def insertions(self) -> int:
        """Words in B that have no counterpart in A."""
        return sum(1 for a, b in self.pairs if a is None)

    @property
    def deletions(self) -> int:
        """Words in A that have no counterpart in B."""
        return sum(1 for a, b in self.pairs if b is None)

    @property
    def substitutions(self) -> int:
        """Words that exist in both sequences but differ."""
        return sum(
            1 for a, b in self.pairs
            if a is not None and b is not None
               and normalize_word(self.a_words[a]) != normalize_word(self.b_words[b])
        )


# ── Needleman-Wunsch global alignment ──


def _score(a: str | None, b: str | None) -> int:
    """Pairwise score for the NW scoring matrix."""
    if a is None or b is None:
        return -1  # gap
    if normalize_word(a) == normalize_word(b):
        return +2   # match
    return -1       # mismatch


def align(a: list[str], b: list[str]) -> Alignment:
    """Needleman-Wunsch global alignment of two word sequences.

    Scoring:
      Match (+2), mismatch (-1), gap (-1).

    Parameters
    ----------
    a : reference sequence (e.g. final transcript words).
    b : hypothesis sequence (e.g. streaming window words).

    Returns
    -------
    Alignment object containing the optimal global alignment.
    """
    n, m = len(a), len(b)

    # Handle empty inputs
    if n == 0 and m == 0:
        return Alignment(a_words=(), b_words=(), pairs=())
    if n == 0:
        return Alignment(a_words=(), b_words=tuple(b), pairs=tuple((None, j) for j in range(m)))
    if m == 0:
        return Alignment(a_words=tuple(a), b_words=(), pairs=tuple((i, None) for i in range(n)))

    # Initialise scoring matrix (n+1 × m+1)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + _score(a[i - 1], None)
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + _score(None, b[j - 1])

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score[i - 1][j - 1] + _score(a[i - 1], b[j - 1])
            gap_a = score[i - 1][j] + _score(a[i - 1], None)       # delete from a
            gap_b = score[i][j - 1] + _score(None, b[j - 1])       # insert into b
            score[i][j] = max(match, gap_a, gap_b)

    # Trace back
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and score[i][j] == score[i - 1][j - 1] + _score(a[i - 1], b[j - 1]):
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and score[i][j] == score[i - 1][j] + _score(a[i - 1], None):
            pairs.append((i - 1, None))
            i -= 1
        elif j > 0 and score[i][j] == score[i][j - 1] + _score(None, b[j - 1]):
            pairs.append((None, j - 1))
            j -= 1
        else:
            # Fallback (should not happen with valid traceback)
            if i > 0 and j > 0:
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif i > 0:
                pairs.append((i - 1, None)); i -= 1
            else:
                pairs.append((None, j - 1)); j -= 1

    pairs.reverse()
    return Alignment(a_words=tuple(a), b_words=tuple(b), pairs=tuple(pairs))
