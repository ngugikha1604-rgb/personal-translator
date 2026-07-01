"""tracking.py — Word lifecycle tracking across streaming windows.

Maintains the state of every logical word in the final transcript as it
appears, evolves, and stabilises across consecutive streaming windows.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto

from alignment import Alignment, align, normalize_word


class WordState(Enum):
    """Legal states for a logical word throughout its lifecycle.

    Transitions are enforced in WordTracker:
      NOT_YET_SEEN → PROVISIONAL
      PROVISIONAL  → REVISED | STABLE
      REVISED      → PROVISIONAL  (re-enter provisional after changing)
      STABLE       → <terminal>   (may never leave)
    """
    NOT_YET_SEEN = auto()
    PROVISIONAL = auto()
    REVISED = auto()
    STABLE = auto()


@dataclass
class WordLifecycle:
    """Tracks one logical word (indexed by its position in the final transcript).

    A logical word exists only if it appears in the final transcript.
    Hallucinated words (insertions with no reference counterpart) are tracked
    separately inside WordTracker._hallucinations.
    """
    word: str
    ref_index: int
    first_appearance_window: int | None = None
    first_correct_window: int | None = None
    stabilization_window: int | None = None
    last_revision_window: int | None = None
    revision_count: int = 0
    forms_seen: list[str] = field(default_factory=list)
    state: WordState = WordState.NOT_YET_SEEN


class WordTracker:
    """Tracks all logical words across a sequence of aligned windows.

    Usage:
        final_words = ["hello", "world", ...]
        tracker = WordTracker(final_words)
        for aligned in aligned_windows:
            tracker.update(aligned)
        states = tracker.final_states()
    """

    def __init__(self, final_words: list[str]) -> None:
        self._final_words = final_words
        self._words: list[WordLifecycle] = [
            WordLifecycle(word=w, ref_index=i) for i, w in enumerate(final_words)
        ]
        # Hallucinated words (None ref index) seen across windows
        self._hallucinations: list[tuple[int, str]] = []  # (window_index, word)

    # ── Public API ──

    def update(self, alignment: Alignment, window_index: int) -> list[dict]:
        """Process one aligned window, updating word lifecycles.

        Parameters
        ----------
        alignment : Alignment between final transcript (a_words) and
                    this window's transcript (b_words).
        window_index : which streaming window this is.

        Returns
        -------
        A list of change records (one per word that changed in this window)
        for incremental metric computation.
        """
        assert alignment.a_words == tuple(self._final_words), (
            f"Alignment reference does not match tracker's final_words: "
            f"{len(alignment.a_words)} vs {len(self._final_words)}"
        )

        changes: list[dict] = []

        # Collect which final-transcript words are present in this window
        ref_indices_this_window: set[int] = set()

        for a_idx, b_idx in alignment.pairs:
            if a_idx is not None and b_idx is not None:
                ref_indices_this_window.add(a_idx)
                self._process_ref_word(a_idx, b_idx, alignment, window_index, changes)
            elif b_idx is not None and a_idx is None:
                # Hallucination / insertion
                word = alignment.b_words[b_idx]
                self._hallucinations.append((window_index, word))
                changes.append({
                    "type": "hallucination",
                    "window": window_index,
                    "word": word,
                })

        # Words not present in this window at all
        for a_idx in range(len(self._final_words)):
            if a_idx not in ref_indices_this_window:
                wl = self._words[a_idx]
                if wl.state in (WordState.PROVISIONAL, WordState.STABLE):
                    # Word was present before but now absent — this counts as a
                    # deletion, which is a form of revision.
                    wl.revision_count += 1
                    wl.last_revision_window = window_index
                    wl.state = WordState.REVISED
                    wl.stabilization_window = None  # No longer stable
                    changes.append({
                        "type": "disappeared",
                        "word": wl.word,
                        "ref_index": a_idx,
                    })

        return changes

    def final_states(self) -> list[WordLifecycle]:
        """Return the lifecycle for every word in the final transcript."""
        return list(self._words)

    @property
    def num_stable(self) -> int:
        return sum(1 for w in self._words if w.state == WordState.STABLE)

    @property
    def num_provisional(self) -> int:
        return sum(1 for w in self._words if w.state == WordState.PROVISIONAL)

    @property
    def num_not_yet_seen(self) -> int:
        return sum(1 for w in self._words if w.state == WordState.NOT_YET_SEEN)

    @property
    def hallucination_count(self) -> int:
        return len(self._hallucinations)

    # ── Internal helpers ──

    def _process_ref_word(
        self,
        a_idx: int,
        b_idx: int,
        alignment: Alignment,
        window_index: int,
        changes: list[dict],
    ) -> None:
        """Update the lifecycle for one reference word that appears in this window."""
        wl = self._words[a_idx]
        surface = alignment.b_words[b_idx]
        surface_norm = normalize_word(surface)
        ref_norm = normalize_word(wl.word)

        is_correct = (surface_norm == ref_norm)

        # Record first appearance
        if wl.first_appearance_window is None:
            wl.first_appearance_window = window_index

        # Record form
        if not wl.forms_seen or wl.forms_seen[-1] != surface:
            wl.forms_seen.append(surface)

        if wl.state == WordState.NOT_YET_SEEN:
            # First sighting: enter PROVISIONAL
            wl.state = WordState.PROVISIONAL
            if is_correct:
                wl.first_correct_window = window_index
            changes.append({
                "type": "appeared",
                "word": wl.word,
                "ref_index": a_idx,
                "surface": surface,
                "is_correct": is_correct,
            })

        elif wl.state in (WordState.PROVISIONAL, WordState.REVISED, WordState.STABLE):
            # Already seen — check if the surface changed
            prev_form = wl.forms_seen[-2] if len(wl.forms_seen) >= 2 else None
            if prev_form is not None and normalize_word(prev_form) != surface_norm:
                # The word's form changed
                wl.revision_count += 1
                wl.last_revision_window = window_index
                wl.stabilization_window = None  # No longer stable
                wl.state = WordState.REVISED
                changes.append({
                    "type": "revised",
                    "word": wl.word,
                    "ref_index": a_idx,
                    "old_form": prev_form,
                    "new_form": surface,
                    "is_correct_now": is_correct,
                })
                # Re-enter provisional
                wl.state = WordState.PROVISIONAL

            # Check if this word can be marked stable.
            # Condition: correct form AND at least one full window of stability
            # (we check that the previous form was also correct and the same).
            if is_correct and wl.first_correct_window is None:
                wl.first_correct_window = window_index

            # A word is STABLE if it has been correct for at least one full
            # window and hasn't changed.
            if is_correct and wl.first_correct_window is not None and wl.first_correct_window < window_index:
                wl.state = WordState.STABLE
                wl.stabilization_window = window_index
                changes.append({
                    "type": "stabilised",
                    "word": wl.word,
                    "ref_index": a_idx,
                    "window": window_index,
                })

        # Sanity: STABLE is terminal
        assert wl.state != WordState.STABLE or (
            wl.stabilization_window is not None
        ), f"Word {wl.word} marked STABLE without stabilization_window"
