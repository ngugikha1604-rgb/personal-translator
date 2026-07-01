"""metrics.py — Transcript stability metrics computation.

All metrics are computed from aligned windows and word lifecycles.
Each function is a pure computation with no side effects.
"""

from __future__ import annotations
from statistics import mean, median
from dataclasses import dataclass, field
from typing import Optional

from alignment import Alignment, align, normalize_word
from tracking import WordTracker, WordLifecycle, WordState


# ── Per-window data structures ──


@dataclass
class StreamingWindow:
    """One decode result from a streaming step."""
    window_index: int
    buffer_ms: int
    start_time: float   # audio time (seconds)
    end_time: float     # audio time (seconds)
    transcript_words: list[str]


@dataclass
class AlignedWindow:
    """One streaming window after alignment to the final transcript."""
    window: StreamingWindow
    alignment: Alignment                           # final transcript → this window
    incremental_alignment: Alignment | None = None  # prev window → this window

    # Derived: stable prefix length for this window
    stable_prefix_length: int = 0
    stable_prefix_ratio: float = 0.0

    # Incremental edit info (from prev window → this window)
    incremental_edit_distance: int = 0
    incremental_insertions: int = 0
    incremental_deletions: int = 0
    incremental_substitutions: int = 0
    revision_depths: list[int] = field(default_factory=list)  # depths of each substitution
    max_revision_depth: int = 0


# ── Per-utterance data structure ──


@dataclass
class UtteranceMetrics:
    """All metrics for one utterance at one buffer size."""

    # Identity
    utt_id: str
    buffer_ms: int
    audio_duration_s: float

    # Reference
    final_words: list[str]

    # Windows (chronological)
    windows: list[AlignedWindow]

    # Word lifecycles (one per final-transcript word)
    word_lifecycles: list[WordLifecycle]

    # ── Computed metrics (populated by compute_utterance_metrics) ──

    # Stable prefix (per window)
    stable_prefix_lengths: list[int] = field(default_factory=list)
    stable_prefix_ratios: list[float] = field(default_factory=list)

    # Word stabilization times (seconds, per word; None=never stabilised)
    word_stabilization_times: list[float | None] = field(default_factory=list)

    # Revision counts (per word)
    word_revision_counts: list[int] = field(default_factory=list)

    # Revision correctness (list of bool per word; pending = per revision event)
    revision_correctness_pct: float | None = None

    # Max rollback across the entire utterance
    max_rollback: int = 0

    # Rollback frequency: fraction of windows where a non-end revision occurred
    rollback_frequency: float = 0.0

    # Time-to-stable: wall-clock from audio end until all words stable (seconds)
    time_to_stable_s: float | None = None

    # Convergence window: index of first window where all words are STABLE
    convergence_window: int | None = None

    # Transcript churn: total edit distance accumulated across transitions
    total_churn: int = 0


# ── Metric computation ──


def _stable_prefix(
    alignment: Alignment,
    lifecycles: list[WordLifecycle],
) -> tuple[int, float]:
    """Compute stable prefix length and ratio.

    The stable prefix is the longest initial substring of the *final transcript*
    where every word is in STABLE state.  Once a gap, mismatch, or non-stable
    word is encountered, the prefix ends.

    Returns (length, ratio) where ratio = length / len(final_words).
    """
    final_length = len(alignment.a_words)
    if final_length == 0:
        return 0, 0.0

    stable_count = 0
    # Walk through alignment pairs from the beginning following reference order
    a_idx = 0
    for a, b in alignment.pairs:
        if a is not None and a == a_idx:
            # This pair corresponds to the next expected reference position
            if a_idx < len(lifecycles) and lifecycles[a_idx].state == WordState.STABLE:
                stable_count += 1
                a_idx += 1
            else:
                break
        elif a is not None and a > a_idx:
            # We skipped some reference indices (gaps in alignment)
            break
    # Also account for any reference words before this position that have no
    # alignment pair but are STABLE (shouldn't happen in practice)
    return stable_count, stable_count / final_length if final_length > 0 else 0.0


def _incremental_edits(
    prev_alignment: Alignment | None,
    curr_transcript: list[str],
) -> tuple[int, int, int, int, list[int]]:
    """Compute incremental edit distance and its decomposition.

    Parameters
    ----------
    prev_alignment : Alignment of the *previous* window to the final transcript,
                     or None if this is the first window.
    curr_transcript : This window's transcript words.

    Returns
    -------
    (edit_distance, insertions, deletions, substitutions, depths)
    where depths = distance-from-end for each substitution.
    """
    if prev_alignment is None:
        # First window — all current words are "new"
        n = len(curr_transcript)
        return n, n, 0, 0, []

    # Extract what the previous window produced (its B side)
    prev_words = list(prev_alignment.b_words)

    # Align prev_words → curr_transcript
    inc_align = align(prev_words, curr_transcript)

    n_curr = len(curr_transcript)
    depths = []
    for a, b in inc_align.pairs:
        if a is not None and b is not None:
            from alignment import normalize_word
            if normalize_word(prev_words[a]) != normalize_word(curr_transcript[b]):
                depths.append(n_curr - b)  # distance from end

    return (
        inc_align.edit_distance,
        inc_align.insertions,
        inc_align.deletions,
        inc_align.substitutions,
        depths,
    )


def compute_utterance_metrics(
    utt_id: str,
    buffer_ms: int,
    audio_duration_s: float,
    final_words: list[str],
    windows: list[StreamingWindow],
    lifecycles: list[WordLifecycle],
    aligned_windows: list[AlignedWindow],
) -> UtteranceMetrics:
    """Compute all stability metrics for one utterance.

    Parameters
    ----------
    utt_id : LibriSpeech utterance identifier.
    buffer_ms : rolling buffer size used.
    audio_duration_s : length of the full audio clip.
    final_words : reference transcript word list (from full-audio decode).
    windows : all streaming windows for this utterance.
    lifecycles : word lifecycles (from WordTracker.final_states()).
    aligned_windows : each window aligned to the final transcript (from
                      alignment module), with incremental alignments set.

    Returns fully populated UtteranceMetrics.
    """
    # Reconstruct the final_length from lifecycles
    final_length = len(final_words)

    # 1. Stable prefix (per window)
    stable_prefix_lengths: list[int] = []
    stable_prefix_ratios: list[float] = []
    for aw in aligned_windows:
        spl, spr = _stable_prefix(aw.alignment, lifecycles)
        stable_prefix_lengths.append(spl)
        stable_prefix_ratios.append(spr)
        aw.stable_prefix_length = spl
        aw.stable_prefix_ratio = spr

    # 2. Incremental edits (per window transition)
    max_rollback = 0
    rollback_window_count = 0
    total_churn = 0
    prev_alignment: Alignment | None = None

    for i, aw in enumerate(aligned_windows):
        curr_words = windows[i].transcript_words
        ed, ins, d, subs, depths = _incremental_edits(prev_alignment, curr_words)
        aw.incremental_edit_distance = ed
        aw.incremental_insertions = ins
        aw.incremental_deletions = d
        aw.incremental_substitutions = subs
        aw.revision_depths = depths
        aw.max_revision_depth = max(depths) if depths else 0

        total_churn += ed

        # Rollback = substitution at depth > 1 (not just the last word)
        # depth = 1 means the LAST word changed (expected streaming behavior)
        has_rollback = any(d > 1 for d in depths)
        if has_rollback:
            rollback_window_count += 1
        if depths:
            max_rollback = max(max_rollback, max(depths))

        prev_alignment = aw.alignment

    rollback_frequency = rollback_window_count / len(aligned_windows) if aligned_windows else 0.0

    # 3. Word stabilization times
    word_stab_times: list[float | None] = []
    word_rev_counts: list[int] = []
    for wl in lifecycles:
        word_rev_counts.append(wl.revision_count)
        if wl.stabilization_window is not None:
            # stabilization time = end_time of the stabilization window
            stab_idx = wl.stabilization_window
            if stab_idx < len(windows):
                stab_time = windows[stab_idx].end_time
            else:
                stab_time = audio_duration_s
            word_stab_times.append(stab_time)
        else:
            word_stab_times.append(None)

    # 4. Time-to-stable
    last_stab = max(
        (ws for ws in word_stab_times if ws is not None),
        default=None,
    )
    if last_stab is not None:
        time_to_stable_s = last_stab - audio_duration_s
    else:
        time_to_stable_s = None

    # 5. Convergence window
    convergence_window: int | None = None
    for i, aw in enumerate(aligned_windows):
        n_stable = sum(1 for wl in lifecycles if wl.stabilization_window is not None and wl.stabilization_window <= i)
        # All words that have appeared are stable
        appeared = [wl for wl in lifecycles if wl.first_appearance_window is not None]
        stable_appeared = [wl for wl in appeared if wl.stabilization_window is not None and wl.stabilization_window <= i]
        if len(stable_appeared) == len(appeared) and len(appeared) == final_length:
            convergence_window = i
            break

    # 6. Revision correctness — what fraction of revisions were improvements?
    total_revisions = 0
    correct_revisions = 0
    for wl in lifecycles:
        for j in range(1, len(wl.forms_seen)):
            total_revisions += 1
            prev_form = wl.forms_seen[j - 1]
            curr_form = wl.forms_seen[j]
            ref_word = wl.word
            from alignment import normalize_word
            prev_dist = 0 if normalize_word(prev_form) == normalize_word(ref_word) else 1
            curr_dist = 0 if normalize_word(curr_form) == normalize_word(ref_word) else 1
            if curr_dist < prev_dist:
                correct_revisions += 1
    revision_correctness = (correct_revisions / total_revisions * 100.0) if total_revisions > 0 else None

    metrics = UtteranceMetrics(
        utt_id=utt_id,
        buffer_ms=buffer_ms,
        audio_duration_s=audio_duration_s,
        final_words=final_words,
        windows=aligned_windows,
        word_lifecycles=lifecycles,
        stable_prefix_lengths=stable_prefix_lengths,
        stable_prefix_ratios=stable_prefix_ratios,
        word_stabilization_times=word_stab_times,
        word_revision_counts=word_rev_counts,
        revision_correctness_pct=revision_correctness,
        max_rollback=max_rollback,
        rollback_frequency=rollback_frequency,
        time_to_stable_s=time_to_stable_s,
        convergence_window=convergence_window,
        total_churn=total_churn,
    )
    return metrics
