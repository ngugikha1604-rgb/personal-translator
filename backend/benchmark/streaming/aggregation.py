"""aggregation.py — Pool per-utterance metrics into (buffer_size, strategy) summaries.

Each utterance produces an UtteranceMetrics object.  This module aggregates
many such objects into corpus-level summary statistics, grouped by BOTH
buffer size AND merge strategy.
"""

from __future__ import annotations
from statistics import mean, median, stdev
from typing import Optional
from dataclasses import dataclass, field

import numpy as np

from metrics import UtteranceMetrics


@dataclass
class ConfigAggregate:
    """Aggregated statistics for one (buffer_size, merge_strategy) configuration."""
    buffer_ms: int
    strategy: str = ""
    num_utterances: int = 0

    # Stable prefix ratio
    spr_mean: float = 0.0
    spr_p50: float = 0.0
    spr_p95: float = 0.0
    spr_p99: float = 0.0
    spr_at_last_window_mean: float = 0.0
    spr_at_last_window_p95: float = 0.0

    # Word stabilization times (pooled across all words)
    wst_seconds_p50: float = 0.0
    wst_seconds_p95: float = 0.0
    wst_seconds_p99: float = 0.0

    # Revision counts (per-word distribution)
    rev_pct_no_revisions: float = 0.0
    rev_pct_one_revision: float = 0.0
    rev_pct_multi_revision: float = 0.0

    # Rollback
    max_rollback_p50: float = 0.0
    max_rollback_p95: float = 0.0
    max_rollback_max: int = 0
    rollback_frequency_mean: float = 0.0
    rollback_frequency_p95: float = 0.0

    # Time-to-stable
    tts_seconds_p50: float = 0.0
    tts_seconds_p95: float = 0.0
    tts_seconds_count: int = 0

    # Convergence
    convergence_window_mean: float = 0.0

    # Churn
    total_churn_mean: float = 0.0
    total_churn_p95: float = 0.0

    # Revision correctness
    revision_correctness_mean: float = 0.0

    # Commit latency (raw → merged appearance)
    commit_latency_seconds_p50: float = 0.0
    commit_latency_seconds_p95: float = 0.0
    commit_latency_count: int = 0

    # Additional
    windows_per_utterance_mean: float = 0.0
    words_per_utterance_mean: float = 0.0


def _bucket_name(word_count: int) -> str:
    if word_count < 10:
        return "short"
    if word_count <= 30:
        return "medium"
    return "long"


def _bucket_utterances(
    metrics_list: list[UtteranceMetrics],
) -> dict[str, list[UtteranceMetrics]]:
    buckets: dict[str, list[UtteranceMetrics]] = {
        "short": [], "medium": [], "long": [],
    }
    for m in metrics_list:
        bucket = _bucket_name(len(m.final_words))
        buckets[bucket].append(m)
    return buckets


def aggregate_by_length(
    all_metrics: list[UtteranceMetrics],
    buffer_sizes: list[int],
) -> dict[tuple[int, str], dict[str, ConfigAggregate]]:
    """Aggregate per-utterance metrics, then group by utterance length.

    Returns dict: (buffer_ms, strategy) → {bucket_name → ConfigAggregate}
    """
    by_key: dict[tuple[int, str], list[UtteranceMetrics]] = {}
    for m in all_metrics:
        key = (m.buffer_ms, m.strategy)
        by_key.setdefault(key, []).append(m)

    result: dict[tuple[int, str], dict[str, ConfigAggregate]] = {}
    for key, metrics_list in by_key.items():
        buckets = _bucket_utterances(metrics_list)
        result[key] = {}
        for name, bucket_list in buckets.items():
            if not bucket_list:
                continue
            subset_agg = aggregate_utterances(bucket_list, [key[0]])
            result[key][name] = subset_agg.get(key, ConfigAggregate(buffer_ms=key[0], strategy=key[1]))
    return result


def aggregate_utterances(
    all_metrics: list[UtteranceMetrics],
    buffer_sizes: list[int],
) -> dict[tuple[int, str], ConfigAggregate]:
    """Aggregate per-utterance metrics into per-(buffer_size, strategy) summaries.

    Parameters
    ----------
    all_metrics : list of UtteranceMetrics from all runs.
    buffer_sizes : list of buffer sizes that were tested.

    Returns
    -------
    dict mapping (buffer_ms, strategy) → ConfigAggregate.
    """
    by_config: dict[tuple[int, str], list[UtteranceMetrics]] = {}
    for m in all_metrics:
        key = (m.buffer_ms, m.strategy)
        by_config.setdefault(key, []).append(m)

    results: dict[tuple[int, str], ConfigAggregate] = {}
    for (bs, strat), metrics_list in by_config.items():
        if not metrics_list:
            results[(bs, strat)] = ConfigAggregate(buffer_ms=bs, strategy=strat, num_utterances=0)
            continue

        agg = ConfigAggregate(buffer_ms=bs, strategy=strat, num_utterances=len(metrics_list))

        # ── Stable prefix ratio ──
        all_sprs: list[float] = []
        last_sprs: list[float] = []
        for m in metrics_list:
            if m.stable_prefix_ratios:
                all_sprs.append(mean(m.stable_prefix_ratios))
                last_sprs.append(m.stable_prefix_ratios[-1])
        if all_sprs:
            agg.spr_mean = round(mean(all_sprs), 4)
            agg.spr_p50 = round(float(np.median(all_sprs)), 4)
            agg.spr_p95 = round(float(np.percentile(all_sprs, 95)), 4)
            agg.spr_p99 = round(float(np.percentile(all_sprs, 99)), 4)
        if last_sprs:
            agg.spr_at_last_window_mean = round(mean(last_sprs), 4)
            agg.spr_at_last_window_p95 = round(float(np.percentile(last_sprs, 95)), 4)

        # ── Word stabilization times ──
        all_wst: list[float] = []
        for m in metrics_list:
            for ws in m.word_stabilization_times:
                if ws is not None:
                    all_wst.append(ws)
        if all_wst:
            agg.wst_seconds_p50 = round(float(np.median(all_wst)), 3)
            agg.wst_seconds_p95 = round(float(np.percentile(all_wst, 95)), 3)
            agg.wst_seconds_p99 = round(float(np.percentile(all_wst, 99)), 3)

        # ── Revision counts ──
        all_word_revs: list[int] = []
        for m in metrics_list:
            all_word_revs.extend(m.word_revision_counts)
        if all_word_revs:
            n = len(all_word_revs)
            agg.rev_pct_no_revisions = round(sum(1 for r in all_word_revs if r == 0) / n * 100, 1)
            agg.rev_pct_one_revision = round(sum(1 for r in all_word_revs if r == 1) / n * 100, 1)
            agg.rev_pct_multi_revision = round(sum(1 for r in all_word_revs if r >= 2) / n * 100, 1)

        # ── Rollback ──
        all_max_rollbacks = [m.max_rollback for m in metrics_list]
        if all_max_rollbacks:
            agg.max_rollback_p50 = round(float(np.median(all_max_rollbacks)), 1)
            agg.max_rollback_p95 = round(float(np.percentile(all_max_rollbacks, 95)), 1)
            agg.max_rollback_max = max(all_max_rollbacks)

        all_rf = [m.rollback_frequency for m in metrics_list]
        if all_rf:
            agg.rollback_frequency_mean = round(mean(all_rf), 3)
            agg.rollback_frequency_p95 = round(float(np.percentile(all_rf, 95)), 3)

        # ── Time-to-stable ──
        valid_tts = [m.time_to_stable_s for m in metrics_list if m.time_to_stable_s is not None]
        if valid_tts:
            agg.tts_seconds_p50 = round(float(np.median(valid_tts)), 3)
            agg.tts_seconds_p95 = round(float(np.percentile(valid_tts, 95)), 3)
            agg.tts_seconds_count = len(valid_tts)

        # ── Convergence ──
        conv_windows = [m.convergence_window for m in metrics_list if m.convergence_window is not None]
        if conv_windows:
            agg.convergence_window_mean = round(mean(conv_windows), 1)

        # ── Churn ──
        all_churn = [m.total_churn for m in metrics_list]
        if all_churn:
            agg.total_churn_mean = round(mean(all_churn), 1)
            agg.total_churn_p95 = round(float(np.percentile(all_churn, 95)), 1)

        # ── Correctness ──
        correct_pcts = [m.revision_correctness_pct for m in metrics_list if m.revision_correctness_pct is not None]
        if correct_pcts:
            agg.revision_correctness_mean = round(mean(correct_pcts), 1)

        # ── Commit latency ──
        all_commit_lat: list[float] = []
        for m in metrics_list:
            for cl in m.commit_latencies:
                if cl is not None:
                    all_commit_lat.append(cl)
        if all_commit_lat:
            agg.commit_latency_seconds_p50 = round(float(np.median(all_commit_lat)), 3)
            agg.commit_latency_seconds_p95 = round(float(np.percentile(all_commit_lat, 95)), 3)
            agg.commit_latency_count = len(all_commit_lat)

        agg.windows_per_utterance_mean = round(mean(len(m.windows) for m in metrics_list), 1)
        agg.words_per_utterance_mean = round(mean(len(m.final_words) for m in metrics_list), 1)

        results[(bs, strat)] = agg

    return results


def add_offline_baseline(
    aggregates: dict[tuple[int, str], ConfigAggregate],
    all_metrics: list[UtteranceMetrics],
) -> ConfigAggregate:
    """Return an 'offline' baseline ConfigAggregate for comparison."""
    num_utt = len(set(m.utt_id for m in all_metrics))
    total_words = sum(len(m.final_words) for m in all_metrics)

    baseline = ConfigAggregate(
        buffer_ms=0,
        num_utterances=num_utt,
        spr_mean=1.0,
        spr_p50=1.0,
        spr_p95=1.0,
        spr_p99=1.0,
        spr_at_last_window_mean=1.0,
        spr_at_last_window_p95=1.0,
        wst_seconds_p50=0.0,
        wst_seconds_p95=0.0,
        wst_seconds_p99=0.0,
        rev_pct_no_revisions=100.0,
        rev_pct_one_revision=0.0,
        rev_pct_multi_revision=0.0,
        max_rollback_p50=0.0,
        max_rollback_p95=0.0,
        max_rollback_max=0,
        rollback_frequency_mean=0.0,
        rollback_frequency_p95=0.0,
        tts_seconds_p50=0.0,
        tts_seconds_p95=0.0,
        tts_seconds_count=num_utt,
        convergence_window_mean=0.0,
        total_churn_mean=0.0,
        total_churn_p95=0.0,
        revision_correctness_mean=100.0,
        commit_latency_seconds_p50=0.0,
        commit_latency_seconds_p95=0.0,
        commit_latency_count=0,
        windows_per_utterance_mean=1.0,
        words_per_utterance_mean=round(total_words / num_utt, 1) if num_utt else 0,
    )
    return baseline


def format_aggregate_table(aggregates: dict[tuple[int, str], ConfigAggregate]) -> str:
    """Return a human-readable summary table string."""
    lines = [
        "\n  Buffer  Strategy               Utter.  SPR     SPR    SPR    wST    wST    Rev%  Rev%   Rollbk TTS    Conv   Commit",
        "  (ms)                        count   mean    P95   final  P50s   P95s   0×    2+×    P95    P95s   win    LatP95",
        "  " + "-" * 120,
    ]
    for (bs, strat) in sorted(aggregates):
        a = aggregates[(bs, strat)]
        lines.append(
            f"  {bs:>5d}  {strat:<25s} {a.num_utterances:>5d}  "
            f"{a.spr_mean:>6.3f} {a.spr_p95:>6.3f} {a.spr_at_last_window_p95:>6.3f} "
            f"{a.wst_seconds_p50:>5.2f} {a.wst_seconds_p95:>5.2f} "
            f"{a.rev_pct_no_revisions:>5.1f} {a.rev_pct_multi_revision:>5.1f} "
            f"{a.max_rollback_p95:>5.0f}  "
            f"{a.tts_seconds_p95:>5.2f} "
            f"{a.convergence_window_mean:>5.1f} "
            f"{a.commit_latency_seconds_p95:>7.3f}"
        )
    return "\n".join(lines)
