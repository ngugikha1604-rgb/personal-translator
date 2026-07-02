"""report.py — Write JSONL, JSON report, and console summary."""

from __future__ import annotations
import json
import os
from dataclasses import asdict
from typing import Optional

from metrics import UtteranceMetrics
from aggregation import ConfigAggregate, format_aggregate_table


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmark_results"
)


def write_jsonl(
    all_metrics: list[UtteranceMetrics],
    path: str | None = None,
) -> str:
    """Write per-utterance, per-window results as JSONL."""
    if path is None:
        path = os.path.join(OUTPUT_DIR, "stability_results.jsonl")
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    row_count = 0
    with open(path, "w") as f:
        for um in all_metrics:
            for win_idx, aw in enumerate(um.windows):
                win = aw.window
                row = {
                    "utt_id": um.utt_id,
                    "buffer_ms": um.buffer_ms,
                    "strategy": um.strategy,
                    "window_index": win.window_index,
                    "window_start_s": round(win.start_time, 3),
                    "window_end_s": round(win.end_time, 3),
                    "transcript": " ".join(win.transcript_words),
                    "stable_prefix_length": aw.stable_prefix_length,
                    "stable_prefix_ratio": round(aw.stable_prefix_ratio, 4),
                    "incremental_edit_distance": aw.incremental_edit_distance,
                    "max_revision_depth": aw.max_revision_depth,
                    "edits_insertions": aw.incremental_insertions,
                    "edits_deletions": aw.incremental_deletions,
                    "edits_substitutions": aw.incremental_substitutions,
                    "commit_latencies": um.commit_latencies,
                }
                f.write(json.dumps(row) + "\n")
                row_count += 1
    print(f"  Saved: {path} ({row_count} rows)")
    return path


def write_report(
    all_metrics: list[UtteranceMetrics],
    aggregates: dict[tuple[int, str], ConfigAggregate],
    buffer_sizes: list[int],
    config_params: dict | None = None,
    path: str | None = None,
) -> str:
    """Write aggregated JSON report.

    JSON keys for aggregates use the format "{buffer_ms}_{strategy}".
    """
    if path is None:
        path = os.path.join(OUTPUT_DIR, "stability_report.json")
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    # Serialize aggregates — keys must be strings
    agg_dict = {}
    for (bs, strat), a in aggregates.items():
        key = f"{bs}_{strat}"
        agg_dict[key] = asdict(a)

    # Per-utterance summary
    utterance_summaries = []
    for um in all_metrics:
        summary = {
            "utt_id": um.utt_id,
            "buffer_ms": um.buffer_ms,
            "strategy": um.strategy,
            "final_word_count": len(um.final_words),
            "window_count": len(um.windows),
            "max_rollback": um.max_rollback,
            "rollback_frequency": round(um.rollback_frequency, 3),
            "time_to_stable_s": round(um.time_to_stable_s, 3) if um.time_to_stable_s is not None else None,
            "convergence_window": um.convergence_window,
            "total_churn": um.total_churn,
            "revision_correctness_pct": round(um.revision_correctness_pct, 1) if um.revision_correctness_pct is not None else None,
            "stable_prefix_ratio_final": round(um.stable_prefix_ratios[-1], 4) if um.stable_prefix_ratios else 0,
            "hallucination_count": um.hallucination_count,
        }
        utterance_summaries.append(summary)

    report = {
        "benchmark": "transcript_stability",
        "streaming_interval_ms": 500,
        "buffer_sizes_ms": buffer_sizes,
        "num_results": len(all_metrics),
        "configuration": config_params or {},
        "aggregates": agg_dict,
        "utterances": utterance_summaries,
    }

    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {path}")
    return path


def print_summary(
    all_metrics: list[UtteranceMetrics],
    aggregates: dict[tuple[int, str], ConfigAggregate],
    baseline: ConfigAggregate | None = None,
) -> None:
    """Print human-readable summary to stdout."""
    print(f"\n{'=' * 100}")
    print(f"  Transcript Stability Benchmark — Summary")
    print(f"{'=' * 100}")
    print(f"  Utterances evaluated: {len(all_metrics)}")

    if baseline is not None:
        print(f"\n  Offline baseline (full-audio decode):")
        print(f"    SPR=1.000, 0-rev=100%, TTS=0.0s (theoretical ceiling)")
        aggregates_display = dict(aggregates)
        aggregates_display[(0, "offline")] = baseline
    else:
        aggregates_display = aggregates

    print(format_aggregate_table(aggregates_display))

    # Word revision distribution
    print(f"\n  Word revision distribution (across all utterances):")
    for (bs, strat) in sorted(aggregates):
        a = aggregates[(bs, strat)]
        print(
            f"    {bs:>5d}ms [{strat:>20s}]:  "
            f"0-rev: {a.rev_pct_no_revisions:>5.1f}%  "
            f"1-rev: {a.rev_pct_one_revision:>5.1f}%  "
            f"2+rev: {a.rev_pct_multi_revision:>5.1f}%"
        )

    # Time-to-stable
    print(f"\n  Time to stable (seconds after utterance end):")
    for (bs, strat) in sorted(aggregates):
        a = aggregates[(bs, strat)]
        print(
            f"    {bs:>5d}ms [{strat:>20s}]:  "
            f"P50={a.tts_seconds_p50:>5.2f}s  "
            f"P95={a.tts_seconds_p95:>5.2f}s  "
            f"({a.tts_seconds_count}/{a.num_utterances} utterances converged)"
        )

    # Rollback
    print(f"\n  Rollback (max depth across utterance):")
    for (bs, strat) in sorted(aggregates):
        a = aggregates[(bs, strat)]
        print(
            f"    {bs:>5d}ms [{strat:>20s}]:  "
            f"P50={a.max_rollback_p50:>5.0f}  "
            f"P95={a.max_rollback_p95:>5.0f}  "
            f"max={a.max_rollback_max:>4d}  "
            f"freq_mean={a.rollback_frequency_mean:.2f}"
        )

    # Correctness
    print(f"\n  Revision correctness (higher = revisions improved accuracy):")
    for (bs, strat) in sorted(aggregates):
        a = aggregates[(bs, strat)]
        print(f"    {bs:>5d}ms [{strat:>20s}]:  {a.revision_correctness_mean:>5.1f}%")

    # Commit latency
    print(f"\n  Commit latency (raw appearance → merged appearance, seconds):")
    for (bs, strat) in sorted(aggregates):
        a = aggregates[(bs, strat)]
        if a.commit_latency_count > 0:
            print(
                f"    {bs:>5d}ms [{strat:>20s}]:  "
                f"P50={a.commit_latency_seconds_p50:>5.2f}s  "
                f"P95={a.commit_latency_seconds_p95:>5.2f}s  "
                f"(n={a.commit_latency_count})"
            )
