"""visualization.py — Matplotlib figures for transcript stability analysis."""

from __future__ import annotations
import os
from statistics import mean
from typing import Optional

import numpy as np

from metrics import UtteranceMetrics, StreamingWindow
from aggregation import ConfigAggregate, aggregate_utterances

# Optional matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmark_results"
)


def generate_plots(
    all_metrics: list[UtteranceMetrics],
    aggregates: dict[int, ConfigAggregate],
    buffer_sizes: list[int],
    seed: int = 42,
) -> None:
    """Generate all stability plots.

    Skips silently if matplotlib is not available.
    """
    if not HAS_MPL:
        print("  matplotlib not installed — skipping plots")
        return

    np.random.seed(seed)

    # 1. Stable prefix growth curve
    _plot_spr_growth(all_metrics, aggregates, buffer_sizes)

    # 2. Stabilization time CDF
    _plot_stabilization_cdf(all_metrics, aggregates, buffer_sizes)

    # 3. Rollback heatmap
    _plot_rollback_heatmap(all_metrics, aggregates, buffer_sizes)

    # 4. Waterfall for a representative utterance (first one, buffer=2000)
    _plot_waterfall(all_metrics, buffer_sizes)

    # 5. Edit distance curve
    _plot_edit_distance_curve(all_metrics, aggregates, buffer_sizes)

    plt.close("all")
    print(f"  Plots: saved to {OUTPUT_DIR}")


def _plot_spr_growth(
    all_metrics: list[UtteranceMetrics],
    aggregates: dict[int, ConfigAggregate],
    buffer_sizes: list[int],
) -> None:
    """Stable Prefix Ratio Growth: SPR vs window index, one line per config."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for bs in buffer_sizes:
        config_metrics = [m for m in all_metrics if m.buffer_ms == bs]
        if not config_metrics:
            continue
        # Compute mean SPR at each window index (across utterances)
        max_windows = max(len(m.windows) for m in config_metrics)
        spr_sum = np.zeros(max_windows)
        count = np.zeros(max_windows)
        for m in config_metrics:
            for i, spr in enumerate(m.stable_prefix_ratios):
                spr_sum[i] += spr
                count[i] += 1
        mean_spr = np.divide(spr_sum, count, out=np.zeros_like(spr_sum), where=count > 0)
        ax.plot(range(len(mean_spr)), mean_spr, "o-", linewidth=1.5, markersize=3,
                label=f"{bs}ms buffer")

    ax.set_xlabel("Window index (every 500ms)")
    ax.set_ylabel("Mean stable prefix ratio")
    ax.set_title("Stable Prefix Ratio Growth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "stable_prefix_growth.png"), dpi=150)


def _plot_stabilization_cdf(
    all_metrics: list[UtteranceMetrics],
    aggregates: dict[int, ConfigAggregate],
    buffer_sizes: list[int],
) -> None:
    """CDF of per-word stabilization times, one line per config."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for bs in buffer_sizes:
        all_wst = []
        for m in all_metrics:
            if m.buffer_ms == bs:
                for ws in m.word_stabilization_times:
                    if ws is not None:
                        all_wst.append(ws)
        if not all_wst:
            continue
        all_wst = sorted(all_wst)
        n = len(all_wst)
        y = np.linspace(0, 1, n)
        ax.plot(all_wst, y, linewidth=1.5, label=f"{bs}ms buffer (n={n})")

    ax.set_xlabel("Word stabilization time (seconds of audio)")
    ax.set_ylabel("Cumulative fraction of words")
    ax.set_title("Word Stabilization Time Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0)  # start at 0
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "stabilization_cdf.png"), dpi=150)


def _plot_rollback_heatmap(
    all_metrics: list[UtteranceMetrics],
    aggregates: dict[int, ConfigAggregate],
    buffer_sizes: list[int],
) -> None:
    """2D histogram of (window_index, revision_depth) for a representative config."""
    # Pick the middle buffer size
    target_bs = buffer_sizes[len(buffer_sizes) // 2] if buffer_sizes else 2000
    depth_pairs: list[tuple[int, int]] = []
    for m in all_metrics:
        if m.buffer_ms == target_bs:
            for aw in m.windows:
                for d in aw.revision_depths:
                    depth_pairs.append((aw.window.window_index, d))

    if not depth_pairs:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if not depth_pairs:
        # No revisions with non-zero depth in this data — skip the heatmap
        plt.close(fig)
        return

    windows_idx = [p[0] for p in depth_pairs]
    depths = [p[1] for p in depth_pairs]

    h = ax.hist2d(
        windows_idx, depths, bins=(max(windows_idx) + 1, max(depths) + 1),
        cmap="Reds", cmin=1,
    )
    ax.set_xlabel("Window index")
    ax.set_ylabel("Revision depth (words from end)")
    ax.set_title(f"Revision Depth Heatmap — {target_bs}ms buffer")
    fig.colorbar(h[3], ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "rollback_heatmap.png"), dpi=150)


def _plot_waterfall(
    all_metrics: list[UtteranceMetrics],
    buffer_sizes: list[int],
) -> None:
    """Stability waterfall for a representative utterance.

    Picks the first utterance at a representative buffer size.
    Grid: rows = windows, columns = final-transcript word positions.
    Colors: green = stable, blue = provisional, red = revised/orange = not yet seen.
    """
    target_bs = buffer_sizes[len(buffer_sizes) // 2] if buffer_sizes else 2000
    target = None
    for m in all_metrics:
        if m.buffer_ms == target_bs and m.windows:
            target = m
            break
    if target is None:
        return

    n_windows = len(target.windows)
    n_words = len(target.final_words)
    if n_windows == 0 or n_words == 0:
        return

    # Build colour matrix: rows = windows, cols = final-transcript word positions
    # Colour codes: 0 = not yet seen, 1 = provisional, 2 = revised, 3 = stable
    cmap = np.zeros((n_windows, n_words), dtype=np.uint8)

    for wl in target.word_lifecycles:
        ref_idx = wl.ref_index
        last_seen = wl.first_appearance_window if wl.first_appearance_window is not None else 0
        stab_win = wl.stabilization_window if wl.stabilization_window is not None else n_windows + 1

        for win_idx in range(n_windows):
            if win_idx < (wl.first_appearance_window or n_windows + 1):
                cmap[win_idx, ref_idx] = 0  # not seen
            elif win_idx >= stab_win:
                cmap[win_idx, ref_idx] = 3  # stable
            else:
                # Check if this window had a revision for this word
                # We check the window's revision_depths — approximate mapping
                aw = target.windows[win_idx]
                # Determine if this window's form matches the final form
                surface = None
                for a, b in aw.alignment.pairs:
                    if a == ref_idx and b is not None:
                        surface = aw.alignment.b_words[b]
                        break
                if surface is not None:
                    from alignment import normalize_word
                    if normalize_word(surface) == normalize_word(wl.word):
                        cmap[win_idx, ref_idx] = 1  # provisional (correct form)
                    else:
                        cmap[win_idx, ref_idx] = 2  # provisional (wrong form)
                else:
                    cmap[win_idx, ref_idx] = 2  # word absent → non-stable

    fig, ax = plt.subplots(figsize=(max(8, n_words * 0.5), max(5, n_windows * 0.3)))
    im = ax.imshow(cmap.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=3,
                   interpolation="nearest")

    ax.set_xlabel("Window index (time →)")
    ax.set_ylabel("Word position in final transcript")
    ax.set_title(f"Stability Waterfall — {target.utt_id} ({target_bs}ms buffer)")
    fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3],
                 label="0=not seen  1=provisional  2=wrong  3=stable")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "waterfall.png"), dpi=150)


def _plot_edit_distance_curve(
    all_metrics: list[UtteranceMetrics],
    aggregates: dict[int, ConfigAggregate],
    buffer_sizes: list[int],
) -> None:
    """Mean incremental edit distance per window step, one line per config."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for bs in buffer_sizes:
        config_metrics = [m for m in all_metrics if m.buffer_ms == bs]
        if not config_metrics:
            continue

        max_windows = max(len(m.windows) for m in config_metrics)
        ed_sum = np.zeros(max_windows)
        count = np.zeros(max_windows)
        for m in config_metrics:
            for i, aw in enumerate(m.windows):
                if aw.incremental_edit_distance > 0:
                    ed_sum[i] += aw.incremental_edit_distance
                    count[i] += 1
        mean_ed = np.divide(ed_sum, count, out=np.zeros_like(ed_sum), where=count > 0)
        ax.plot(range(len(mean_ed)), mean_ed, "o-", linewidth=1.5, markersize=3,
                label=f"{bs}ms buffer")

    ax.set_xlabel("Window index (every 500ms)")
    ax.set_ylabel("Mean incremental edit distance")
    ax.set_title("Incremental Edit Distance per Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "edit_distance_curve.png"), dpi=150)
