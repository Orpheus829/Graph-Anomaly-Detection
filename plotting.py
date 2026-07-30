"""
Plotting Module for Graph-LCA Anomaly Detection
Generates publication-ready figures for:
  1. Streaming experiment  — warm vs cold iterations + residuals
  2. Magnitude sweep       — AUROC/F1 curves per anomaly type (main result)
  3. Baseline comparison   — detection metrics across methods
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Any

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────
COLORS = {
    "warm"   : "#5cb85c",
    "cold"   : "#d9534f",
    "residual": "#337ab7",
    "spike"  : "#e67e22",
    "blob"   : "#8e44ad",
    "lca"    : "#2ecc71",
    "graphtv": "#3498db",
    "rpca"   : "#e74c3c",
}

plt.rcParams.update({
    "font.family"     : "serif",
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
    "figure.dpi"      : 150,
})


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Streaming Experiment
# ─────────────────────────────────────────────────────────────────────────────

def plot_streaming_experiment(
    metrics_path: str = "results/streaming_metrics.npz",
    output_path : str = "results/streaming_experiment.png",
) -> None:
    """
    Two-panel figure:
      Top    — warm vs cold iterations per frame
      Bottom — constraint residual over frames
    """
    data       = np.load(metrics_path)
    warm_iters = data["warm_iters"]
    cold_iters = data["cold_iters"]
    residuals  = data["warm_residuals"]
    frames     = np.arange(len(warm_iters))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Panel 1 — iterations
    ax1.plot(frames, cold_iters, color=COLORS["cold"],
             label="Cold-Start (re-init at zero every frame)",
             alpha=0.8, linewidth=1.2)
    ax1.plot(frames, warm_iters, color=COLORS["warm"],
             label="Warm-Start (state persists frame-to-frame)",
             alpha=0.9, linewidth=1.2)
    ax1.set_ylabel("Iterations to Convergence", fontsize=11)
    ax1.set_title(
        "LCA Streaming: Warm-Start Tracks Moving Optimum vs Cold Re-Solve",
        fontsize=12, fontweight="bold", pad=10
    )
    ax1.legend(fontsize=9, frameon=True)

    mean_w = warm_iters.mean()
    mean_c = cold_iters.mean()
    ax1.axhline(mean_w, color=COLORS["warm"], linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.axhline(mean_c, color=COLORS["cold"],  linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.text(frames[-1]*0.98, mean_w*1.02, f"μ={mean_w:.0f}",
             color=COLORS["warm"], fontsize=8, ha="right")
    ax1.text(frames[-1]*0.98, mean_c*1.02, f"μ={mean_c:.0f}",
             color=COLORS["cold"],  fontsize=8, ha="right")

    # Panel 2 — residual
    ax2.plot(frames, residuals, color=COLORS["residual"],
             label=r"Constraint residual $\|t - (x+e)\|_2$", linewidth=1.2)
    ax2.set_xlabel("METR-LA Frame Index", fontsize=11)
    ax2.set_ylabel(r"$\|t - x - e\|_2$", fontsize=11)
    ax2.legend(fontsize=9, frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Streaming plot saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Magnitude Sweep Curves (primary detection result)
# ─────────────────────────────────────────────────────────────────────────────

def plot_magnitude_sweep(
    curves     : Dict[str, Dict[float, dict]],
    output_path: str = "results/magnitude_sweep.png",
) -> None:
    """
    Two-panel figure — one panel per anomaly type (spike, blob).
    Each panel shows AUROC and F1 vs magnitude.
    This is the curve Ameer asked for.
    """
    anom_types = [t for t in ["spike", "blob"] if t in curves]
    n_panels   = len(anom_types)

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, anom_type in zip(axes, anom_types):
        mag_dict = curves[anom_type]
        mags     = sorted(mag_dict.keys())
        aurocs   = [mag_dict[m]["auroc"] for m in mags]
        f1s      = [mag_dict[m]["f1"]    for m in mags]
        aps      = [mag_dict[m]["ap"]    for m in mags]

        ax.plot(mags, aurocs, marker="o", color=COLORS[anom_type],
                label="AUROC", linewidth=1.8, markersize=5)
        ax.plot(mags, f1s,    marker="s", color=COLORS[anom_type],
                label="F1",    linewidth=1.8, markersize=5, linestyle="--", alpha=0.8)
        ax.plot(mags, aps,    marker="^", color=COLORS[anom_type],
                label="Avg Precision", linewidth=1.4, markersize=4,
                linestyle=":", alpha=0.7)

        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8,
                   label="Random baseline (AUROC=0.5)")
        ax.set_xlabel("Anomaly Magnitude (× local σ)", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(
            f"Detection Curve — {anom_type.capitalize()} Anomalies\n"
            f"(LCA · METR-LA · 207 sensors)",
            fontsize=11, fontweight="bold"
        )
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9, frameon=True)
        ax.set_xticks(mags)

    plt.suptitle(
        "Graph-LCA Anomaly Detection: Performance vs Injection Magnitude",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Magnitude sweep plot saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Baseline Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_baseline_comparison(
    method_curves: Dict[str, Dict[str, Dict[float, dict]]],
    anom_type    : str = "spike",
    metric       : str = "auroc",
    output_path  : str = "results/baseline_comparison.png",
) -> None:
    """
    Overlays AUROC (or chosen metric) vs magnitude for multiple methods.

    Args:
        method_curves : { method_name: curves_dict_from_evaluate_magnitude_sweep }
        anom_type     : 'spike' or 'blob'
        metric        : 'auroc', 'f1', 'ap', 'precision', 'recall'
    """
    method_colors = {
        "LCA"     : COLORS["lca"],
        "Graph TV": COLORS["graphtv"],
        "RPCA"    : COLORS["rpca"],
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, curves in method_curves.items():
        if anom_type not in curves:
            continue
        mag_dict = curves[anom_type]
        mags     = sorted(mag_dict.keys())
        scores   = [mag_dict[m][metric] for m in mags]
        color    = method_colors.get(method, "gray")
        ax.plot(mags, scores, marker="o", label=method, color=color,
                linewidth=1.8, markersize=5)

    if metric == "auroc":
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8,
                   label="Random baseline")

    ax.set_xlabel("Anomaly Magnitude (× local σ)", fontsize=11)
    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.set_title(
        f"Baseline Comparison — {anom_type.capitalize()} Anomalies · {metric.upper()}",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10, frameon=True)
    ax.set_xticks(mags)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Baseline comparison plot saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick standalone test (streaming plot only)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    metrics_file = os.path.join(RESULTS_DIR, "streaming_metrics.npz")
    if os.path.exists(metrics_file):
        plot_streaming_experiment(metrics_file)
    else:
        print("No streaming_metrics.npz found — run streaming experiment first.")