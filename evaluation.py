"""
Evaluation Module for Graph-LCA Anomaly Detection
Covers two distinct evaluation tracks:
  1. Streaming benchmark  — warm vs cold iteration counts, convergence rates, residuals
  2. Magnitude sweep      — AUROC, F1, Precision, Recall per magnitude per anomaly type
     This is the core paper result: a curve, not a single number.
"""

import os
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)
from typing import Dict, List, Any


# ─────────────────────────────────────────────────────────────────────────────
# Track 1 — Streaming Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def load_streaming_metrics(filepath: str = "results/streaming_metrics.npz") -> dict:
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Metrics file not found at {filepath}. Run streaming experiment first."
        )
    data = np.load(filepath)
    return {
        "warm_iters"   : data["warm_iters"],
        "cold_iters"   : data["cold_iters"],
        "warm_conv"    : data["warm_converged"],
        "cold_conv"    : data["cold_converged"],
        "warm_residuals": data["warm_residuals"],
        "cold_residuals": data["cold_residuals"],
    }


def compute_streaming_stats(metrics: dict) -> dict:
    warm  = metrics["warm_iters"]
    cold  = metrics["cold_iters"]
    avg_w = float(np.mean(warm))
    avg_c = float(np.mean(cold))
    return {
        "total_frames"          : len(warm),
        "avg_warm_iters"        : avg_w,
        "avg_cold_iters"        : avg_c,
        "speedup_factor"        : avg_c / max(avg_w, 1.0),
        "warm_convergence_rate" : float(np.mean(metrics["warm_conv"])) * 100.0,
        "cold_convergence_rate" : float(np.mean(metrics["cold_conv"])) * 100.0,
        "mean_warm_residual"    : float(np.mean(metrics["warm_residuals"])),
        "mean_cold_residual"    : float(np.mean(metrics["cold_residuals"])),
    }


def print_streaming_report(stats: dict) -> None:
    print("\n" + "=" * 65)
    print("        LCA STREAMING BENCHMARK REPORT")
    print("=" * 65)
    print(f"  Total Frames Evaluated          : {stats['total_frames']}")
    print(f"  Mean Warm-Start Iterations      : {stats['avg_warm_iters']:.2f}")
    print(f"  Mean Cold-Start Iterations      : {stats['avg_cold_iters']:.2f}")
    print(f"  Iteration Speedup Factor        : {stats['speedup_factor']:.2f}x")
    print(f"  Warm Convergence Rate           : {stats['warm_convergence_rate']:.1f}%")
    print(f"  Cold Convergence Rate           : {stats['cold_convergence_rate']:.1f}%")
    print(f"  Mean Warm Constraint Residual   : {stats['mean_warm_residual']:.5f}")
    print(f"  Mean Cold Constraint Residual   : {stats['mean_cold_residual']:.5f}")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Track 2 — Magnitude Sweep Evaluation
# This is the main detection result Ameer asked for.
# ─────────────────────────────────────────────────────────────────────────────

def anomaly_scores_from_e(e: np.ndarray) -> np.ndarray:
    """
    Per-node anomaly score: L2 norm of e per node.
    For 1D signals this is simply |e_i|.
    """
    return np.abs(e)


def evaluate_single_config(e_lca: np.ndarray, labels: np.ndarray) -> dict:
    """
    Computes detection metrics for one sweep configuration.

    Args:
        e_lca  : anomaly vector output from LCA, shape (N,)
        labels : ground truth boolean array, shape (N,)

    Returns:
        dict with auroc, ap, f1, precision, recall
    """
    scores = anomaly_scores_from_e(e_lca)

    # AUROC and AP are threshold-free — preferred primary metrics
    try:
        auroc = roc_auc_score(labels.astype(int), scores)
        ap    = average_precision_score(labels.astype(int), scores)
    except ValueError:
        # Happens if labels are all one class (degenerate config)
        auroc = 0.5
        ap    = float(labels.mean())

    # For F1/Precision/Recall use a simple threshold: mean + 2*std of scores
    thresh  = scores.mean() + 2.0 * scores.std()
    preds   = (scores > thresh).astype(int)
    gt      = labels.astype(int)

    f1   = f1_score(gt, preds, zero_division=0)
    prec = precision_score(gt, preds, zero_division=0)
    rec  = recall_score(gt, preds, zero_division=0)

    return {
        "auroc"    : auroc,
        "ap"       : ap,
        "f1"       : f1,
        "precision": prec,
        "recall"   : rec,
    }


def evaluate_magnitude_sweep(
    sweep_configs : List[Dict[str, Any]],
    lca_results   : Dict[str, np.ndarray],
) -> Dict[str, Dict[float, dict]]:
    """
    Aggregates detection metrics per magnitude per anomaly type.
    Returns a curve (metric vs magnitude) for each anomaly type.

    Args:
        sweep_configs : list of config dicts from build_sweep_config()
        lca_results   : dict keyed by (type, magnitude, frame_idx) → e array

    Returns:
        curves : {
            'spike': { magnitude: {auroc, ap, f1, precision, recall} },
            'blob' : { magnitude: {auroc, ap, f1, precision, recall} },
        }
    """
    from collections import defaultdict

    # Group results by type and magnitude
    grouped: Dict[str, Dict[float, list]] = {
        'spike': defaultdict(list),
        'blob' : defaultdict(list),
    }

    for cfg in sweep_configs:
        key     = (cfg['type'], cfg['magnitude'], cfg['frame_idx'])
        e_array = lca_results.get(key)
        if e_array is None:
            continue
        metrics = evaluate_single_config(e_array, cfg['labels'])
        grouped[cfg['type']][cfg['magnitude']].append(metrics)

    # Average across frames per magnitude
    curves: Dict[str, Dict[float, dict]] = {}
    for anom_type, mag_dict in grouped.items():
        curves[anom_type] = {}
        for mag, metric_list in sorted(mag_dict.items()):
            curves[anom_type][mag] = {
                k: float(np.mean([m[k] for m in metric_list]))
                for k in metric_list[0].keys()
            }

    return curves


def print_sweep_report(curves: Dict[str, Dict[float, dict]]) -> None:
    for anom_type, mag_dict in curves.items():
        print(f"\n{'='*65}")
        print(f"  MAGNITUDE SWEEP — {anom_type.upper()} ANOMALIES")
        print(f"{'='*65}")
        print(f"  {'Magnitude':>10}  {'AUROC':>7}  {'AP':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}")
        print(f"  {'-'*55}")
        for mag, m in sorted(mag_dict.items()):
            print(
                f"  {mag:>10.1f}  "
                f"{m['auroc']:>7.4f}  "
                f"{m['ap']:>7.4f}  "
                f"{m['f1']:>7.4f}  "
                f"{m['precision']:>7.4f}  "
                f"{m['recall']:>7.4f}"
            )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Correctness Check Reporter
# ─────────────────────────────────────────────────────────────────────────────

def print_correctness_report(correctness_results: List[dict]) -> None:
    """
    Prints objective gap and residual comparison from correctness check runs.
    """
    gaps      = [r['objective_gap']  for r in correctness_results]
    lca_res   = [r['lca_residual']   for r in correctness_results]
    cvx_res   = [r['cvxpy_residual'] for r in correctness_results]
    lca_conv  = [r['lca_converged']  for r in correctness_results]

    print("\n" + "=" * 65)
    print("  CORRECTNESS CHECK — LCA vs CVXPY")
    print("=" * 65)
    print(f"  Frames checked            : {len(correctness_results)}")
    print(f"  LCA convergence rate      : {np.mean(lca_conv)*100:.1f}%")
    print(f"  Mean objective gap        : {np.mean(gaps):.4e}")
    print(f"  Max  objective gap        : {np.max(gaps):.4e}")
    print(f"  Mean LCA residual         : {np.mean(lca_res):.5f}")
    print(f"  Mean CVXPY residual       : {np.mean(cvx_res):.5f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    metrics = load_streaming_metrics()
    stats   = compute_streaming_stats(metrics)
    print_streaming_report(stats)