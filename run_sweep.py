"""
Orchestration script: runs the magnitude sweep across LCA, Graph TV, and
(optionally) RPCA, then evaluates and plots the comparison.

This is the piece that was missing — data.py builds sweep configs,
evaluation.py scores them, plotting.py plots them, but nothing called
them in sequence until now.
"""

import time
import numpy as np

import config
from data import load_traffic_data, load_adjacency_matrix, build_laplacian, build_sweep_config
from lca import lca_ode_solver
from baselines import solve_graph_tv, solve_matrix_rpca
from evaluation import evaluate_magnitude_sweep, print_sweep_report
from plotting import plot_baseline_comparison

# ── Sweep scope ─────────────────────────────────────────────────────────
# RPCA needs a full N x T matrix per config (context window + corrupted frame), which makes it the expensive part of this loop. 
# We keep the frame count small first to confirm everything wires together, then scale up.

N_SWEEP_FRAMES = 10          # number of distinct frame_idx to sweep over
RPCA_WINDOW    = 30          # clean context frames used to build RPCA's matrix
RUN_RPCA_SWEEP = False        # setting False to skip RPCA (to prevent runtime bottleneck)


def run_rpca_on_config(df, adj_mx, cfg, window=RPCA_WINDOW):
    """
    Builds a [N x (window+1)] matrix of clean context frames plus the corrupted frame as the last column, runs RPCA once, and returns the anomaly vector (E's last column) for that config's frame.
    """
    t = cfg['frame_idx']
    start = max(0, t - window)
    context = df.iloc[start:t].values.T.astype(np.float64) / 100.0   # (N, window)
    corrupted_col = (cfg['corrupted_signal'] / 100.0).reshape(-1, 1)  # (N, 1)
    M = np.hstack([context, corrupted_col])
    _, E = solve_matrix_rpca(M)
    return E[:, -1]


def main():
    print("Loading data and building graph...")
    df = load_traffic_data(config.DATA_PATH)
    adj_mx = load_adjacency_matrix(config.ADJ_PATH)
    L = build_laplacian(adj_mx)

    # Fixed, evenly spaced frame indices — chosen independent of injected labels.
    rng = np.random.default_rng(config.SEED)
    frame_idx = sorted(rng.choice(
        np.arange(200, len(df) - 200), size=N_SWEEP_FRAMES, replace=False).tolist())
    print(f"Sweeping over frames: {frame_idx}")

    configs = build_sweep_config(df, adj_mx, frame_idx, magnitudes=config.MAGNITUDES)

    lca_results, graphtv_results, rpca_results = {}, {}, {}

    t0 = time.time()
    for i, cfg in enumerate(configs):
        key = (cfg['type'], cfg['magnitude'], cfg['frame_idx'])
        t_signal = cfg['corrupted_signal'].astype(np.float64) / 100.0

        # LCA (core method)
        _, e_lca, iters, converged, residual, _ = lca_ode_solver(
            t_signal, L,
            alpha=config.ALPHA, lam=config.LAMBDA, gamma=config.GAMMA,
            tau=config.TAU, dt=config.DT, max_iters=config.MAX_ITER, tol=config.TOL
        )
        lca_results[key] = e_lca
        if not converged:
            print(f"  [warn] LCA did not converge for {key} (residual={residual:.4f})")

        # Graph TV baseline
        _, e_tv = solve_graph_tv(t_signal, L, alpha=config.ALPHA, beta=config.LAMBDA)
        graphtv_results[key] = e_tv

        # RPCA baseline (expensive — optional)
        if RUN_RPCA_SWEEP:
            e_rpca = run_rpca_on_config(df, adj_mx, cfg)
            rpca_results[key] = e_rpca

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(configs)}] configs done — {elapsed:.1f}s elapsed "
                  f"({elapsed/(i+1):.2f}s/config)")

    print(f"\nSweep loop finished in {time.time()-t0:.1f}s total.")

    # ── Evaluate ────────────────────────────────────────────────────────
    curves_lca = evaluate_magnitude_sweep(configs, lca_results)
    curves_tv  = evaluate_magnitude_sweep(configs, graphtv_results)
    method_curves = {"LCA": curves_lca, "Graph TV": curves_tv}

    if RUN_RPCA_SWEEP:
        curves_rpca = evaluate_magnitude_sweep(configs, rpca_results)
        method_curves["RPCA"] = curves_rpca

    print("\n=== LCA ===")
    print_sweep_report(curves_lca)
    print("\n=== Graph TV ===")
    print_sweep_report(curves_tv)
    if RUN_RPCA_SWEEP:
        print("\n=== RPCA ===")
        print_sweep_report(curves_rpca)

    # ── Plot ────────────────────────────────────────────────────────────
    for anom_type in ["spike", "blob"]:
        for metric in ["auroc", "f1"]:
            plot_baseline_comparison(
                method_curves, anom_type=anom_type, metric=metric,
                output_path=f"results/baseline_comparison_{anom_type}_{metric}.png"
            )

    np.savez(
        "results/sweep_results.npz",
        lca_curves=curves_lca, graphtv_curves=curves_tv,
        rpca_curves=(curves_rpca if RUN_RPCA_SWEEP else None),
        frame_idx=frame_idx,
    )
    print("\nSaved results/sweep_results.npz")


if __name__ == "__main__":
    main()