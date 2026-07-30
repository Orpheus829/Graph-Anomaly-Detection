"""
Streaming Experiment for Graph Anomaly Detection (Joint QP LCA)
Implements the frame-by-frame continuous-time streaming loop as mandated by Priority 2. 
Tracks moving optima by comparing warm-start state management against cold re-solves from zero under identical convergence tolerances.
Saves performance metrics to disk for downstream plotting and evaluation.
"""

import os
import time
import numpy as np
from typing import Dict, List, Any

from data import load_traffic_data, load_adjacency_matrix, build_laplacian
from lca import lca_ode_solver
import config



def run_streaming_experiment() -> Dict[str, List[Any]]:
    """
    Executes the frame-by-frame streaming experiment across METR-LA timestamps.
    
    Returns:
        results (dict): Dictionary containing iteration counts, residuals,  and convergence flags for both warm and cold arms.
    """
    print("=" * 60)
    print("INITIALIZING STREAMING EXPERIMENT (METR-LA)")
    print("=" * 60)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 1. Data loading and construct graph topology
    print(f"Loading traffic data from {config.DATA_PATH}...")
    df = load_traffic_data(config.DATA_PATH)
    
    print(f"Loading adjacency matrix from {config.ADJ_PATH}...")
    adj_mx = load_adjacency_matrix(config.ADJ_PATH)
    
    print("Building normalized Laplacian with threshold {:.2f}...".format(config.ADJ_THRESHOLD))
    L = build_laplacian(adj_mx)

    n_frames = config.N_FRAMES
    tol = config.TOL
    max_iters = config.MAX_ITER
    dt = config.DT
    tau = config.TAU
    alpha = config.ALPHA
    lam = config.LAMBDA
    gamma = config.GAMMA

    print(f"Configuration loaded: {n_frames} frames | Tolerance: {tol} | Max Iters: {max_iters}")

    # Metrics logging containers
    warm_iters_list: List[int] = []
    cold_iters_list: List[int] = []
    warm_residuals: List[float] = []
    cold_residuals: List[float] = []
    warm_converged_flags: List[bool] = []
    cold_converged_flags: List[bool] = []

    # Persistence memory for the Warm-Start arm
    x_warm_prev = None
    e_warm_prev = None

    print("\nStarting frame-by-frame continuous-time simulation loop...")
    start_time = time.time()

    for t_idx in range(n_frames):
        # Extraction of the current temporal frame across all 207 sensors
        t_signal = df.iloc[t_idx].values.astype(np.float64)/100.0  # Normalization for numerical stability  

        # -------------------------------------------------------------
        # ARM 1: WARM-START LCA SOLVER
        # Initializes from the converged states of the previous frame
        # -------------------------------------------------------------
        x_w, e_w, iters_w, conv_w, res_w, obj_w = lca_ode_solver(
            t_signal=t_signal,
            L=L,
            alpha=alpha,
            lam=lam,
            gamma=gamma,
            tau=tau,
            dt=dt,
            max_iters=max_iters,
            tol=tol,
            x_init=x_warm_prev,
            e_init=e_warm_prev
        )

        # Update of persistent memory for the next frame's warm start
        x_warm_prev = x_w.copy()
        e_warm_prev = e_w.copy()

        # -------------------------------------------------------------
        # ARM 2: COLD-START LCA SOLVER
        # Re-initializes states at zero for every frame (batch baseline)
        # -------------------------------------------------------------
        x_c, e_c, iters_c, conv_c, res_c, obj_c = lca_ode_solver(
            t_signal=t_signal,
            L=L,
            alpha=alpha,
            lam=lam,
            gamma=gamma,
            tau=tau,
            dt=dt,
            max_iters=max_iters,
            tol=tol,
            x_init=None,
            e_init=None
        )

        # Log records
        warm_iters_list.append(iters_w)
        cold_iters_list.append(iters_c)
        warm_residuals.append(res_w)
        cold_residuals.append(res_c)
        warm_converged_flags.append(conv_w)
        cold_converged_flags.append(conv_c)

        # Periodic progress logging
        if (t_idx + 1) % 25 == 0 or t_idx == 0:
            print(f"Frame [{t_idx + 1}/{n_frames}] | "
                  f"Warm Iters: {iters_w:3d} (Conv: {conv_w}) | "
                  f"Cold Iters: {iters_c:3d} (Conv: {conv_c}) | "
                  f"Constraint Res: {res_w:.2e}")

    total_elapsed = time.time() - start_time
    print(f"\nStreaming experiment completed in {total_elapsed:.2f} seconds.")

    # Computation of comparative summary statistics
    mean_warm_iters = float(np.mean(warm_iters_list))
    mean_cold_iters = float(np.mean(cold_iters_list))
    speedup_factor = mean_cold_iters / max(mean_warm_iters, 1.0)
    warm_success_rate = float(np.mean(warm_converged_flags)) * 100.0
    cold_success_rate = float(np.mean(cold_converged_flags)) * 100.0

    print("=" * 60)
    print("STREAMING PERFORMANCE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Average Warm-Start Iterations : {mean_warm_iters:.2f}")
    print(f"Average Cold-Start Iterations : {mean_cold_iters:.2f}")
    print(f"Iteration Reduction (Speedup) : {speedup_factor:.2f}x fewer steps")
    print(f"Warm-Start Convergence Rate   : {warm_success_rate:.1f}%")
    print(f"Cold-Start Convergence Rate   : {cold_success_rate:.1f}%")
    print("=" * 60)

    # Package results dictionary
    results = {
        'warm_iters': np.array(warm_iters_list),
        'cold_iters': np.array(cold_iters_list),
        'warm_residuals': np.array(warm_residuals),
        'cold_residuals': np.array(cold_residuals),
        'warm_converged': np.array(warm_converged_flags),
        'cold_converged': np.array(cold_converged_flags)
    }

    output_filepath = os.path.join(config.RESULTS_DIR, 'streaming_metrics.npz')
    np.savez(output_filepath, **results)
    print(f"Metrics successfully saved to {output_filepath}")

    return results


if __name__ == '__main__':
    run_streaming_experiment()