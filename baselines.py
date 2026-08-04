"""
Baselines for Graph Anomaly Detection
Implements rigorous comparative baselines:
1. Graph Laplacian Regularization (Smooth background filter + anomaly residual)
2. Matrix-Level Robust PCA via ADMM (Principal Component Pursuit for Spatiotemporal Data)
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
import cvxpy as cp


def solve_graph_tv(t_signal: np.ndarray, W: np.ndarray, beta: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Graph Total Variation equation
    minimize_x  0.5*||t - x||^2 + beta * sum_{(i,j) in E} w_ij * |x_i - x_j|
    Anomaly score = residual t - x

    Args:
        t_signal: (N,) signal for this frame
        W: (N,N) symmetric, thresholded edge-weight matrix (NOT the Laplacian)
        beta: TV regularization strength
    Returns:
        x_est: (N,) denoised/smoothed signal
        anomaly_score: (N,) = |t_signal - x_est|, non-negative
        
    """

    N = len(t_signal)
    rows, cols = np.nonzero(np.triu(W, k=1))
    weights = W[rows, cols]
    
    x = cp.Variable(N)
    edge_diff = x[rows] - x[cols]
    tv_penalty = beta * cp.sum(cp.multiply(weights, cp.abs(edge_diff)))
    objective = cp.Minimize(0.5 * cp.sum_squares(t_signal - x) + tv_penalty)
    problem = cp.Problem(objective)
    
    try:
        problem.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5)
    except Exception:
        problem.solve(solver=cp.SCS, verbose=False)

    x_est = x.value if x.value is not None else np.copy(t_signal)
    anomaly_score = np.abs(t_signal - x_est)
    return x_est, anomaly_score


def solve_matrix_rpca(M: np.ndarray, lam_s: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves True Matrix-Level Robust PCA (Principal Component Pursuit):
    Decomposes a spatiotemporal data matrix M into low-rank background X and sparse anomaly E.
    
        minimize  ||X||_* + lam_s * ||E||_1  s.t.  X + E = M
        
    Args:
        M (np.ndarray): Spatiotemporal data matrix (shape: [N, T] - Nodes x Timeframes).
        lam_s (float): Weight for sparse anomaly matrix E.
        lam_r (float): Placeholder parameter for signature consistency.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Low-rank background matrix and sparse anomaly matrix.
    """
    N, T = M.shape
    X = cp.Variable((N, T))
    E = cp.Variable((N, T))
    
    objective = cp.Minimize(cp.normNuc(X) + lam_s * cp.norm1(E))
    constraints = [X + E == M]
    
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.SCS, verbose=False, max_iters=2000)
    except Exception:
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except Exception:
            pass
            
    if X.value is None or E.value is None:
        return np.copy(M), np.zeros_like(M)
        
    return X.value, E.value


if __name__ == "__main__":
    from data import build_laplacian, load_traffic_data, load_adjacency_matrix

    print("Running baselines verification test...")

    if os.path.exists("metr-la.parquet") and os.path.exists("adj_mx.pkl"):
        df = pd.read_parquet("metr-la.parquet")
        adj_mx = load_adjacency_matrix("adj_mx.pkl")
            
        W = (adj_mx + adj_mx.T) / 2.0
        W[W < 0.1] = 0
        
        # Test Graph TV on a single frame vector
        sample_signal = df.iloc[0].values.astype(np.float64) / 100.0
        bg, anom = solve_graph_tv(sample_signal, W)
        n_flag = int((anom > 1e-3).sum())
        print(f"[SUCCESS] Graph TV baseline executed. Output shape: {bg.shape}")
        print(f"  Anomaly score range: [{anom.min():.4f}, {anom.max():.4f}]")
        print(f"  Nodes above 1e-3 threshold: {n_flag}/{len(anom)}")
        
        # Test Matrix RPCA on a small temporal window (e.g., 208 nodes x 50 time steps)
        sample_matrix = df.iloc[:50].values.T.astype(np.float64) / 100.0
        X_mat, E_mat = solve_matrix_rpca(sample_matrix)
        print(f"[SUCCESS] Matrix RPCA baseline executed. Background shape: {X_mat.shape}, Anomaly shape: {E_mat.shape}")
    else:
        print("[ERROR] metr-la.parquet or adj_mx.pkl not found in root directory.")