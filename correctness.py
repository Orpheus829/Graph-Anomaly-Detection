import numpy as np
import cvxpy as cp

def compute_lca_objective(t, x, e, L, alpha, gamma, lamb):
    """
    Exact scalar value of the Joint QP objective for the LCA outputs.
    Formula: 0.5 * ||t - x - e||^2 + (alpha/2)*x^T*L*x + lamb*||e||_1 + (gamma/2)*e^T*L*e
    """
    reconstruction = 0.5 * np.sum((t - x - e)**2)
    smoothness = (alpha / 2.0) * (x.T @ L @ x)
    sparsity = lamb * np.sum(np.abs(e))
    blob = (gamma / 2.0) * (e.T @ L @ e)
    
    return reconstruction + smoothness + sparsity + blob


def solve_cvxpy_ground_truth(t_signal, L, alpha, gamma, lamb):
    """Solves the Unified Joint QP using the CVXPY batch solver (OSQP)."""

    N = len(t_signal)
    x = cp.Variable(N)
    e = cp.Variable(N)
    
    # Positive Semi-Definite (PSD) status for the Laplacian.
    # The 1e-8 diagonal addition fixes floating-point rounding errors that crash CVXPY.
    L_psd = cp.psd_wrap(L + np.eye(N) * 1e-8)
    
    # Exact objective formulation
    objective = cp.Minimize(0.5 * cp.sum_squares(t_signal - x - e) + (alpha / 2) * cp.quad_form(x, L_psd) + lamb * cp.norm1(e) + (gamma / 2) * cp.quad_form(e, L_psd) )
    
    prob = cp.Problem(objective)
    # OSQP is robust for QPs, using eps_abs/eps_rel to match standard tolerance
    prob.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5)
    
    x_opt = x.value if x.value is not None else np.zeros(N)
    e_opt = e.value if e.value is not None else np.zeros(N)
    
    # Constraint Residual ||t - (x+e)||
    residual = np.linalg.norm(t_signal - (x_opt + e_opt), ord=2) if x.value is not None else np.inf
    # Converged Flag
    converged = prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
    
    return {
        'x': x_opt,
        'e': e_opt,
        'obj': prob.value if prob.value is not None else np.inf,
        'converged': converged,
        'residual': residual
    }

def run_correctness_check(t_signal, L, alpha, gamma, lamb, lca_x, lca_e, lca_converged):
    """
    Runs the CVXPY solver and directly compares it against the provided LCA outputs.
    Returns a dictionary of exact metrics for the streaming loop to log.
    """
    # 1. CVXPY ground truth
    cvx = solve_cvxpy_ground_truth(t_signal, L, alpha, gamma, lamb)
    
    # 2. LCA metrics
    lca_obj = compute_lca_objective(t_signal, lca_x, lca_e, L, alpha, gamma, lamb)
    lca_residual = np.linalg.norm(t_signal - (lca_x + lca_e), ord=2)
    
    # 3. Objective Gap
    obj_gap = abs(cvx['obj'] - lca_obj)
    
    return {
        'cvxpy_converged': cvx['converged'],
        'lca_converged': lca_converged,
        'cvxpy_obj': cvx['obj'],
        'lca_obj': lca_obj,
        'objective_gap': obj_gap,
        'cvxpy_residual': cvx['residual'],
        'lca_residual': lca_residual
    }


# -------------------------------------------------------------------------
# Test to check mathematical stability before plugging into METR-LA
if __name__ == '__main__':

    from lca import lca_ode_solver
    # A 5-node graph
    N_nodes = 5
    np.random.seed(42)
    dummy_t = np.random.randn(N_nodes)
    dummy_adj = np.array([[0,1,0,0,0],[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[0,0,0,1,0]])
    d = dummy_adj.sum(axis=1)
    d_inv_sqrt = np.diag(np.where(d > 0, 1.0 / np.sqrt(d), 0.0))
    dummy_L = np.eye(N_nodes) - d_inv_sqrt @ dummy_adj @ d_inv_sqrt

    # Running the solver
    print("\n--- Running LCA ODE Solver on Dummy Graph ---")
    x_opt, e_opt, steps, is_converged, _ , _ = lca_ode_solver( t_signal = dummy_t, L = dummy_L, alpha = 1.0, lam = 0.1, gamma = 0.5,  tau = 1.0,  dt = 0.001,  max_iters = 50000,  tol = 1e-6)
    
    # Passing the results into the checker
    print("\n--- Running CVXPY Ground Truth ---")
    
    results = run_correctness_check(dummy_t, dummy_L, 1.0, 0.5, 0.1, x_opt, e_opt, is_converged)
    
    print("\n--- Validation against ---")
    print(f"1. Converged Flags   | CVXPY: {results['cvxpy_converged']} | LCA: {results['lca_converged']}")
    print(f"2. Objective Gap     | {results['objective_gap']:.6e}")
    print(f"3. Constraint Resid. | CVXPY: {results['cvxpy_residual']:.6f} | LCA: {results['lca_residual']:.6f}")

    
    # Now on a real METR-LA frame to check correctness on a real graph
    print("\n--- Running on REAL METR-LA graph (Ameer's Priority 1) ---")
    from data import load_traffic_data, load_adjacency_matrix, build_laplacian

    df = load_traffic_data("metr-la.parquet")
    adj_mx = load_adjacency_matrix("adj_mx.pkl")
    L_real = build_laplacian(adj_mx)
    t_real = df.iloc[5000].values.astype(np.float64) / 100.0   # pick any real frame

    x_opt, e_opt, steps, is_converged, _, _ = lca_ode_solver( t_signal=t_real, L=L_real, alpha=1.0, lam=0.1, gamma=0.0, tau=1.0, dt=0.001, max_iters=50000, tol=1e-5)
    results_real = run_correctness_check(t_real, L_real, 1.0, 0.0, 0.1, x_opt, e_opt, is_converged)
    
    print(f"1. Converged Flags   | CVXPY: {results_real['cvxpy_converged']} | LCA: {results_real['lca_converged']}")
    print(f"2. Objective Gap     | {results_real['objective_gap']:.6e}")
    print(f"3. Constraint Resid. | CVXPY: {results_real['cvxpy_residual']:.6f} | LCA: {results_real['lca_residual']:.6f}")