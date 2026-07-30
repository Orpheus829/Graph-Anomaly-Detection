"""
LCA ODE Solver for Graph Anomaly Detection (Joint QP)
Mathematical translation of the Unified Joint QP (Section 03) and LCA ODE Circuit Dynamics (Section 04).

KKT derivation:
    For e: (t - x - γLe) - e = λ∂‖e‖₁  →  e* = T_λ(t - x - γLe)
    For x: -(t - x - e) + αLx = 0        →  τẋ = -(I + αL)x + (t - e)
Both cases use standard soft-threshold. Elastic net effect is via γLe in argument.
"""

import numpy as np
from typing import Tuple, Optional


# ── Thresholding Operator ─────────────────────────────────────────────────
def soft_threshold(u: np.ndarray, lam: float) -> np.ndarray:
    """
    Standard soft-thresholding operator T_lambda(u).
    Proximal operator of L1 norm — used for BOTH γ=0 and γ>0 cases.
    For elastic net (γ>0), the graph coupling appears in the argument as T_λ(t - x - γLe), not as a modification to this function.
    """
    return np.sign(u) * np.maximum(np.abs(u) - lam, 0.0)


# ── Objective Function ────────────────────────────────────────────────────
def compute_objective(x: np.ndarray, e: np.ndarray, t_signal: np.ndarray,  L: np.ndarray, alpha: float, gamma: float, lam: float) -> float:
    """
    UNIFIED JOINT QP objective value.
    min  ½‖t - x - e‖² + (α/2)xᵀLx + λ‖e‖₁ + (γ/2)eᵀLe
    """
    reconstruction = 0.5 * np.sum((t_signal - x - e) ** 2)
    smoothness     = (alpha / 2.0) * float(x @ L @ x)
    sparsity       = lam * np.sum(np.abs(e))
    blob           = (gamma / 2.0) * float(e @ L @ e)
    return reconstruction + smoothness + sparsity + blob


# ── Core LCA ODE Solver ───────────────────────────────────────────────────
def lca_ode_solver(
    t_signal  : np.ndarray,
    L         : np.ndarray,
    alpha     : float,
    lam       : float,
    gamma     : float       = 0.0,
    tau       : float       = 1.0,
    dt        : float       = 0.01,
    max_iters : int         = 2000,
    tol       : float       = 1e-5,
    x_init    : Optional[np.ndarray] = None,
    e_init    : Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int, bool, float, float]:
    """
    Integrates the Joint QP ODEs using forward Euler.

    ODE system :
        τẋ = -(I + αL)x + (t - e)          [background]
        τė = -e + T_λ(t - x - γLe)         [anomaly — soft thresh always]

    Args:
        t_signal  : input signal frame, shape (N,)
        L         : normalized Laplacian, shape (N, N)
        alpha     : smoothness weight on x
        lam       : sparsity threshold lambda
        gamma     : elastic net weight (0.0 = L1 primary case)
        tau       : ODE time constant
        dt        : Euler step size
        max_iters : maximum integration steps
        tol       : convergence tolerance (inf-norm of update step)
                    MUST be identical for warm and cold solves
        x_init    : warm-start for x (None = cold start at zero)
        e_init    : warm-start for e (None = cold start at zero)

    Returns:
        x         : converged background signal
        e         : converged sparse anomaly signal
        iters     : iterations taken
        converged : True if tolerance was met within max_iters
        residual  : ‖t - (x + e)‖₂  (constraint residual)
        objective : final QP objective value (for correctness check)
    """
    N = len(t_signal)

    # Section 06: warm-start or cold-start
    x = x_init.copy() if x_init is not None else np.zeros(N, dtype=np.float64)
    e = e_init.copy() if e_init is not None else np.zeros(N, dtype=np.float64)

    converged = False
    iters     = 0

    for _ in range(max_iters):

        # Background ODE: τẋ = -(I + αL)x + (t - e)
        dx = -x - alpha * (L @ x) + (t_signal - e)

        # Anomaly ODE: τė = -e + T_λ(t - x - γLe)
        # Elastic net coupling is inside the argument — not in the operator
        u_e = t_signal - x - gamma * (L @ e)
        de  = -e + soft_threshold(u_e, lam)

        # Euler step, scaled by τ
        x_new = x + (dt / tau) * dx
        e_new = e + (dt / tau) * de

        # Convergence: inf-norm of combined update
        max_update = max(
            np.max(np.abs(x_new - x)),
            np.max(np.abs(e_new - e))
        )

        x = x_new
        e = e_new
        iters += 1

        if max_update < tol:
            converged = True
            break

    residual  = float(np.linalg.norm(t_signal - (x + e), ord=2))
    objective = compute_objective(x, e, t_signal, L, alpha, gamma, lam)

    return x, e, iters, converged, residual, objective