
# ── Graph ──────────────────────────────────────────────────────────────────
DATA_PATH      = 'metr-la.parquet'
ADJ_PATH       = 'adj_mx.pkl'
ADJ_THRESHOLD  = 0.1        # edge weight threshold for Laplacian
N_SENSORS      = 207

# ── Anomaly Injection ──────────────────────────────────────────────────────
MAGNITUDES     = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 16.0]
BLOB_SIZE      = 8           # max neighbors for blob injection
STD_WINDOW     = 100         # frames used to compute local std
SEED           = 87

# ── LCA ODE ───────────────────────────────────────────────────────────────
ALPHA          = 1.0         # Laplacian smoothness weight on x
LAMBDA         = 0.1         # sparsity penalty on e
GAMMA          = 0.0         # elastic net weight (0 = L1 only, primary case)
TAU            = 1.0         # ODE time constant
DT             = 0.001       # Euler step size
MAX_ITER       = 50000        # maximum ODE iterations per solve
TOL            = 1e-5   # convergence tolerance — same for warm and cold

# ── Streaming ─────────────────────────────────────────────────────────────
N_FRAMES       = 200         # number of frames for streaming experiment

# ── Evaluation ────────────────────────────────────────────────────────────
ANOMALY_THRESHOLD = None     # if None, use AUROC (threshold-free)

# ----Correctness Checks (small-scale)----
CORRECTNESS_CHECK_EVERY = 20   # run cvxpy correctness check every N frames when RUN_CVXPY_BASELINE=True

# ── Execution Toggles ─────────────────────────────────────────────────────
RUN_CVXPY_BASELINE = False   # Set to True only for small correctness checks
RUN_COLD_START_LCA = True    # To compare your warm-start vs. cold-start iterations

# ── Outputs ───────────────────────────────────────────────────────────────
RESULTS_DIR = 'results/'     # Folder to save generated plots and metrics CSVs

import os
os.makedirs(RESULTS_DIR, exist_ok=True)