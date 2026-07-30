# GAD — Graph Anomaly Detection via LCA ODE Dynamics

A graph-signal anomaly detection framework for traffic sensor networks (METR-LA,
207 sensors). The core method solves a **unified joint QP** — a graph-Laplacian
smoothness term plus an L1 sparsity term (with an optional elastic-net
extension) — using a **Locally Competitive Algorithm (LCA)** ODE circuit,
integrated with forward Euler. The project compares this solver against
convex baselines (Graph Total Variation, Matrix Robust PCA) and a graph
deep-learning baseline (DOMINANT), and benchmarks streaming performance via
warm-start vs. cold-start convergence.

## Method summary

For a signal frame `t` over a sensor graph with normalized Laplacian `L`,
the joint objective is:

```
min_{x,e}  ½‖t - x - e‖² + (α/2) xᵀLx + λ‖e‖₁ + (γ/2) eᵀLe
```

- `x` — smooth background traffic signal
- `e` — sparse (or blob-correlated, if γ > 0) anomaly residual
- `α` — Laplacian smoothness weight
- `λ` — sparsity weight
- `γ` — elastic-net / blob-correlation weight (0 = pure L1 case)

`lca.py` solves this via the ODE system:

```
τẋ = -(I + αL)x + (t - e)
τė = -e + T_λ(t - x - γLe)      (T_λ = soft-threshold)
```

## Project structure

```
GAD/
├── config.py          Central hyperparameters, paths, execution toggles
├── data.py             Data/adjacency loading, Laplacian construction,
│                        spike/blob anomaly injection, sweep config builder
├── lca.py              Core LCA ODE solver (joint QP) + objective function
├── baselines.py         Graph Total Variation and Matrix RPCA (cvxpy)
├── dominant.py          DOMINANT (PyGOD) deep-learning baseline — standalone,
│                        kept separate to isolate the torch/PyG dependency stack
├── correctness.py       Validates LCA output against a CVXPY ground-truth
│                        solve (objective gap, residuals) — standalone script
├── streaming.py         Frame-by-frame streaming experiment: warm-start vs.
│                        cold-start iteration counts / convergence / residuals
├── evaluation.py        Metric computation: streaming stats, and AUROC/AP/F1/
│                        precision/recall aggregated per anomaly magnitude
├── plotting.py           Figure generation: streaming plot, magnitude-sweep
│                        curves, baseline-comparison plots
├── run_sweep.py         Orchestrator: builds sweep configs, runs LCA + Graph TV
│                        (+ optional RPCA), evaluates, and plots comparisons
├── visualize.py          Standalone sensor-network layout visualization
├── metr-la.parquet      Raw METR-LA traffic signal data (207 sensors)
├── adj_mx.pkl           Sensor adjacency matrix
├── metr-la_graph.png    Pre-rendered sensor network visualization
└── results/             Generated plots and metric archives (.png / .npz)
```

## Dataset

Experiments were conducted on the **METR-LA** traffic dataset, derived from the
**California Performance Measurement System (PeMS)**. The dataset contains
traffic speed measurements from 207 loop detectors deployed across the Los
Angeles highway network.

The original dataset is **not included** in this repository. Users should
download the dataset from the official or canonical distribution before running
the experiments.

- **PeMS:** https://pems.dot.ca.gov/
- **DCRNN dataset (METR-LA release):** https://github.com/liyaguang/DCRNN

Place the downloaded data files (e.g., `metr-la.parquet` and `adj_mx.pkl`, or
their equivalent formats) in the project root or update the paths in
`config.py`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The DOMINANT baseline (`dominant.py`) needs an additional, heavier
`torch` / `torch_geometric` / `pygod` stack — see the commented-out block
at the bottom of `requirements.txt`.

## Running the pipeline

```bash
# 1. Inspect data + Laplacian sanity checks
python data.py

# visualization of the graph constructed (optional)
python visualize.py

# 2. Core mathematical engine with ODE solver
python lca.py

# 3. Sanity-check the LCA solver against a CVXPY ground truth
python correctness.py

# 4. Run the streaming (warm vs. cold start) benchmark
python streaming.py

# 5. Baselines of Graph TV and Robust PCA
python baselines.py

# 6. Using existing ML model (PyGOD)
python dominant.py

# 7. About the results and plots and validation scores
python evaluation.py
python plotting.py          # renders results/streaming_experiment.png

# 8. Run the full magnitude-sweep comparison (LCA vs. Graph TV [vs. RPCA])
python run_sweep.py


## Configuration (`config.py`)

`config.py` centralizes the project's configuration, including:

- **Dataset paths** – input data and adjacency matrix locations.
- **LCA solver parameters** – `ALPHA`, `LAMBDA`, `GAMMA`, `TAU`, `DT`, `MAX_ITER`, and convergence tolerance.
- **Synthetic anomaly generation** – anomaly magnitude, blob size, and injection settings.
- **Streaming experiment** – number of frames, warm-start/cold-start options, and evaluation settings.
- **Execution flags** – enable or disable optional baselines, correctness verification, and plotting.

Most experiments can be reproduced by modifying only `config.py`, without changing the source code.

All generated figures and metric archives are written to `results/`.

## Configuration (`config.py`)

Key parameters: `ALPHA` / `LAMBDA` / `GAMMA` / `TAU` / `DT` / `MAX_ITER` /
`TOL` control the LCA solver; `MAGNITUDES` and `BLOB_SIZE` control anomaly
injection strength/spread; `N_FRAMES` controls the streaming experiment
length; `RUN_CVXPY_BASELINE` / `RUN_COLD_START_LCA` are execution toggles.

## Known gaps 

- `CORRECTNESS_CHECK_EVERY` / `RUN_CVXPY_BASELINE` are declared in `config.py`
  but not currently read by `streaming.py` — the periodic correctness check
  implied by those settings isn't wired into the streaming loop yet.
- `plotting.plot_magnitude_sweep()` is implemented but not called from
  `run_sweep.py` (no `results/magnitude_sweep.png` is produced).
- `dominant.py` is not yet integrated into `run_sweep.py`'s baseline
  comparison — it runs standalone only.
- `RUN_RPCA_SWEEP = False` by default in `run_sweep.py`, so RPCA is implemented but not exercised in the full sweep unless toggled on.
