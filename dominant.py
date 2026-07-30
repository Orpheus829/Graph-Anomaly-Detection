"""
DOMINANT (Deep Anomaly Detection on Attributed Networks, Ding et al.) baseline via PyGOD. 
Kept separate from baselines.py so the torch/PyG/PyGOD dependency stack doesn't leak into the lightweight cvxpy baselines.
"""
import numpy as np
import torch
from torch_geometric.data import Data
from pygod.detector import DOMINANT

def build_pyg_graph(adj_mx: np.ndarray, signal: np.ndarray) -> Data:
    """
    Single-snapshot graph: 207 sensor nodes, edges from adjacency matrix,node feature = the (possibly corrupted) scalar reading at this frame.
    """
    W = (adj_mx + adj_mx.T) / 2.0
    W[W < 0.1] = 0
    edge_index = np.array(np.nonzero(W))
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.tensor(signal, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index)

def solve_dominant(adj_mx: np.ndarray, t_signal: np.ndarray, epoch: int = 100, hid_dim: int = 32) -> np.ndarray:
    """
    Trains DOMINANT unsupervised on a single graph snapshot and returns per-node anomaly scores (reconstruction error). No labels touched.
    """
    data = build_pyg_graph(adj_mx, t_signal)
    detector = DOMINANT(hid_dim=hid_dim, epoch=epoch, verbose=0)
    detector.fit(data)
    scores = detector.decision_score_  # per-node anomaly score
    return np.asarray(scores)

def build_signal_window(df, frame_idx: int, corrupted_signal: np.ndarray, w: int = 12) -> np.ndarray:
    """
    Builds a (N, w) feature window ending at frame_idx, using clean history
    for the first w-1 frames and the corrupted signal as the last column.
    Falls back to repeating the corrupted signal if frame_idx < w-1.
    """
    N = corrupted_signal.shape[0]
    start = frame_idx - (w - 1)

    if start < 0:
        # not enough history — pad by repeating the earliest available frame
        history = df.iloc[0:frame_idx].values.astype(np.float64).T  # (N, frame_idx)
        pad_needed = (w - 1) - frame_idx
        if history.shape[1] > 0:
            pad = np.repeat(history[:, :1], pad_needed, axis=1)
            history = np.concatenate([pad, history], axis=1)
        else:
            history = np.zeros((N, w - 1))
    else:
        history = df.iloc[start:frame_idx].values.astype(np.float64).T  # (N, w-1)

    window = np.concatenate([history, corrupted_signal.reshape(N, 1)], axis=1)  # (N, w)
    return window

if __name__ == "__main__":
    from data import load_traffic_data, load_adjacency_matrix, build_sweep_config
    from evaluation import evaluate_single_config  # or whatever your existing AUROC/F1 fn is named
    import config

    df = load_traffic_data(config.DATA_PATH)
    adj_mx = load_adjacency_matrix(config.ADJ_PATH)
    
    rng = np.random.default_rng(config.SEED)
    frame_idx = sorted(rng.choice(np.arange(200, len(df) - 200), size=6, replace=False).tolist())
    configs = build_sweep_config(df, adj_mx, frame_idx=frame_idx, magnitudes=config.MAGNITUDES)

    dominant_results = {}

    for cfg in configs:
        window = build_signal_window(df, cfg['frame_idx'], cfg['corrupted_signal'], w = 12)
        key = (cfg['type'], cfg['magnitude'], cfg['frame_idx'])

        scores = solve_dominant(adj_mx, window, epoch=50)  # per-node anomaly scores
        dominant_results[key] = scores

        labels = cfg['labels']  # 0/1 per node, from injection
        metrics = evaluate_single_config(scores, labels)  # AUROC, F1, precision, recall
        top_nodes = np.argsort(scores)[-5:][::-1]

        print(f"[{cfg['type']} | mag={cfg['magnitude']}]")
        print(f"  AUROC={metrics['auroc']:.3f}  F1={metrics['f1']:.3f}")
        print(f"  top-5 flagged nodes: {top_nodes.tolist()}")
        print(f"  true anomaly nodes:  {np.nonzero(labels)[0].tolist()}")

    