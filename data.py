import pandas as pd
import pickle
import numpy as np

def load_traffic_data(file_path = 'metr-la.parquet'):
    df = pd.read_parquet(file_path)
    return df


def load_adjacency_matrix(file_path = 'adj_mx.pkl'):
    with open(file_path, 'rb') as f:
        _, _, adj_mx = pickle.load(f, encoding='latin1')
    return adj_mx


def build_laplacian(adj_mx):
    W = (adj_mx + adj_mx.T) / 2  # ensure symmetry
    W[W < 0.1] = 0  # threshold small weights

    d = W.sum(axis = 1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)

    L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    L = (L + L.T) / 2  # ensure symmetry

    eigenvalues = np.linalg.eigvalsh(L)
    lam_max = eigenvalues.max()
    rel_tol = lam_max * W.shape[0] * np.finfo(float).eps * 1e4
    n_compo = int((eigenvalues < rel_tol).sum())

    nonzero = eigenvalues[eigenvalues >= rel_tol]
    cond = nonzero.max() / nonzero.min() if len(nonzero) > 0 else np.inf

    print(f"Connected components (zero-eigenvalue multiplicity): {n_compo}")
    print(f"Laplacian condition number (on non-trivial spectrum): {cond:.4f}")
    print(f"Min eigenvalue: {eigenvalues.min():.6e} (should be >= 0 for valid PSD Laplacian)")

    return L


def get_signal_frame(df, t):
    return df.iloc[t].values.astype(np.float64)


Magnitudes = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 16.0]

# --------------------------------Finding Neighbors-----------------------------------
def find_adjacent_nodes(adj_mx, center_node, n_neigh = 8):
    """
    Return up to n_neigh nodes adjacent to the center_node, ranked by edge weight descending.
    Excludes self.
    Returns min(n_neigh, available) nodes.
    """
    weights = adj_mx[center_node].copy()
    weights[center_node] = 0  # exclude self
    neighbors = np.where(weights > 0.1)[0]  # above threshold only
    sorted_idx = np.argsort(weights[neighbors])[::-1]  # strongest first
    return neighbors[sorted_idx][:n_neigh]  # cap at n_neigh


# ----------------------------------- Spike Injection (L1 Lens — Single Sensor) -----------------------------------
def inject_spike(signal, node_idx, magnitude, local_std=None):
    """
    A point anomlay at a single sensor.
    Magnitude is a multiplier of local_std if provided, otherwise treated as an absolute additive value.
    
    Returns:
        corrupted : signal with anomaly injected
        labels : boolean array, true at anomalous nodes
    """
    corrupted = signal.copy()
    scale = local_std[node_idx] if local_std is not None else 1.0
    corrupted[node_idx] += magnitude * scale
    labels = np.zeros(len(signal), dtype = bool)
    labels[node_idx] = True
    return corrupted, labels


# ----------------------------------- Blob Injection (Elastic Net Lens — Correlated Cluster) -----------------------------------
def inject_blob(signal, adj_mx, center_node, magnitude, local_std=None, n_neigh=8):
    """
    Injection of a spatially correlated anomaly at center_node and its strongest neighbours (up to n_neigh).
    All injected nodes receive the same magnitude * local_std additive shift, making the anomaly correlated across the blob.
    
    Returns:
        corrupted : signal with anomaly injected
        labels : boolean array, true at anomalous nodes
        blob_nodes : list of nodes that were actually injected
    """

    blob_nodes = find_adjacent_nodes(adj_mx, center_node, n_neigh)
    blob_nodes = np.append(blob_nodes, center_node)  # including center

    corrupted = signal.copy()
    labels = np.zeros(len(signal), dtype = bool)

    for node in blob_nodes:
        scale = local_std[node] if local_std is not None else 1.0
        corrupted[node] += magnitude * scale
        labels[node] = True

    return corrupted, labels, blob_nodes.tolist()


def calc_local_std(df, window_size = 100):
    std = df.iloc[:window_size].std(axis = 0).values.astype(np.float64)
    std = np.clip(std, 1e-6, None)  # avoid zero std
    return std   # To compute local std for each sensor, using the first `window_size` frames of the data.


#--------------------------Magnitudes Sweep -------------------------------------------
def build_sweep_config(df, adj_mx, frame_idx, magnitudes = Magnitudes, spike_nodes = None, blob_centers = None, n_neigh = 8, seed = 87):
    """
    Configuration for sweeping through different magnitudes of anomalies.
    For each magnitude and each frame, creates one spike config and one blob config with ground truth labels.

    Args:
        df : DataFrame containing the traffic data
        adj_mx : adjacency matrix of the graph
        frame_idx : index of the frame to inject anomalies into
        magnitudes : list of magnitudes to sweep through
        spike_nodes : list of nodes to inject spikes into (if None, all nodes are considered)
        blob_centers : list of nodes to inject blobs into (if None, all nodes are considered)
        n_neigh : number of neighbors for blob injection
        seed : random seed for reproducibility

    Returns:
        configs : a list of configurations, each containing:
                  type, magnitude, frame_idx, clean_signal, corrupted_signal, labels, bolb_nodes (for blob injection)
    """
    rng = np.random.default_rng(seed)
    local_std = calc_local_std(df, window_size=100)
    N = df.shape[1]

    # Fixing of injection nodes before sweep - not against labels
    if spike_nodes is None: spike_nodes = rng.choice(N, size=len(frame_idx), replace=False).tolist()
    if blob_centers is None: blob_centers = rng.choice(N, size=len(frame_idx), replace=False).tolist()

    configs = []

    for mag in magnitudes:
        for i, t in enumerate(frame_idx):
            clean = df.iloc[t].values.astype(np.float64)

            # spike config
            spike_node = spike_nodes[i % len(spike_nodes)]
            corrupted_spike, labels_spike = inject_spike(clean, spike_node, mag, local_std)
            configs.append({
                'type': 'spike',
                'magnitude': mag,
                'frame_idx': t,
                'clean_signal': clean,
                'corrupted_signal': corrupted_spike,
                'labels': labels_spike,
                'injected_nodes': [spike_node]
            })
            
            # blob config
            blob_center = blob_centers[i % len(blob_centers)]
            corrupted_blob, labels_blob, blob_nodes = inject_blob(clean, adj_mx, blob_center, mag, local_std, n_neigh)
            configs.append({
                'type': 'blob',
                'magnitude': mag,
                'frame_idx': t,
                'clean_signal': clean,
                'corrupted_signal': corrupted_blob,
                'labels': labels_blob,
                'injected_nodes': blob_nodes
            })
    
    print(f"Sweep configs built: {len(configs)} total"
          f"({len(magnitudes)} magnitudes x {len(frame_idx)} frames x 2 types)")
    return configs


if __name__ == "__main__":
    traffic_data = load_traffic_data()
    adj_matrix = load_adjacency_matrix()
    laplacian = build_laplacian(adj_matrix)
    frame = get_signal_frame(traffic_data, 0)

    print("Traffic Data Shape:", traffic_data.shape)
    print("Adjacency Matrix Shape:", adj_matrix.shape)
    print("Laplacian Matrix Shape:", laplacian.shape)
    print("Signal Frame Shape:", frame.shape)

