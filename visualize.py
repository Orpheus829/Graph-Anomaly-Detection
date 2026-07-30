import networkx as nx
import matplotlib.pyplot as plt

import os

from data import load_adjacency_matrix

def visualize_graph(adj_matrix):
    """Visualizes the graph represented by the adjacency matrix."""
    #print("Visualizing the graph...")
    G = nx.from_numpy_array(adj_matrix)

    plt.figure(figsize=(12, 12))
    plt.axis('off')

    pos = nx.spring_layout(G, seed = 307)  # positions for all nodes
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color='blue', edgecolors='black', alpha=0.6)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
    
    plt.title("METR-LA SENSOR NETWORK (207 NODES)", fontsize=16)

    output_file = 'metr-la_graph.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if os.path.exists(output_file):
        print(f"Graph visualization saved as {output_file}")
    else:
        print("Failed to save the graph visualization.")

    plt.show()


if __name__ == "__main__":
    adj_matrix = load_adjacency_matrix()
    visualize_graph(adj_matrix)