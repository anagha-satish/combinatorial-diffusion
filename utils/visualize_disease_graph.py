#!/usr/bin/env python3
"""
Visualize the disease graph structure.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from environment.disease_graph_loader import load_disease_graph_instance

# Load HIV graph
print("Loading HIV graph...")
G, covariates, theta_unary, theta_pairwise, statuses = load_disease_graph_instance(
    std_name='HIV',
    cc_threshold=100,
    inst_idx=0
)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Infected: {sum(statuses.values())}/{len(statuses)}")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Layout
pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)

# Left plot: Infection status
ax = axes[0]
infected = [i for i in G.nodes() if statuses[i] == 1]
uninfected = [i for i in G.nodes() if statuses[i] == 0]

nx.draw_networkx_nodes(G, pos, nodelist=infected, node_color='red', 
                       node_size=100, alpha=0.8, label='Infected', ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=uninfected, node_color='lightblue', 
                       node_size=50, alpha=0.6, label='Uninfected', ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)

ax.set_title(f'HIV Transmission Network (n={G.number_of_nodes()}, infected={len(infected)})', 
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.axis('off')

# Right plot: Connected components
ax = axes[1]
ccs = list(nx.connected_components(G))
colors = plt.cm.tab10(np.linspace(0, 1, len(ccs)))

for cc_idx, cc_nodes in enumerate(ccs):
    nx.draw_networkx_nodes(G, pos, nodelist=list(cc_nodes), 
                          node_color=[colors[cc_idx]], 
                          node_size=80, alpha=0.8, 
                          label=f'CC{cc_idx+1} ({len(cc_nodes)} nodes)', ax=ax)

nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)
ax.set_title(f'Connected Components (n={len(ccs)})', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.axis('off')

plt.tight_layout()
plt.savefig('disease_graph_visualization.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization to: disease_graph_visualization.png")

# Print statistics
print("\n" + "="*60)
print("GRAPH STATISTICS")
print("="*60)
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Connected components: {len(ccs)}")
print(f"Infected nodes: {sum(statuses.values())}")
print(f"Infection rate: {100*sum(statuses.values())/len(statuses):.1f}%")
print(f"\nComponent sizes: {sorted([len(cc) for cc in ccs], reverse=True)}")
degrees = [G.degree(n) for n in G.nodes()]
print(f"Degree: min={min(degrees)}, mean={np.mean(degrees):.1f}, max={max(degrees)}")
print(f"Is forest: {nx.is_forest(G)}")
print(f"Diameter: {max([nx.diameter(G.subgraph(cc)) for cc in ccs])}")
