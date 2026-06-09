import numpy as np
import pandas as pd
import meshio
import torch
from torch_geometric.data import Data
import pickle

# ══════════════════════════════════════════════
#  READ MESH
# ══════════════════════════════════════════════
print("Reading airfoil mesh...")
mesh = meshio.read("airfoil.msh")

# Extract nodes (x, y coordinates)
points = mesh.points[:, :2].astype(np.float32)  # shape: [N_nodes, 2]

# Extract triangular elements
triangles = None
for cell_block in mesh.cells:
    if cell_block.type == "triangle":
        triangles = cell_block.data
        break

if triangles is None:
    raise ValueError("No triangles found in mesh!")

print(f"Nodes    : {points.shape[0]}")
print(f"Elements : {triangles.shape[0]}")

# ══════════════════════════════════════════════
#  BUILD EDGE INDEX FROM TRIANGLES
# ══════════════════════════════════════════════
print("Building edge index...")
edges = set()
for tri in triangles:
    i, j, k = tri
    for a, b in [(i,j),(j,i),(j,k),(k,j),(i,k),(k,i)]:
        edges.add((a, b))

edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
print(f"Edges    : {edge_index.shape[1]}")

# ══════════════════════════════════════════════
#  NORMALIZE NODE COORDINATES
# ══════════════════════════════════════════════
pts_min = points.min(axis=0)
pts_max = points.max(axis=0)
points_norm = (points - pts_min) / (pts_max - pts_min + 1e-8)

# ══════════════════════════════════════════════
#  LOAD SIMULATION RESULTS
# ══════════════════════════════════════════════
print("Loading simulation dataset...")
df = pd.read_csv('airfoil_dataset.csv')
print(f"Simulations: {len(df)}")

# ══════════════════════════════════════════════
#  BUILD ONE GRAPH PER SIMULATION
# ══════════════════════════════════════════════
print("Building graphs...")
graphs = []

for idx, row in df.iterrows():
    Vf    = float(row['Vf'])
    theta = float(row['theta'])
    F     = float(row['F'])
    tip_d = float(row['tip_deflection'])
    stress = float(row['max_stress'])

    # Normalize parameters
    Vf_n    = (Vf    - 0.3)  / (0.7  - 0.3)
    theta_n = (theta - 0.0)  / (90.0 - 0.0)
    F_n     = (F     - 500)  / (5000 - 500)

    # Node features: [x, y, Vf, theta, F] for each node
    params = np.array([[Vf_n, theta_n, F_n]] * len(points_norm),
                      dtype=np.float32)
    node_features = np.concatenate([points_norm, params], axis=1)
    # shape: [N_nodes, 5]

    # Normalize targets (log scale)
    y_def    = np.log(tip_d)
    y_stress = np.log(stress)
    y        = torch.tensor([[y_def, y_stress]], dtype=torch.float32)

    graph = Data(
        x          = torch.tensor(node_features, dtype=torch.float32),
        edge_index = edge_index,
        y          = y,
        params     = torch.tensor([[Vf_n, theta_n, F_n]],
                                  dtype=torch.float32),
    )
    graphs.append(graph)

    if (idx + 1) % 100 == 0:
        print(f"  Built {idx+1}/{len(df)} graphs")

# ══════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════
with open('graphs.pt', 'wb') as f:
    pickle.dump(graphs, f)

print(f"\nSaved {len(graphs)} graphs → graphs.pt")
print(f"Node features : {graphs[0].x.shape}")
print(f"Edge index    : {graphs[0].edge_index.shape}")
print(f"Target        : {graphs[0].y.shape}")
print("Done!")