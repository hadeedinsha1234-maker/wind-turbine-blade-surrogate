import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import time

# ══════════════════════════════════════════════
#  LOAD GRAPHS
# ══════════════════════════════════════════════
print("Loading graphs...")
with open('graphs.pt', 'rb') as f:
    graphs = pickle.load(f)

print(f"Total graphs: {len(graphs)}")

# ══════════════════════════════════════════════
#  SPLIT
# ══════════════════════════════════════════════
np.random.seed(42)
idx     = np.random.permutation(len(graphs))
n_train = int(0.8 * len(graphs))
n_val   = int(0.1 * len(graphs))

train_graphs = [graphs[i] for i in idx[:n_train]]
val_graphs   = [graphs[i] for i in idx[n_train:n_train+n_val]]
test_graphs  = [graphs[i] for i in idx[n_train+n_val:]]

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_graphs,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_graphs,  batch_size=32, shuffle=False)

print(f"Train: {len(train_graphs)}  Val: {len(val_graphs)}  "
      f"Test: {len(test_graphs)}")

# ══════════════════════════════════════════════
#  GNN MODEL
# ══════════════════════════════════════════════
class GNN(nn.Module):
    def __init__(self, in_channels=5, hidden=64, out_channels=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.conv4 = GCNConv(hidden, hidden)

        self.fc1 = nn.Linear(hidden, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, out_channels)

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.bn4 = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = F.relu(self.bn4(self.conv4(x, edge_index)))

        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# ══════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════
device    = torch.device('cpu')
model     = GNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                             weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5)

print(f"\nModel parameters: "
      f"{sum(p.numel() for p in model.parameters()):,}")
print("Training GNN...\n")

train_losses = []
val_losses   = []
best_val     = float('inf')
best_state   = None
EPOCHS       = 200

for epoch in range(1, EPOCHS + 1):

    # ── Train ──
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = F.mse_loss(pred, batch.y.squeeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    train_loss = total_loss / len(train_graphs)

    # ── Validate ──
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch    = batch.to(device)
            pred     = model(batch)
            val_loss += F.mse_loss(
                pred, batch.y.squeeze(1)).item() * batch.num_graphs
    val_loss /= len(val_graphs)

    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val:
        best_val   = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"Train Loss: {train_loss:.6f}  "
              f"Val Loss: {val_loss:.6f}  "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

# ══════════════════════════════════════════════
#  EVALUATE ON TEST SET
# ══════════════════════════════════════════════
model.load_state_dict(best_state)
model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred  = model(batch)
        all_preds.append(pred.numpy())
        all_labels.append(batch.y.squeeze(1).numpy())

preds  = np.concatenate(all_preds,  axis=0)
labels = np.concatenate(all_labels, axis=0)

pred_def    = np.exp(preds[:, 0])
pred_stress = np.exp(preds[:, 1])
true_def    = np.exp(labels[:, 0])
true_stress = np.exp(labels[:, 1])

r2_d   = r2_score(true_def,    pred_def)
r2_s   = r2_score(true_stress, pred_stress)
rmse_d = np.sqrt(mean_squared_error(true_def,    pred_def))
rmse_s = np.sqrt(mean_squared_error(true_stress, pred_stress))

print("\n" + "=" * 55)
print("         GNN RESULTS — NACA 0015 BLADE")
print("=" * 55)
print(f"Tip Deflection  →  R² = {r2_d:.4f}  RMSE = {rmse_d:.2e} m")
print(f"Max Stress      →  R² = {r2_s:.4f}  RMSE = {rmse_s:.2e} Pa")
print("=" * 55)

# ── Inference speed ──
single_loader = DataLoader([test_graphs[0]], batch_size=1)
single_batch  = next(iter(single_loader))
start = time.time()
for _ in range(100):
    with torch.no_grad():
        model(single_batch.to(device))
infer_ms = (time.time() - start) / 100 * 1000
print(f"Inference time  : {infer_ms:.2f} ms")

# ══════════════════════════════════════════════
#  SAVE MODEL
# ══════════════════════════════════════════════
torch.save(best_state, 'gnn_model.pt')
print("Model saved: gnn_model.pt")

# ══════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("GNN — NACA 0015 Composite Blade",
             fontsize=13, fontweight='bold')

# Loss curve
axes[0].plot(train_losses, label='Train', color='steelblue')
axes[0].plot(val_losses,   label='Val',   color='darkorange')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_yscale('log')
axes[0].set_title('Training Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Deflection
axes[1].scatter(true_def, pred_def, alpha=0.5, s=20, color='steelblue')
lims = [true_def.min()*0.95, true_def.max()*1.05]
axes[1].plot(lims, lims, 'r--', lw=1.5)
axes[1].set_xlabel('FEM (actual) [m]')
axes[1].set_ylabel('GNN (predicted) [m]')
axes[1].set_title(f'Tip Deflection\nR² = {r2_d:.4f}')
axes[1].grid(True, alpha=0.3)

# Stress
axes[2].scatter(true_stress, pred_stress, alpha=0.5, s=20,
                color='darkorange')
lims = [true_stress.min()*0.95, true_stress.max()*1.05]
axes[2].plot(lims, lims, 'r--', lw=1.5)
axes[2].set_xlabel('FEM (actual) [Pa]')
axes[2].set_ylabel('GNN (predicted) [Pa]')
axes[2].set_title(f'Max Stress\nR² = {r2_s:.4f}')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gnn_results.png', dpi=600, bbox_inches='tight')
print("Plot saved: gnn_results.png")
print("\nAll done!")