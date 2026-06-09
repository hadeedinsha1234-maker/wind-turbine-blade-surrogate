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
#  PHYSICS LOSS
# ══════════════════════════════════════════════
F_min, F_max = 500.0, 5000.0

def linear_elasticity_loss(pred, params_norm):
    """
    Linear elasticity constraint:
    For same Vf and theta, outputs scale linearly with F.
    In log space: log(pred_j) - log(pred_i) = log(F_j/F_i)
    """
    Vf_n    = params_norm[:, 0]
    theta_n = params_norm[:, 1]
    F_n     = params_norm[:, 2]

    n = pred.shape[0]
    if n < 2:
        return torch.tensor(0.0)

    residuals = []
    for i in range(n):
        for j in range(i+1, min(i+4, n)):
            dVf    = (Vf_n[i]    - Vf_n[j]).abs()
            dtheta = (theta_n[i] - theta_n[j]).abs()

            if dVf < 0.01 and dtheta < 0.01:
                F_i   = F_n[i] * (F_max - F_min) + F_min
                F_j   = F_n[j] * (F_max - F_min) + F_min
                log_k = torch.log(F_j / F_i)

                res_def    = (pred[j, 0] - pred[i, 0] - log_k).pow(2)
                res_stress = (pred[j, 1] - pred[i, 1] - log_k).pow(2)
                residuals.append(res_def + res_stress)

    if len(residuals) == 0:
        return torch.tensor(0.0)

    return torch.stack(residuals).mean()

# ══════════════════════════════════════════════
#  MODEL — hidden=64, 4 layers
# ══════════════════════════════════════════════
class GNNModel(nn.Module):
    def __init__(self, in_channels=5, hidden=64, out_channels=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.conv4 = GCNConv(hidden, hidden)

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.bn4 = nn.BatchNorm1d(hidden)

        self.fc1     = nn.Linear(hidden, 64)
        self.fc2     = nn.Linear(64, 32)
        self.fc3     = nn.Linear(32, out_channels)
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
#  TRAINING FUNCTION
# ══════════════════════════════════════════════
def train_model(use_physics=False, epochs=300):
    model_name = "PI-GNN" if use_physics else "GNN"
    print(f"\n{'='*55}")
    print(f"  Training {model_name} — {epochs} epochs  "
          f"hidden=64  layers=4")
    print(f"{'='*55}")

    model     = GNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_losses = []
    val_losses   = []
    data_losses  = []
    phys_losses  = []
    best_val     = float('inf')
    best_state   = None

    for epoch in range(1, epochs + 1):

        # Curriculum physics weight
        if epoch <= 50:
            lam = 0.001
        elif epoch <= 150:
            lam = 0.01
        else:
            lam = 0.05

        # ── Train ──
        model.train()
        total_loss = total_data = total_phys = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)

            loss_data = F.mse_loss(pred, batch.y.squeeze(1))

            if use_physics:
                loss_phys = linear_elasticity_loss(
                    pred, batch.params.squeeze(1))
                loss = loss_data + lam * loss_phys
            else:
                loss_phys = torch.tensor(0.0)
                loss      = loss_data

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()      * batch.num_graphs
            total_data += loss_data.item() * batch.num_graphs
            total_phys += loss_phys.item() * batch.num_graphs

        train_loss = total_loss / len(train_graphs)
        data_loss  = total_data / len(train_graphs)
        phys_loss_ = total_phys / len(train_graphs)

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
        data_losses.append(data_loss)
        phys_losses.append(phys_loss_)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"Train: {train_loss:.5f}  "
                  f"Val: {val_loss:.5f}  "
                  f"Data: {data_loss:.5f}  "
                  f"Phys: {phys_loss_:.5f}  "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    # ── Evaluate ──
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

    # Inference speed
    single_loader = DataLoader([test_graphs[0]], batch_size=1)
    single_batch  = next(iter(single_loader))
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            model(single_batch.to(device))
    infer_ms = (time.time() - start) / 100 * 1000

    print(f"\n{model_name} FINAL RESULTS:")
    print(f"  Deflection  R² = {r2_d:.4f}  RMSE = {rmse_d:.2e} m")
    print(f"  Stress      R² = {r2_s:.4f}  RMSE = {rmse_s:.2e} Pa")
    print(f"  Inference   {infer_ms:.2f} ms")

    fname = 'pignn_final.pt' if use_physics else 'gnn_final.pt'
    torch.save(best_state, fname)
    print(f"  Saved: {fname}")

    return {
        'model_name'  : model_name,
        'r2_d'        : r2_d,
        'r2_s'        : r2_s,
        'rmse_d'      : rmse_d,
        'rmse_s'      : rmse_s,
        'infer_ms'    : infer_ms,
        'train_losses': train_losses,
        'val_losses'  : val_losses,
        'data_losses' : data_losses,
        'phys_losses' : phys_losses,
        'pred_def'    : pred_def,
        'pred_stress' : pred_stress,
        'true_def'    : true_def,
        'true_stress' : true_stress,
    }

# ══════════════════════════════════════════════
#  RUN BOTH
# ══════════════════════════════════════════════
device    = torch.device('cpu')
gnn_res   = train_model(use_physics=False, epochs=300)
pignn_res = train_model(use_physics=True,  epochs=300)

# ══════════════════════════════════════════════
#  FINAL COMPARISON TABLE
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"{'Model':<12} {'R²_def':>8} {'R²_str':>8} "
      f"{'RMSE_def':>12} {'RMSE_str':>12} {'ms':>8}")
print("-" * 65)
for r in [gnn_res, pignn_res]:
    print(f"{r['model_name']:<12} {r['r2_d']:>8.4f} {r['r2_s']:>8.4f} "
          f"{r['rmse_d']:>12.2e} {r['rmse_s']:>12.2e} "
          f"{r['infer_ms']:>8.2f}")
print("=" * 65)

# ══════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle("GNN vs PI-GNN — NACA 0015 Composite Blade",
             fontsize=14, fontweight='bold')

colors = {'GNN': 'steelblue', 'PI-GNN': 'darkorange'}

for row, r in enumerate([gnn_res, pignn_res]):
    name  = r['model_name']
    color = colors[name]

    # Loss curve
    axes[row][0].plot(r['train_losses'], label='Train', color=color)
    axes[row][0].plot(r['val_losses'],   label='Val',
                      color='gray', ls='--')
    axes[row][0].set_yscale('log')
    axes[row][0].set_title(f'{name} — Training Curve')
    axes[row][0].set_xlabel('Epoch')
    axes[row][0].set_ylabel('MSE Loss')
    axes[row][0].legend()
    axes[row][0].grid(True, alpha=0.3)

    # Deflection
    axes[row][1].scatter(r['true_def'], r['pred_def'],
                         alpha=0.5, s=15, color=color)
    lims = [r['true_def'].min()*0.95, r['true_def'].max()*1.05]
    axes[row][1].plot(lims, lims, 'r--', lw=1.5)
    axes[row][1].set_title(
        f'{name} — Tip Deflection\nR² = {r["r2_d"]:.4f}')
    axes[row][1].set_xlabel('FEM [m]')
    axes[row][1].set_ylabel('Predicted [m]')
    axes[row][1].grid(True, alpha=0.3)

    # Stress
    axes[row][2].scatter(r['true_stress'], r['pred_stress'],
                         alpha=0.5, s=15, color=color)
    lims = [r['true_stress'].min()*0.95, r['true_stress'].max()*1.05]
    axes[row][2].plot(lims, lims, 'r--', lw=1.5)
    axes[row][2].set_title(
        f'{name} — Max Stress\nR² = {r["r2_s"]:.4f}')
    axes[row][2].set_xlabel('FEM [Pa]')
    axes[row][2].set_ylabel('Predicted [Pa]')
    axes[row][2].grid(True, alpha=0.3)

    # Data vs Physics loss
    axes[row][3].plot(r['data_losses'], label='Data',    color='seagreen')
    axes[row][3].plot(r['phys_losses'], label='Physics', color='crimson')
    axes[row][3].set_yscale('log')
    axes[row][3].set_title(f'{name} — Data vs Physics Loss')
    axes[row][3].set_xlabel('Epoch')
    axes[row][3].set_ylabel('Loss')
    axes[row][3].legend()
    axes[row][3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gnn_pignn_tuned.png', dpi=600, bbox_inches='tight')
print("\nPlot saved: gnn_pignn_tuned.png")
print("\nAll done!")