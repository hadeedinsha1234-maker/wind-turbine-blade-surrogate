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
#  PHYSICS CONSTANTS
# ══════════════════════════════════════════════
# NACA 0015 airfoil geometry
L    = 1.0     # chord length (m)
h    = 0.15    # max thickness (m)
I    = (1/12) * 1.0 * h**3   # second moment of area (m^4)

# Material parameter ranges (for denormalization)
Vf_min, Vf_max       = 0.3, 0.7
theta_min, theta_max = 0.0, 90.0
F_min,  F_max        = 500.0, 5000.0

# E-glass/epoxy fiber and matrix properties
E_fiber  = 72e9
E_matrix = 3.5e9

def denormalize_params(params_norm):
    """Convert normalized [Vf, theta, F] back to physical values."""
    Vf    = params_norm[:, 0] * (Vf_max - Vf_min) + Vf_min
    theta = params_norm[:, 1] * (theta_max - theta_min) + theta_min
    F     = params_norm[:, 2] * (F_max - F_min) + F_min
    return Vf, theta, F

def beam_theory_deflection(Vf, F):
    """
    Cantilever beam tip deflection: δ = F*L³ / (3*E1*I)
    E1 = longitudinal modulus from rule of mixtures
    """
    E1 = Vf * E_fiber + (1 - Vf) * E_matrix
    delta = (F * L**3) / (3.0 * E1 * I)
    return delta

def beam_theory_stress(Vf, F):
    """
    Max bending stress at root: σ = M*c/I
    M = F*L (moment at root), c = h/2 (distance to neutral axis)
    """
    M     = F * L
    c     = h / 2.0
    sigma = (M * c) / I
    return sigma

# ══════════════════════════════════════════════
#  PI-GNN MODEL
# ══════════════════════════════════════════════
class PIGNN(nn.Module):
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

def physics_loss(pred_log, params_norm):
    """
    Energy-based physics loss.
    Enforces beam theory as a soft constraint:
      predicted deflection ≈ F*L³/(3*E1*I)
      predicted stress     ≈ F*L*c/I
    Works in log space since predictions are log-scaled.
    """
    Vf, theta, F = denormalize_params(params_norm)

    # Beam theory ground truth (physics)
    delta_phys = beam_theory_deflection(Vf, F)
    sigma_phys = beam_theory_stress(Vf, F)

    # Convert to log space
    log_delta_phys = torch.log(delta_phys.clamp(min=1e-10))
    log_sigma_phys = torch.log(sigma_phys.clamp(min=1e-10))

    # Residuals: how far are predictions from physics?
    res_deflection = (pred_log[:, 0] - log_delta_phys).pow(2).mean()
    res_stress     = (pred_log[:, 1] - log_sigma_phys).pow(2).mean()

    return res_deflection + res_stress

# ══════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════
device    = torch.device('cpu')
model     = PIGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                             weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5)

print(f"\nModel parameters: "
      f"{sum(p.numel() for p in model.parameters()):,}")
print("Training PI-GNN with beam theory physics loss...\n")

train_losses = []
val_losses   = []
data_losses  = []
phys_losses  = []
best_val     = float('inf')
best_state   = None
EPOCHS       = 200

for epoch in range(1, EPOCHS + 1):

    # Curriculum: gradually increase physics weight
    if epoch <= 50:
        lambda_physics = 0.01
    elif epoch <= 100:
        lambda_physics = 0.05
    else:
        lambda_physics = 0.1

    # ── Train ──
    model.train()
    total_loss = 0
    total_data = 0
    total_phys = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)

        # Data loss (MSE in log space)
        loss_data = F.mse_loss(pred, batch.y.squeeze(1))

        # Physics loss (beam theory residual)
        loss_phys = physics_loss(pred, batch.params.squeeze(1))

        # Combined loss
        loss = loss_data + lambda_physics * loss_phys

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
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"Train: {train_loss:.6f}  "
              f"Val: {val_loss:.6f}  "
              f"Data: {data_loss:.6f}  "
              f"Phys: {phys_loss_:.6f}  "
              f"λ: {lambda_physics}")

# ══════════════════════════════════════════════
#  EVALUATE
# ══════════════════════════════════════════════
model.load_state_dict(best_state)
model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch  = batch.to(device)
        pred   = model(batch)
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
print("       PI-GNN RESULTS — NACA 0015 BLADE")
print("=" * 55)
print(f"Tip Deflection  →  R² = {r2_d:.4f}  RMSE = {rmse_d:.2e} m")
print(f"Max Stress      →  R² = {r2_s:.4f}  RMSE = {rmse_s:.2e} Pa")
print("=" * 55)

# Inference speed
single_loader = DataLoader([test_graphs[0]], batch_size=1)
single_batch  = next(iter(single_loader))
start = time.time()
for _ in range(100):
    with torch.no_grad():
        model(single_batch.to(device))
infer_ms = (time.time() - start) / 100 * 1000
print(f"Inference time  : {infer_ms:.2f} ms")

# ══════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════
torch.save(best_state, 'pignn_model.pt')
print("Model saved: pignn_model.pt")

# ══════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("PI-GNN — NACA 0015 Composite Blade",
             fontsize=13, fontweight='bold')

# Loss curve
axes[0].plot(train_losses, label='Total',   color='steelblue')
axes[0].plot(val_losses,   label='Val',     color='darkorange')
axes[0].plot(data_losses,  label='Data',    color='seagreen',  ls='--')
axes[0].plot(phys_losses,  label='Physics', color='crimson',   ls='--')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_yscale('log')
axes[0].set_title('Training Curve\n(Data + Physics Loss)')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Deflection
axes[1].scatter(true_def, pred_def, alpha=0.5, s=20, color='steelblue')
lims = [true_def.min()*0.95, true_def.max()*1.05]
axes[1].plot(lims, lims, 'r--', lw=1.5)
axes[1].set_xlabel('FEM (actual) [m]')
axes[1].set_ylabel('PI-GNN (predicted) [m]')
axes[1].set_title(f'Tip Deflection\nR² = {r2_d:.4f}')
axes[1].grid(True, alpha=0.3)

# Stress
axes[2].scatter(true_stress, pred_stress, alpha=0.5, s=20,
                color='darkorange')
lims = [true_stress.min()*0.95, true_stress.max()*1.05]
axes[2].plot(lims, lims, 'r--', lw=1.5)
axes[2].set_xlabel('FEM (actual) [Pa]')
axes[2].set_ylabel('PI-GNN (predicted) [Pa]')
axes[2].set_title(f'Max Stress\nR² = {r2_s:.4f}')
axes[2].grid(True, alpha=0.3)

# Data vs Physics loss
axes[3].plot(data_losses,  label='Data loss',    color='seagreen')
axes[3].plot(phys_losses,  label='Physics loss', color='crimson')
axes[3].set_xlabel('Epoch')
axes[3].set_ylabel('Loss')
axes[3].set_yscale('log')
axes[3].set_title('Data vs Physics Loss')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pignn_results.png', dpi=150, bbox_inches='tight')
print("Plot saved: pignn_results.png")
print("\nAll done!")