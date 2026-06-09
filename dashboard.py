import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('.')

# ══════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Wind Turbine Blade Surrogate",
    page_icon="🌬️",
    layout="wide",
)

# ══════════════════════════════════════════════
#  LOAD SCALAR MODELS
# ══════════════════════════════════════════════
@st.cache_resource
def load_models():
    with open('trained_models.pkl', 'rb') as f:
        saved = pickle.load(f)
    return saved['models'], saved['scaler']

models, scaler = load_models()

# ══════════════════════════════════════════════
#  LOAD GNN / PI-GNN ON DEMAND
# ══════════════════════════════════════════════
@st.cache_resource
def load_gnn():
    import torch
    from gnn_model import GNNModel
    m = GNNModel()
    m.load_state_dict(torch.load('gnn_final.pt', map_location='cpu'))
    m.eval()
    return m

@st.cache_resource
def load_pignn():
    import torch
    from gnn_model import GNNModel
    m = GNNModel()
    m.load_state_dict(
        torch.load('pignn_final.pt', map_location='cpu'))
    m.eval()
    return m

@st.cache_resource
def load_mesh_graph():
    import torch
    import meshio
    from torch_geometric.data import Data

    mesh     = meshio.read("airfoil.msh")
    points   = mesh.points[:, :2].astype(np.float32)
    pts_min  = points.min(axis=0)
    pts_max  = points.max(axis=0)
    pts_norm = (points - pts_min) / (pts_max - pts_min + 1e-8)

    triangles = None
    for cb in mesh.cells:
        if cb.type == "triangle":
            triangles = cb.data
            break

    edges = set()
    for tri in triangles:
        i, j, k = tri
        for a, b in [(i,j),(j,i),(j,k),(k,j),(i,k),(k,i)]:
            edges.add((a, b))
    edge_index = torch.tensor(
        list(edges), dtype=torch.long).t().contiguous()

    return pts_norm, edge_index

# ══════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════
st.title("🌬️ Wind Turbine Blade Structural Surrogate")
st.markdown("""
**AI-accelerated structural analysis of NACA 0015 composite blades.**
Replace expensive FEM simulations with instant surrogate predictions.
""")

st.markdown("""
| Surrogate Models | Training Simulations | Best R² | Speedup vs FEM |
|:---:|:---:|:---:|:---:|
| **6** | **1,000 FEM runs** | **1.0000** | **~265,000×** |
""")

st.divider()

# ══════════════════════════════════════════════
#  SIDEBAR — INPUTS
# ══════════════════════════════════════════════
st.sidebar.title("🔧 Blade Parameters")
st.sidebar.markdown(
    "Adjust composite material and loading conditions:")

Vf = st.sidebar.slider(
    "Fiber Volume Fraction (Vf)",
    min_value=0.30, max_value=0.70,
    value=0.50, step=0.01,
    help="Fraction of fiber in composite (0.3=30%, 0.7=70%)")

theta = st.sidebar.slider(
    "Fiber Orientation θ (degrees)",
    min_value=0.0, max_value=90.0,
    value=0.0, step=1.0,
    help="Fiber angle relative to blade axis")

F = st.sidebar.slider(
    "Applied Load F (N)",
    min_value=500.0, max_value=5000.0,
    value=1000.0, step=100.0,
    help="Aerodynamic pressure load on blade surface")

model_name = st.sidebar.selectbox(
    "Surrogate Model",
    ['GPR', 'XGBoost', 'MLP', 'Random Forest', 'GNN', 'PI-GNN'],
    index=0,
    help="Select surrogate model for prediction")

st.sidebar.divider()
st.sidebar.markdown("**Material:** E-glass/epoxy composite")
st.sidebar.markdown("**Geometry:** NACA 0015 airfoil")
st.sidebar.markdown("**Chord length:** 1.0 m")

# ══════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════
X_input    = np.array([[Vf, theta, F]])
unc_def    = None
unc_stress = None

if model_name in ['GNN', 'PI-GNN']:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    pts_norm, edge_index = load_mesh_graph()
    active_gnn = load_pignn() if model_name == 'PI-GNN' else load_gnn()

    Vf_n    = (Vf    - 0.3) / (0.7  - 0.3)
    theta_n = theta  / 90.0
    F_n     = (F     - 500) / (5000 - 500)

    params        = np.array(
        [[Vf_n, theta_n, F_n]] * len(pts_norm), dtype=np.float32)
    node_features = np.concatenate([pts_norm, params], axis=1)

    graph = Data(
        x          = torch.tensor(node_features, dtype=torch.float32),
        edge_index = edge_index,
        y          = torch.zeros(1, 2),
        params     = torch.tensor(
            [[Vf_n, theta_n, F_n]], dtype=torch.float32),
    )
    loader = DataLoader([graph], batch_size=1)
    batch  = next(iter(loader))

    start = time.time()
    with torch.no_grad():
        out = active_gnn(batch)
    infer_ms    = (time.time() - start) * 1000
    pred_def    = float(np.exp(out[0, 0].item()))
    pred_stress = float(np.exp(out[0, 1].item()))

else:
    cfg      = models[model_name]
    X_scaled = scaler.transform(X_input) if cfg['scaled'] else X_input

    start = time.time()
    if cfg['log_transform']:
        pred_def    = float(np.exp(
            cfg['deflection'].predict(X_scaled)[0]))
        pred_stress = float(np.exp(
            cfg['stress'].predict(X_scaled)[0]))
    else:
        pred_def    = float(cfg['deflection'].predict(X_scaled)[0])
        pred_stress = float(cfg['stress'].predict(X_scaled)[0])
    infer_ms = (time.time() - start) * 1000

    if model_name == 'GPR':
        _, std_def    = cfg['deflection'].predict(
            X_scaled, return_std=True)
        _, std_stress = cfg['stress'].predict(
            X_scaled, return_std=True)
        unc_def    = float(np.exp(std_def[0])    * pred_def)
        unc_stress = float(np.exp(std_stress[0]) * pred_stress)

safety = 1000 / (pred_stress / 1e6)

# ══════════════════════════════════════════════
#  RESULTS TABLE
# ══════════════════════════════════════════════
st.subheader("📊 Prediction Results")

unc_def_str    = f" ± {unc_def*1000:.3f} mm"    if unc_def    else ""
unc_stress_str = f" ± {unc_stress/1e6:.3f} MPa" if unc_stress else ""
safety_label   = "✅ Safe" if safety > 2 else "⚠️ Warning"

st.markdown(f"""
| Tip Deflection | Max Von Mises Stress | Inference Time | Safety Factor |
|:---:|:---:|:---:|:---:|
| **{pred_def*1000:.3f} mm**{unc_def_str} | **{pred_stress/1e6:.3f} MPa**{unc_stress_str} | **{infer_ms:.2f} ms** (vs ~60,000 ms FEM) | **{safety:.1f}×** {safety_label} |
""")

st.divider()

# ══════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════
col_left, col_right = st.columns(2)

# ── Airfoil visualization ──
with col_left:
    st.subheader("🛩️ NACA 0015 Blade Cross-Section")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    x_c = np.linspace(0, 1, 100)
    yt  = 5*0.15*(0.2969*np.sqrt(x_c)
                  - 0.1260*x_c
                  - 0.3516*x_c**2
                  + 0.2843*x_c**3
                  - 0.1015*x_c**4)

    ax.fill_between(x_c,  yt, -yt, alpha=0.3, color='#2196F3')
    ax.plot(x_c,  yt, color='#2196F3', lw=2, label='NACA 0015')
    ax.plot(x_c, -yt, color='#2196F3', lw=2)

    t_rad = np.radians(theta)
    for xi in [0.2, 0.4, 0.6, 0.8]:
        yi_u = 5*0.15*(0.2969*np.sqrt(xi) - 0.1260*xi
                       - 0.3516*xi**2 + 0.2843*xi**3
                       - 0.1015*xi**4) * 0.5
        ax.annotate('', xy=(xi + 0.08*np.cos(t_rad),
                             yi_u + 0.08*np.sin(t_rad)),
                    xytext=(xi, yi_u),
                    arrowprops=dict(arrowstyle='->',
                                   color='#FF9800', lw=1.5))

    ax.annotate('', xy=(0.5, -yt[50]-0.04),
                xytext=(0.5, -yt[50]-0.12),
                arrowprops=dict(arrowstyle='->',
                                color='#F44336', lw=2))
    ax.text(0.52, -yt[50]-0.08, f'F = {F:.0f} N',
            color='#F44336', fontsize=9)

    for yi in np.linspace(-0.08, 0.08, 6):
        ax.plot([-0.02, -0.06], [yi, yi+0.02],
                color='#9E9E9E', lw=1.5)
    ax.axvline(x=0, color='#9E9E9E', lw=2)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.25, 0.25)
    ax.set_aspect('equal')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.set_xlabel('x/c (normalized chord)', color='white')
    ax.set_ylabel('y/c', color='white')
    ax.legend(loc='upper right', facecolor='#1E1E2E',
              labelcolor='white', fontsize=9)
    ax.set_title(
        f'θ = {theta:.0f}°  |  Vf = {Vf:.2f}  |  F = {F:.0f} N',
        color='white', fontsize=10)

    st.pyplot(fig)
    plt.close()

# ── Parameter sensitivity ──
with col_right:
    st.subheader("📈 Sensitivity to Parameters")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.patch.set_facecolor('#0E1117')

    sweep_configs = [
        ('Vf',    np.linspace(0.3, 0.7, 30), Vf),
        ('theta', np.linspace(0, 90, 30),    theta),
    ]

    for ax, (param, sweep, cur_val) in zip(axes, sweep_configs):
        ax.set_facecolor('#1E1E2E')
        defs     = []
        stresses = []

        for val in sweep:
            Xi = np.array([[val, theta, F]]) if param == 'Vf' \
                 else np.array([[Vf, val, F]])

            if model_name in ['GNN', 'PI-GNN']:
                import torch
                from torch_geometric.data import Data
                from torch_geometric.loader import DataLoader
                Vf_n_s    = (Xi[0,0] - 0.3) / 0.4
                theta_n_s = Xi[0,1] / 90.0
                F_n_s     = (Xi[0,2] - 500) / 4500
                params_s  = np.array(
                    [[Vf_n_s, theta_n_s, F_n_s]] * len(pts_norm),
                    dtype=np.float32)
                nf = np.concatenate([pts_norm, params_s], axis=1)
                g  = Data(
                    x=torch.tensor(nf, dtype=torch.float32),
                    edge_index=edge_index,
                    y=torch.zeros(1, 2),
                    params=torch.tensor(
                        [[Vf_n_s, theta_n_s, F_n_s]],
                        dtype=torch.float32))
                lb = next(iter(DataLoader([g], batch_size=1)))
                with torch.no_grad():
                    o = active_gnn(lb)
                defs.append(float(np.exp(o[0,0].item())) * 1000)
                stresses.append(float(np.exp(o[0,1].item())) / 1e6)
            else:
                Xs = scaler.transform(Xi) if cfg['scaled'] else Xi
                if cfg['log_transform']:
                    d = float(np.exp(
                        cfg['deflection'].predict(Xs)[0]))
                    s = float(np.exp(
                        cfg['stress'].predict(Xs)[0]))
                else:
                    d = float(cfg['deflection'].predict(Xs)[0])
                    s = float(cfg['stress'].predict(Xs)[0])
                defs.append(d * 1000)
                stresses.append(s / 1e6)

        ax2 = ax.twinx()
        ax.plot(sweep,  defs,     color='#2196F3', lw=2,
                label='Deflection (mm)')
        ax2.plot(sweep, stresses, color='#FF9800', lw=2,
                 ls='--', label='Stress (MPa)')
        ax.axvline(x=cur_val, color='white', lw=1, ls=':', alpha=0.7)

        ax.set_xlabel(param, color='white')
        ax.set_ylabel('Deflection (mm)', color='#2196F3')
        ax2.set_ylabel('Stress (MPa)',   color='#FF9800')
        ax.tick_params(colors='white')
        ax2.tick_params(colors='#FF9800')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#333333')
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, labs1+labs2,
                  facecolor='#1E1E2E', labelcolor='white', fontsize=7)

    fig.suptitle(
        'Parameter Sweep (white dotted line = current value)',
        color='white', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ══════════════════════════════════════════════
#  SOBOL' SUMMARY
# ══════════════════════════════════════════════
st.subheader("🎯 Sobol' Sensitivity Analysis Summary")

col_s1, col_s2 = st.columns(2)

with col_s1:
    st.markdown(
        "**Tip Deflection — Key Drivers (First-Order S1)**")
    for param, val in [
        ('F — Applied Load',           0.464),
        ('θ — Fiber Orientation',      0.336),
        ('Vf — Fiber Volume Fraction', 0.091),
    ]:
        st.progress(val, text=f"{param}: S1 = {val:.3f}")

with col_s2:
    st.markdown(
        "**Max Von Mises Stress — Key Drivers (First-Order S1)**")
    for param, val in [
        ('F — Applied Load',           0.893),
        ('θ — Fiber Orientation',      0.086),
        ('Vf — Fiber Volume Fraction', 0.001),
    ]:
        st.progress(val, text=f"{param}: S1 = {val:.3f}")

st.divider()

# ══════════════════════════════════════════════
#  MODEL COMPARISON TABLE
# ══════════════════════════════════════════════
st.subheader("🤖 Surrogate Model Comparison")

import pandas as pd
comparison_df = pd.DataFrame({
    'Model'              : ['GPR', 'XGBoost', 'MLP',
                            'Random Forest', 'PI-GNN', 'GNN'],
    'Type'               : ['Probabilistic', 'Gradient Boosting',
                            'Neural Network', 'Ensemble',
                            'Physics-Informed Graph Neural Network',
                            'Graph Neural Network'],
    'R² Deflection'      : [1.0000, 0.9996, 0.9966,
                            0.9960, 0.9967, 0.9911],
    'R² Stress'          : [1.0000, 1.0000, 0.9805,
                            0.9997, 0.9834, 0.9757],
    'Inference Time (ms)': [0.260, 1.130, 0.240,
                            197.870, 6.490, 5.850],
})
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

st.divider()
st.markdown(
    "Built by **Hadeed** | "
    "NACA 0015 E-glass/epoxy composite | "
    "1,000 FEM simulations | "
    "6 surrogate models | "
    "Sobol' sensitivity analysis"
)