import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════
#  LOAD & PREPARE DATA
# ══════════════════════════════════════════════
df = pd.read_csv('airfoil_dataset.csv')
print(f"Dataset: {len(df)} rows\n")

X     = df[['Vf', 'theta', 'F']].values
y_def = df['tip_deflection'].values
y_str = df['max_stress'].values

X_train, X_test, yd_train, yd_test, ys_train, ys_test = train_test_split(
    X, y_def, y_str, test_size=0.2, random_state=42)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

yd_train_log = np.log(yd_train)
ys_train_log = np.log(ys_train)

# ══════════════════════════════════════════════
#  DEFINE MODELS
# ══════════════════════════════════════════════
models = {
    'GPR': {
        'deflection': GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=[1.0, 1.0, 1.0],
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5),
            n_restarts_optimizer=10,
            normalize_y=True),
        'stress': GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=[1.0, 1.0, 1.0],
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5),
            n_restarts_optimizer=10,
            normalize_y=True),
        'log_transform': True,
        'scaled': True,
    },
    'Random Forest': {
        'deflection': RandomForestRegressor(
            n_estimators=200, max_depth=None,
            random_state=42, n_jobs=-1),
        'stress': RandomForestRegressor(
            n_estimators=200, max_depth=None,
            random_state=42, n_jobs=-1),
        'log_transform': True,
        'scaled': False,
    },
    'XGBoost': {
        'deflection': XGBRegressor(
            n_estimators=300, max_depth=6,
            learning_rate=0.05, subsample=0.8,
            random_state=42, verbosity=0),
        'stress': XGBRegressor(
            n_estimators=300, max_depth=6,
            learning_rate=0.05, subsample=0.8,
            random_state=42, verbosity=0),
        'log_transform': True,
        'scaled': False,
    },
    'MLP': {
        'deflection': MLPRegressor(
            hidden_layer_sizes=(256, 256, 128, 64),
            activation='relu', max_iter=2000,
            learning_rate_init=0.001,
            random_state=42, early_stopping=True),
        'stress': MLPRegressor(
            hidden_layer_sizes=(256, 256, 128, 64),
            activation='relu', max_iter=2000,
            learning_rate_init=0.001,
            random_state=42, early_stopping=True),
        'log_transform': True,
        'scaled': True,
    },
}

# ══════════════════════════════════════════════
#  TRAIN & EVALUATE
# ══════════════════════════════════════════════
results = {}

for name, cfg in models.items():
    print(f"Training {name}...")

    X_tr = X_train_s if cfg['scaled'] else X_train
    X_te = X_test_s  if cfg['scaled'] else X_test

    # ── Deflection ──
    t0 = time.time()
    if cfg['log_transform']:
        cfg['deflection'].fit(X_tr, yd_train_log)
        yd_pred = np.exp(cfg['deflection'].predict(X_te))
    else:
        cfg['deflection'].fit(X_tr, yd_train)
        yd_pred = cfg['deflection'].predict(X_te)
    train_time = time.time() - t0

    # ── Stress ──
    if cfg['log_transform']:
        cfg['stress'].fit(X_tr, ys_train_log)
        ys_pred = np.exp(cfg['stress'].predict(X_te))
    else:
        cfg['stress'].fit(X_tr, ys_train)
        ys_pred = cfg['stress'].predict(X_te)

    # ── Inference speed ──
    start = time.time()
    for _ in range(1000):
        cfg['deflection'].predict(
            X_te[:1] if cfg['scaled'] else X_test[:1])
    infer_ms = (time.time() - start) / 1000 * 1000

    # ── Metrics ──
    r2_d   = r2_score(yd_test, yd_pred)
    r2_s   = r2_score(ys_test, ys_pred)
    rmse_d = np.sqrt(mean_squared_error(yd_test, yd_pred))
    rmse_s = np.sqrt(mean_squared_error(ys_test, ys_pred))

    results[name] = {
        'r2_d': r2_d, 'r2_s': r2_s,
        'rmse_d': rmse_d, 'rmse_s': rmse_s,
        'infer_ms': infer_ms,
        'train_s': train_time,
        'yd_pred': yd_pred,
        'ys_pred': ys_pred,
    }

    print(f"  Deflection  R² = {r2_d:.4f}  RMSE = {rmse_d:.2e} m")
    print(f"  Stress      R² = {r2_s:.4f}  RMSE = {rmse_s:.2e} Pa")
    print(f"  Inference   {infer_ms:.3f} ms  |  Train {train_time:.1f}s\n")

# ══════════════════════════════════════════════
#  SUMMARY TABLE
# ══════════════════════════════════════════════
print("=" * 75)
print(f"{'Model':<16} {'R²_def':>8} {'R²_str':>8} "
      f"{'RMSE_def':>12} {'RMSE_str':>12} {'Infer(ms)':>10}")
print("-" * 75)
for name, r in results.items():
    print(f"{name:<16} {r['r2_d']:>8.4f} {r['r2_s']:>8.4f} "
          f"{r['rmse_d']:>12.2e} {r['rmse_s']:>12.2e} "
          f"{r['infer_ms']:>10.3f}")
print("=" * 75)

fem_time = 60000
fastest  = min(results, key=lambda k: results[k]['infer_ms'])
speedup  = int(fem_time / results[fastest]['infer_ms'])
print(f"\nFEM simulation time : ~{fem_time:,} ms")
print(f"Fastest surrogate   : {fastest} → "
      f"{results[fastest]['infer_ms']:.3f} ms "
      f"(~{speedup:,}x speedup)")

# ══════════════════════════════════════════════
#  SAVE MODELS
# ══════════════════════════════════════════════
with open('trained_models.pkl', 'wb') as f:
    pickle.dump({'models': models, 'scaler': scaler}, f)
print("\nModels saved: trained_models.pkl")

# ══════════════════════════════════════════════
#  PLOT 1 — Predicted vs Actual
# ══════════════════════════════════════════════
colors = {
    'GPR': 'steelblue',
    'Random Forest': 'seagreen',
    'XGBoost': 'darkorange',
    'MLP': 'mediumpurple'
}

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle(
    "Surrogate Models — NACA 0015 Composite Blade\nPredicted vs Actual",
    fontsize=14, fontweight='bold')

for col, (name, r) in enumerate(results.items()):
    for row, (actual, predicted, target, unit) in enumerate([
        (yd_test, r['yd_pred'], 'Tip Deflection', 'm'),
        (ys_test, r['ys_pred'], 'Max Stress', 'Pa'),
    ]):
        ax = axes[row][col]
        ax.scatter(actual, predicted, alpha=0.4, s=15, color=colors[name])
        lims = [min(actual.min(), predicted.min()) * 0.95,
                max(actual.max(), predicted.max()) * 1.05]
        ax.plot(lims, lims, 'r--', lw=1.5, label='Perfect fit')
        ax.set_title(
            f"{name}\n{target} R²={r2_score(actual, predicted):.4f}",
            fontsize=10)
        ax.set_xlabel(f"FEM [{unit}]", fontsize=8)
        ax.set_ylabel(f"Predicted [{unit}]", fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('all_models_scatter.png', dpi=600, bbox_inches='tight')
print("Plot saved: all_models_scatter.png")

# ══════════════════════════════════════════════
#  PLOT 2 — R² Comparison Bar Chart
# ══════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Model Comparison — NACA 0015 Composite Blade",
             fontsize=13, fontweight='bold')

names = list(results.keys())
clrs  = [colors[n] for n in names]

# R² deflection
r2_ds = [results[n]['r2_d'] for n in names]
bars  = axes[0].bar(names, r2_ds, color=clrs, edgecolor='white')
axes[0].set_ylim(min(r2_ds) * 0.998, 1.001)
axes[0].set_title('Tip Deflection R²')
axes[0].set_ylabel('R²')
axes[0].grid(True, axis='y', alpha=0.3)
for bar, val in zip(bars, r2_ds):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.0001,
                 f'{val:.4f}', ha='center', fontsize=9)

# R² stress
r2_ss = [results[n]['r2_s'] for n in names]
bars  = axes[1].bar(names, r2_ss, color=clrs, edgecolor='white')
axes[1].set_ylim(min(r2_ss) * 0.998, 1.001)
axes[1].set_title('Max Stress R²')
axes[1].set_ylabel('R²')
axes[1].grid(True, axis='y', alpha=0.3)
for bar, val in zip(bars, r2_ss):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.0001,
                 f'{val:.4f}', ha='center', fontsize=9)

# Inference time
infer = [results[n]['infer_ms'] for n in names]
bars  = axes[2].bar(names, infer, color=clrs, edgecolor='white')
axes[2].set_title('Inference Time (ms)')
axes[2].set_ylabel('ms per query')
axes[2].grid(True, axis='y', alpha=0.3)
for bar, val in zip(bars, infer):
    axes[2].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f'{val:.2f}ms', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=600, bbox_inches='tight')
print("Plot saved: model_comparison.png")

print("\nAll done!")