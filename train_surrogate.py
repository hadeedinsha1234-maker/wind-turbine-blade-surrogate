import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ── Load dataset ──
df = pd.read_csv('blade_dataset.csv')

X = df[['Vf', 'theta', 'F']].values
y_deflection = df['tip_deflection'].values
y_stress     = df['max_stress'].values

# ── Split 80/20 ──
X_train, X_test, yd_train, yd_test, ys_train, ys_test = train_test_split(
    X, y_deflection, y_stress, test_size=0.2, random_state=42)

# ── Scale inputs ──
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Log-scale outputs (both span large ranges) ──
yd_train_log = np.log(yd_train)
yd_test_log  = np.log(yd_test)
ys_train_log = np.log(ys_train)
ys_test_log  = np.log(ys_test)

# ── Train GPR for tip deflection ──
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

gpr_deflection = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
gpr_deflection.fit(X_train_s, yd_train_log)

# ── Train GPR for max stress ──
gpr_stress = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
gpr_stress.fit(X_train_s, ys_train_log)

# ── Predict (back to original scale) ──
yd_pred_log, yd_std = gpr_deflection.predict(X_test_s, return_std=True)
ys_pred_log, ys_std = gpr_stress.predict(X_test_s, return_std=True)

yd_pred = np.exp(yd_pred_log)
ys_pred = np.exp(ys_pred_log)

# ── Metrics ──
r2_d   = r2_score(yd_test, yd_pred)
r2_s   = r2_score(ys_test, ys_pred)
rmse_d = np.sqrt(mean_squared_error(yd_test, yd_pred))
rmse_s = np.sqrt(mean_squared_error(ys_test, ys_pred))

print("=" * 50)
print("         GPR SURROGATE RESULTS")
print("=" * 50)
print(f"Tip Deflection  →  R² = {r2_d:.4f}  |  RMSE = {rmse_d:.2e} m")
print(f"Max Stress      →  R² = {r2_s:.4f}  |  RMSE = {rmse_s:.2e} Pa")
print("=" * 50)

# ── Speed comparison ──
import time
start = time.time()
for _ in range(1000):
    gpr_deflection.predict(X_test_s[:1])
elapsed = (time.time() - start) / 1000 * 1000
print(f"\nSurrogate inference time : {elapsed:.3f} ms per query")
print(f"FEM simulation time      : ~60,000 ms per query")
print(f"Speedup                  : ~{int(60000/elapsed)}x faster")

# ── Plot ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("GPR Surrogate vs FEM Ground Truth", fontsize=14, fontweight='bold')

configs = [
    (yd_test, yd_pred, yd_std, "Tip Deflection", "m",   "steelblue"),
    (ys_test, ys_pred, ys_std, "Max Von Mises Stress", "Pa", "darkorange"),
]

for ax, (actual, predicted, std, title, unit, color) in zip(axes, configs):
    # error bars in original scale (approximate)
    err = predicted * std

    ax.errorbar(actual, predicted, yerr=err, fmt='o',
                alpha=0.6, color=color, ecolor='lightgray',
                elinewidth=1, capsize=3, markersize=5)

    lims = [min(actual.min(), predicted.min()) * 0.95,
            max(actual.max(), predicted.max()) * 1.05]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
    ax.set_xlabel(f"FEM (actual) [{unit}]")
    ax.set_ylabel(f"GPR (predicted) [{unit}]")
    ax.set_title(f"{title}\nR² = {r2_score(actual, predicted):.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('surrogate_results.png', dpi=150, bbox_inches='tight')
print("\nPlot saved: surrogate_results.png")